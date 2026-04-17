#!/usr/bin/env python
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Generator, List, Optional
import json
import logging
import numpy as np
import os
import pyaudio
import queue
import sys
import threading
import torch
import wave
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel

# Simplified logging configuration
logging.basicConfig(level=logging.INFO)
for logger in ["speechbrain", "pyannote", "faster_whisper"]:
    logging.getLogger(logger).setLevel(logging.WARNING)

class AudioBuffer:
    def __init__(self, sample_rate: int, buffer_duration: float = 5.0):
        self.sample_rate = sample_rate
        self.buffer_size = max(1, int(sample_rate * buffer_duration))
        self.max_buffer_size = self.buffer_size * 3  # Prevent unbounded growth
        self.buffer = np.zeros(self.max_buffer_size, dtype=np.int16)
        self._read_pos = 0
        self._write_pos = 0
        self._count = 0
        
    def add(self, data: np.ndarray) -> bool:
        samples = np.asarray(data, dtype=np.int16).ravel()
        if samples.size == 0:
            return self._count >= self.buffer_size

        # If incoming chunk is larger than max buffer, keep only the latest part.
        if samples.size >= self.max_buffer_size:
            samples = samples[-self.max_buffer_size:]
            self._read_pos = 0
            self._write_pos = 0
            self._count = 0

        overflow = self._count + samples.size - self.max_buffer_size
        if overflow > 0:
            # Drop oldest samples to preserve newest audio under backpressure.
            self._read_pos = (self._read_pos + overflow) % self.max_buffer_size
            self._count -= overflow
            logging.warning(f"AudioBuffer overflow: trimmed {overflow} samples")

        first_write = min(samples.size, self.max_buffer_size - self._write_pos)
        self.buffer[self._write_pos:self._write_pos + first_write] = samples[:first_write]
        remaining = samples.size - first_write
        if remaining > 0:
            self.buffer[:remaining] = samples[first_write:]

        self._write_pos = (self._write_pos + samples.size) % self.max_buffer_size
        self._count += samples.size
        
        return self._count >= self.buffer_size
    
    def get(self) -> np.ndarray:
        if self._count < self.buffer_size:
            return np.array([], dtype=np.int16)

        data = np.empty(self.buffer_size, dtype=np.int16)
        first_read = min(self.buffer_size, self.max_buffer_size - self._read_pos)
        data[:first_read] = self.buffer[self._read_pos:self._read_pos + first_read]
        remaining = self.buffer_size - first_read
        if remaining > 0:
            data[first_read:] = self.buffer[:remaining]

        self._read_pos = (self._read_pos + self.buffer_size) % self.max_buffer_size
        self._count -= self.buffer_size
        return data
    
@dataclass
class AudioChunk:
    audio_buffer: np.ndarray
    timestamp: float  # Absolute timestamp (epoch time)

class TimestampManager:
    """Manages absolute and relative timestamps for audio processing."""
    def __init__(self):
        self.start_time = datetime.now().timestamp()
        self._lock = threading.Lock()
    
    def get_absolute_timestamp(self) -> float:
        """Get current absolute timestamp (epoch time)."""
        return datetime.now().timestamp()
    
    def get_relative_time(self, absolute_timestamp: float) -> float:
        """Convert absolute timestamp to relative time from start."""
        with self._lock:
            return absolute_timestamp - self.start_time
    
    def format_timestamp(self, timestamp: float, use_relative: bool = False) -> str:
        """Format timestamp for display."""
        if use_relative:
            # Format as HH:MM:SS.mmm for relative time
            hours = int(timestamp // 3600)
            minutes = int((timestamp % 3600) // 60)
            seconds = timestamp % 60
            return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
        else:
            # Format as absolute time
            return datetime.fromtimestamp(timestamp).strftime('%H:%M:%S.%f')[:-3]

class AudioProcessor:
    def __init__(self, sample_rate: int, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        
    def process_file(self, file_path: str) -> Generator[np.ndarray, None, None]:
        with wave.open(file_path, 'rb') as wf:
            chunk_size = int(self.sample_rate)
            while True:
                frames = wf.readframes(chunk_size)
                if not frames:
                    break
                    
                audio_chunk = np.frombuffer(frames, dtype=np.int16)
                if wf.getnchannels() == 2:
                    # Proper stereo to mono conversion avoiding overflow
                    audio_chunk = audio_chunk.reshape(-1, 2)
                    audio_chunk = (audio_chunk[:, 0].astype(np.float32) + audio_chunk[:, 1].astype(np.float32)) / 2.0
                    audio_chunk = audio_chunk.astype(np.int16)
                yield audio_chunk

class Speaker:
    def __init__(self):
        self._lock = threading.Lock()
        self._current = "SPEAKER_00"
        self._timestamp = 0

    def update(self, speaker: str, timestamp: float) -> None:
        with self._lock:
            self._current = speaker
            self._timestamp = timestamp

    @property
    def current(self) -> tuple[str, float]:
        with self._lock:
            return self._current, self._timestamp

class Diarizer:
    def __init__(self, auth_token: str, device: str = "cpu"):
        self.cache_dir = os.path.expanduser("~/.cache/pyannote")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=auth_token,
            cache_dir=self.cache_dir
        ).to(torch.device(device))
        
        self.device = device
        self.min_duration = 0.5
        
    def process(self, audio_buffer: np.ndarray, sample_rate: int) -> List[dict]:
        waveform = torch.from_numpy(audio_buffer.astype(np.float32) / 32768.0).unsqueeze(0).to(self.device)
        duration = waveform.shape[1] / sample_rate
        if duration < self.min_duration:
            return [{"speaker": "SPEAKER_00", "start": 0, "end": duration}]
            
        diarization = self.pipeline(
            {"waveform": waveform, "sample_rate": sample_rate},
            min_speakers=1,
            max_speakers=5
        )
        
        return [{
            "speaker": label,
            "start": segment.start,
            "end": segment.end
        } for segment, _, label in diarization.itertracks(yield_label=True)]

class DiarizationWorker(threading.Thread):
    def __init__(self, diarizer: Diarizer, diarization_queue: queue.Queue, speaker: Speaker, sample_rate: int):
        super().__init__()
        self.diarizer = diarizer
        self.diarization_queue = diarization_queue
        self.speaker = speaker
        self.sample_rate = sample_rate
        self.stop_event = threading.Event()
        self.max_retries = 3
        self.retry_delay = 1.0

    def _process_chunk(self, chunk: AudioChunk, retries: int) -> None:
        for attempt in range(retries):
            try:
                results = self.diarizer.process(
                    audio_buffer=chunk.audio_buffer,
                    sample_rate=self.sample_rate
                )

                if results:
                    latest = results[-1]
                    self.speaker.update(latest["speaker"], chunk.timestamp)
                return

            except Exception as e:
                if attempt < retries - 1:
                    logging.warning(f"Diarization attempt {attempt + 1} failed: {e}. Retrying...")
                    self.stop_event.wait(self.retry_delay)
                else:
                    logging.error(f"Diarization failed after {retries} attempts: {e}")

    def run(self):
        while not self.stop_event.is_set():
            try:
                chunk = self.diarization_queue.get(timeout=1.0)
                if chunk is None:  # Sentinel value
                    break

                self._process_chunk(chunk, retries=self.max_retries)
                self.diarization_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Unexpected diarization error: {e}")

        # Process any remaining items in the queue
        while not self.diarization_queue.empty():
            try:
                chunk = self.diarization_queue.get_nowait()
                if chunk is None:
                    break
                self._process_chunk(chunk, retries=1)
                self.diarization_queue.task_done()
            except queue.Empty:
                break

    def stop(self):
        self.stop_event.set()

class TranscriptionWorker(threading.Thread):
    def __init__(self, whisper: WhisperModel, transcription_queue: queue.Queue, speaker: Speaker, config: dict, write_transcript_func):
        super().__init__()
        self.whisper = whisper
        self.transcription_queue = transcription_queue
        self.speaker = speaker
        self.stop_event = threading.Event()
        self.last_prompt = None
        self.config = config
        self.write_transcript_func = write_transcript_func
        self.max_retries = 3
        self.retry_delay = 1.0
        self.transcribe_options = self._build_transcribe_options()

    def _build_transcribe_options(self) -> dict[str, Any]:
        default_temperature = (0.0, 0.2, 0.4)
        configured_temperature = self.config.get("temperature", default_temperature)
        if isinstance(configured_temperature, (list, tuple)):
            try:
                temperatures = tuple(float(value) for value in configured_temperature)
            except (TypeError, ValueError):
                temperatures = default_temperature
        else:
            temperatures = default_temperature

        beam_size = self.config.get("beam_size", 5)
        try:
            beam_size = int(beam_size)
        except (TypeError, ValueError):
            beam_size = 5

        return {
            "temperature": temperatures,
            "compression_ratio_threshold": 2.4,
            "no_speech_threshold": 0.6,
            "condition_on_previous_text": True,
            "word_timestamps": True,
            "beam_size": max(1, beam_size),
            "language": self.config.get("language")
        }

    def _transcribe(self, audio_buffer: np.ndarray):
        return self.whisper.transcribe(
            audio=audio_buffer,
            initial_prompt=self.last_prompt,
            **self.transcribe_options
        )

    def _process_chunk(self, chunk: AudioChunk, retries: int) -> None:
        for attempt in range(retries):
            try:
                segments, _ = self._transcribe(chunk.audio_buffer)
                processed_segments = self.process_segments(segments, chunk.timestamp)
                self.write_transcript_func(processed_segments)
                return
            except Exception as e:
                if attempt < retries - 1:
                    logging.warning(f"Transcription attempt {attempt + 1} failed: {e}. Retrying...")
                    self.stop_event.wait(self.retry_delay)
                else:
                    logging.error(f"Transcription failed after {retries} attempts: {e}")

    def run(self):
        while not self.stop_event.is_set():
            try:
                chunk = self.transcription_queue.get(timeout=1.0)
                if chunk is None:  # Sentinel value
                    break

                self._process_chunk(chunk, retries=self.max_retries)
                self.transcription_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Unexpected transcription error: {e}")

        # Process any remaining items in the queue
        while not self.transcription_queue.empty():
            try:
                chunk = self.transcription_queue.get_nowait()
                if chunk is None:
                    break
                self._process_chunk(chunk, retries=1)
                self.transcription_queue.task_done()
            except queue.Empty:
                break

    def process_segments(self, segments, buffer_timestamp):
        processed_segments = []
        for segment in segments:
            if not segment.words:
                continue
            
            current_speaker, _ = self.speaker.current
            current_group = {
                "speaker": current_speaker,
                "words": [],
                "start": None,
                "end": None,
                "text": ""
            }
            
            for word in segment.words:
                word_start = buffer_timestamp + word.start
                word_end = buffer_timestamp + word.end
                
                cleaned_word = word.word.strip()
                if cleaned_word:
                    current_group["words"].append({
                        "word": cleaned_word,
                        "start": word_start,
                        "end": word_end,
                        "probability": word.probability
                    })
                    
                    if current_group["start"] is None:
                        current_group["start"] = word_start
                    current_group["end"] = word_end
            
            if current_group["words"]:
                current_group["text"] = " ".join(
                    word_info["word"] for word_info in current_group["words"]
                ).strip()
                processed_segments.append(current_group)
        
        if processed_segments:
            combined_text = " ".join(segment["text"].strip() for segment in processed_segments)
            self.last_prompt = combined_text[-200:] if len(combined_text) > 200 else combined_text
        
        return processed_segments

    def stop(self):
        self.stop_event.set()

class Whisperize:
    def __init__(self, config_path: Optional[str] = None, input_source: str = None):
        resolved_config_path = config_path or (
            "config.local.json" if os.path.exists("config.local.json") else "config.json"
        )

        with open(resolved_config_path) as f:
            self.config = json.load(f)
            if input_source:
                self.config["input_source"] = input_source
        
        # Validate configuration
        self._validate_config()
        
        os.makedirs(self.config["output_folder"], exist_ok=True)
        self.timestamp_manager = TimestampManager()
        self.queue_put_lock = threading.Lock()
        self.transcript_lock = threading.Lock()
        self.json_write_lock = threading.Lock()
        self.transcript_file: Optional[Any] = None
        self.json_segments_path: Optional[str] = None
        self._init_models()
        self._init_audio()
        self._init_transcript()
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        # Validate HuggingFace token from environment.
        config_token = self.config.get("huggingface_token", "")
        token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
        if not token:
            raise ValueError(
                "HuggingFace token not found. "
                "Set HUGGINGFACE_TOKEN (or HF_TOKEN) in your environment."
            )
        if config_token and config_token != "your_huggingface_token_here":
            logging.warning(
                "Ignoring huggingface_token in config file. "
                "Use HUGGINGFACE_TOKEN/HF_TOKEN environment variable instead."
            )
        self.config["huggingface_token"] = token
        
        # Validate buffer_duration
        buffer_duration = self.config.get("buffer_duration")
        if buffer_duration is None:
            self.config["buffer_duration"] = 5.0  # Default value
            logging.warning("buffer_duration not set in config, using default: 5.0 seconds")
        elif not isinstance(buffer_duration, (int, float)) or buffer_duration <= 0:
            raise ValueError(f"Invalid buffer_duration: {buffer_duration}. Must be a positive number.")
        
        # Validate output_folder
        if not self.config.get("output_folder"):
            raise ValueError("output_folder must be specified in config.json")
        
        # Validate output_format
        output_format = self.config.get("output_format", "text")
        if output_format not in ["text", "json"]:
            raise ValueError(f"Invalid output_format: {output_format}. Must be 'text' or 'json'.")

    def _init_models(self):
        default_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        force_cpu = self.config.get("whisper_force_cpu", False)
        device = "cpu" if force_cpu else default_device
        
        self.whisper = WhisperModel(
            self.config.get("model", "base"),
            device=device,
            compute_type="int8" if device == "cpu" else "float16"
        )
        
        token = self.config.get("huggingface_token")
        if not token:
            raise ValueError("HuggingFace token required from environment variables")
            
        self.diarizer = Diarizer(auth_token=token, device=device)
        self.diarization_queue = queue.Queue(maxsize=100)
        self.transcription_queue = queue.Queue(maxsize=100)
        self.speaker = Speaker()
        
        self.diarization_worker = DiarizationWorker(
            diarizer=self.diarizer,
            diarization_queue=self.diarization_queue,
            speaker=self.speaker,
            sample_rate=16000
        )
        self.diarization_worker.start()

        self.transcription_worker = TranscriptionWorker(
            whisper=self.whisper,
            transcription_queue=self.transcription_queue,
            speaker=self.speaker,
            config=self.config,
            write_transcript_func=self._write_transcript
        )
        self.transcription_worker.start()

    def _init_audio(self):
        self.audio_buffer = AudioBuffer(
            sample_rate=16000,
            buffer_duration=self.config.get("buffer_duration")
        )
        self.audio_processor = AudioProcessor(sample_rate=16000)

    def _init_transcript(self):
        """Initialize transcript file with error handling."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        source = self.config.get("input_source", "microphone")
        source_name = "microphone" if source == "microphone" else os.path.splitext(os.path.basename(source))[0]
        
        self.transcript_path = os.path.join(
            self.config["output_folder"],
            f"transcript_{source_name}_{timestamp}.txt"
        )
        
        try:
            self.transcript_file = open(self.transcript_path, 'w', encoding='utf-8', buffering=1)
            self.transcript_file.write(f"# Transcript started at {datetime.now()}\n\n")

            if self.config.get("output_format") == "json":
                self.json_segments_path = self.transcript_path.replace('.txt', '.segments.jsonl')
                with open(self.json_segments_path, 'w', encoding='utf-8'):
                    pass
            logging.info(f"Transcript file: {self.transcript_path}")
        except IOError as e:
            raise IOError(f"Failed to create transcript file at {self.transcript_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error creating transcript file: {e}")

    def is_silent(self, audio_buffer: np.ndarray, threshold: float = 0.003) -> bool:
        if len(audio_buffer) == 0:
            return True

        float_buffer = audio_buffer.astype(np.float32) / 32768.0
        filtered_buffer = np.diff(float_buffer, prepend=float_buffer[0])
        rms = np.sqrt(np.mean(filtered_buffer ** 2))
        return rms <= threshold

    def process_audio_chunk(self, audio_chunk: np.ndarray):
        buffer_timestamp = datetime.now().timestamp()
        audio_data = AudioChunk(audio_buffer=audio_chunk, timestamp=buffer_timestamp)
        
        # Keep diarization/transcription queues aligned.
        with self.queue_put_lock:
            if self.diarization_queue.full() or self.transcription_queue.full():
                logging.warning("Queue full, dropping audio chunk to keep worker queues aligned")
                return
            try:
                self.diarization_queue.put_nowait(audio_data)
                self.transcription_queue.put_nowait(audio_data)
            except queue.Full:
                logging.warning("Queue became full while enqueueing; dropped chunk to avoid desync")

    def _write_transcript(self, segments: List[dict]) -> None:
        """Write transcript in text or JSON format based on config."""
        if not segments:
            return

        output_format = self.config.get("output_format", "text")

        with self.transcript_lock:
            if not self.transcript_file:
                return

            for segment in segments:
                if segment["text"].strip():
                    # Calculate relative time for better readability
                    relative_start = self.timestamp_manager.get_relative_time(segment["start"])
                    relative_end = self.timestamp_manager.get_relative_time(segment["end"])

                    start_time = self.timestamp_manager.format_timestamp(relative_start, use_relative=True)
                    end_time = self.timestamp_manager.format_timestamp(relative_end, use_relative=True)

                    line = f'[{start_time}-{end_time}] [{segment["speaker"]}]: {segment["text"]}\n'
                    print(line, end='')
                    self.transcript_file.write(line)

        if output_format == "json" and self.json_segments_path:
            with self.json_write_lock:
                with open(self.json_segments_path, 'a', encoding='utf-8') as f:
                    for segment in segments:
                        if segment["text"].strip():
                            f.write(json.dumps(self._build_json_segment(segment), ensure_ascii=False))
                            f.write("\n")

    def _build_json_segment(self, segment: dict) -> dict:
        relative_start = self.timestamp_manager.get_relative_time(segment["start"])
        relative_end = self.timestamp_manager.get_relative_time(segment["end"])
        return {
            "speaker": segment["speaker"],
            "start": round(relative_start, 3),
            "end": round(relative_end, 3),
            "text": segment["text"],
            "words": [
                {
                    "word": word["word"],
                    "start": round(self.timestamp_manager.get_relative_time(word["start"]), 3),
                    "end": round(self.timestamp_manager.get_relative_time(word["end"]), 3),
                    "probability": round(word["probability"], 3)
                }
                for word in segment.get("words", [])
            ]
        }
    
    def _save_json_transcript(self) -> None:
        """Save transcript in JSON format."""
        if not self.json_segments_path or not os.path.exists(self.json_segments_path):
            return
        
        json_path = self.transcript_path.replace('.txt', '.json')

        metadata = {
            "start_time": datetime.fromtimestamp(self.timestamp_manager.start_time).isoformat(),
            "duration": self.timestamp_manager.get_relative_time(datetime.now().timestamp()),
            "model": self.config.get("model", "base"),
            "language": self.config.get("language", "auto"),
            "source": self.config.get("input_source", "microphone")
        }

        with self.json_write_lock:
            with open(json_path, 'w', encoding='utf-8') as out:
                out.write("{\n")
                out.write('  "metadata": ')
                out.write(json.dumps(metadata, ensure_ascii=False, indent=2).replace("\n", "\n  "))
                out.write(",\n")
                out.write('  "segments": [\n')

                first = True
                with open(self.json_segments_path, 'r', encoding='utf-8') as segments_file:
                    for line in segments_file:
                        payload = line.strip()
                        if not payload:
                            continue
                        if not first:
                            out.write(",\n")
                        out.write("    ")
                        out.write(payload)
                        first = False

                out.write("\n  ]\n")
                out.write("}\n")
        
        logging.info(f"JSON transcript saved to: {json_path}")

        try:
            os.remove(self.json_segments_path)
        except OSError:
            logging.warning(f"Could not remove temporary JSON segments file: {self.json_segments_path}")

    def process_file(self, audio_path: str) -> None:
        """Process audio file with validation."""
        # Validate audio file
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if not os.path.isfile(audio_path):
            raise ValueError(f"Path is not a file: {audio_path}")
        
        # Check if it's a valid WAV file
        try:
            with wave.open(audio_path, 'rb') as wf:
                # Basic validation
                if wf.getnchannels() not in [1, 2]:
                    raise ValueError(f"Unsupported number of channels: {wf.getnchannels()}")
                if wf.getsampwidth() != 2:  # 16-bit
                    raise ValueError(f"Unsupported sample width: {wf.getsampwidth()} bytes (expected 2)")
                if wf.getframerate() != 16000:
                    raise ValueError(
                        f"Unsupported sample rate: {wf.getframerate()} Hz (expected 16000 Hz)."
                    )
        except wave.Error as e:
            raise ValueError(f"Invalid WAV file: {e}")
        
        logging.info(f"Processing {audio_path}...")
        
        try:
            for chunk in self.audio_processor.process_file(audio_path):
                if not self.is_silent(chunk):
                    if self.audio_buffer.add(chunk):
                        buffer_data = self.audio_buffer.get()
                        self.process_audio_chunk(buffer_data)
            
            # Add sentinel value to signal end of processing
            self._try_put_sentinel(self.transcription_queue, "transcription")
            self._try_put_sentinel(self.diarization_queue, "diarization")
            
            # Wait for the workers to finish with timeout
            timeout = 30.0
            self.transcription_worker.join(timeout=timeout)
            if self.transcription_worker.is_alive():
                logging.warning("Transcription worker did not finish within timeout")
            
            self.diarization_worker.join(timeout=timeout)
            if self.diarization_worker.is_alive():
                logging.warning("Diarization worker did not finish within timeout")
                
        except Exception as e:
            logging.error(f"Error processing file: {e}")
            raise
        finally:
            self.cleanup()

    def process_microphone(self) -> None:
        audio = pyaudio.PyAudio()
        stream = None
        
        try:
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=1024
            )
            
            logging.info("Listening... Press Ctrl+C to stop.")
            
            while True:
                data = stream.read(1024, exception_on_overflow=False)
                chunk = np.frombuffer(data, dtype=np.int16)
                
                if not self.is_silent(chunk):
                    if self.audio_buffer.add(chunk):
                        buffer_data = self.audio_buffer.get()
                        self.process_audio_chunk(buffer_data)
                    
        except KeyboardInterrupt:
            logging.info("\nStopping...")
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            audio.terminate()
            self.cleanup()

    def cleanup(self) -> None:
        """Gracefully shutdown worker threads with timeout."""
        timeout = 10.0  # seconds
        workers_still_running = False
        
        if hasattr(self, 'diarization_worker') and self.diarization_worker.is_alive():
            self.diarization_worker.stop()
            self._try_put_sentinel(self.diarization_queue, "diarization")
            self.diarization_worker.join(timeout=timeout)
            if self.diarization_worker.is_alive():
                logging.warning("Diarization worker did not terminate within timeout")
                workers_still_running = True
        
        if hasattr(self, 'transcription_worker') and self.transcription_worker.is_alive():
            self.transcription_worker.stop()
            self._try_put_sentinel(self.transcription_queue, "transcription")
            self.transcription_worker.join(timeout=timeout)
            if self.transcription_worker.is_alive():
                logging.warning("Transcription worker did not terminate within timeout")
                workers_still_running = True
        
        for q in [self.diarization_queue, self.transcription_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        
        # Save JSON transcript if configured
        if self.config.get("output_format") == "json":
            if workers_still_running:
                logging.warning("Skipping JSON export because workers are still running")
            else:
                self._save_json_transcript()

        if self.transcript_file:
            if workers_still_running:
                self.transcript_file.flush()
            else:
                self.transcript_file.close()
                self.transcript_file = None

    def _try_put_sentinel(self, target_queue: queue.Queue, queue_name: str) -> None:
        try:
            target_queue.put(None, timeout=0.2)
        except queue.Full:
            logging.warning(
                f"{queue_name.capitalize()} queue full during shutdown; "
                "relying on stop event to terminate worker"
            )

def main():
    input_source = sys.argv[1] if len(sys.argv) > 1 else "microphone"
    
    try:
        app = Whisperize(input_source=input_source)
        if input_source == "microphone":
            app.process_microphone()
        else:
            app.process_file(input_source)
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
