#!/usr/bin/env python
from dataclasses import dataclass
from datetime import datetime
from typing import List, Generator
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
        self.buffer_size = int(sample_rate * buffer_duration)
        self.buffer = []
        
    def add(self, data: np.ndarray) -> bool:
        self.buffer.extend(data)
        return len(self.buffer) >= self.buffer_size
    
    def get(self) -> np.ndarray:
        data = np.array(self.buffer[:self.buffer_size], dtype=np.int16)
        self.buffer = self.buffer[self.buffer_size:]
        return data
    
@dataclass
class AudioChunk:
    audio_buffer: np.ndarray
    timestamp: float

class AudioProcessor:
    def __init__(self, sample_rate: int, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        
    def process_file(self, file_path: str) -> Generator[np.ndarray, None, None]:
        with wave.open(file_path, 'rb') as wf:
            if wf.getframerate() != self.sample_rate:
                logging.warning(f"Sample rate mismatch: {wf.getframerate()} Hz vs {self.sample_rate} Hz")
            
            chunk_size = int(self.sample_rate)
            while True:
                frames = wf.readframes(chunk_size)
                if not frames:
                    break
                    
                audio_chunk = np.frombuffer(frames, dtype=np.int16)
                if wf.getnchannels() == 2:
                    audio_chunk = audio_chunk.reshape(-1, 2).mean(axis=1).astype(np.int16)
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
        self.running = True

    def run(self):
        while self.running:
            try:
                chunk = self.diarization_queue.get(timeout=1.0)
                if chunk is None:  # Sentinel value
                    break
                results = self.diarizer.process(
                    audio_buffer=chunk.audio_buffer,
                    sample_rate=self.sample_rate
                )
                
                if results:
                    latest = results[-1]
                    self.speaker.update(latest["speaker"], chunk.timestamp)
                
                self.diarization_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Diarization error: {e}")

        # Process any remaining items in the queue
        while not self.diarization_queue.empty():
            try:
                chunk = self.diarization_queue.get_nowait()
                if chunk is None:
                    break
                # Process the chunk (copy the processing code from above)
            except queue.Empty:
                break

    def stop(self):
        self.running = False

class TranscriptionWorker(threading.Thread):
    def __init__(self, whisper: WhisperModel, transcription_queue: queue.Queue, speaker: Speaker, config: dict, write_transcript_func):
        super().__init__()
        self.whisper = whisper
        self.transcription_queue = transcription_queue
        self.speaker = speaker
        self.running = True
        self.last_prompt = None
        self.config = config
        self.write_transcript_func = write_transcript_func

    def run(self):
        while self.running:
            try:
                chunk = self.transcription_queue.get(timeout=1.0)
                if chunk is None:  # Sentinel value
                    break
                segments, _ = self.whisper.transcribe(
                    audio=chunk.audio_buffer,
                    temperature=(0.0, 0.2, 0.4),
                    compression_ratio_threshold=2.4,
                    no_speech_threshold=0.6,
                    condition_on_previous_text=True,
                    initial_prompt=self.last_prompt,
                    word_timestamps=True,
                    beam_size=5,
                    language=self.config.get('language', None)
                )
                
                processed_segments = self.process_segments(segments, chunk.timestamp)
                self.write_transcript_func(processed_segments)
                
                self.transcription_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Transcription error: {e}")

        # Process any remaining items in the queue
        while not self.transcription_queue.empty():
            try:
                chunk = self.transcription_queue.get_nowait()
                if chunk is None:
                    break
                # Process the chunk (copy the processing code from above)
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
        self.running = False

class Whisperize:
    def __init__(self, config_path: str = "config.json", input_source: str = None):
        with open(config_path) as f:
            self.config = json.load(f)
            if input_source:
                self.config["input_source"] = input_source

        os.makedirs(self.config["output_folder"], exist_ok=True)
        self._init_models()
        self._init_audio()
        self._init_transcript()

    def _init_models(self):
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        
        self.whisper = WhisperModel(
            self.config.get("model", "base"),
            device="cpu" if self.config.get("whisper_force_cpu", False) else device,
            compute_type="int8" if self.config.get("whisper_force_cpu", False) or device == "cpu" else "float16"
        )
        
        token = self.config.get("huggingface_token")
        if not token:
            raise ValueError("HuggingFace token required in config.json")
            
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        source = self.config.get("input_source", "microphone")
        source_name = "microphone" if source == "microphone" else os.path.splitext(os.path.basename(source))[0]
        
        self.transcript_path = os.path.join(
            self.config["output_folder"],
            f"transcript_{source_name}_{timestamp}.txt"
        )
        
        with open(self.transcript_path, 'w') as f:
            f.write(f"# Transcript started at {datetime.now()}\n\n")
        
        logging.info(f"Transcript file: {self.transcript_path}")

    def is_silent(self, audio_buffer: np.ndarray, threshold: float = 0.003) -> bool:
        if len(audio_buffer) == 0:
            return True

        float_buffer = audio_buffer.astype(np.float32) / 32768.0
        filtered_buffer = np.diff(float_buffer, prepend=float_buffer[0])
        
        window_size = min(1024, len(filtered_buffer))
        windows = filtered_buffer[:len(filtered_buffer) - (len(filtered_buffer) % window_size)]
        windows = windows.reshape(-1, window_size)
        
        rms_values = np.sqrt(np.mean(windows ** 2, axis=1))
        return not np.any(rms_values > threshold)

    def process_audio_chunk(self, audio_chunk: np.ndarray):
        buffer_timestamp = datetime.now().timestamp()
        audio_data = AudioChunk(audio_buffer=audio_chunk, timestamp=buffer_timestamp)
        self.diarization_queue.put(audio_data)
        self.transcription_queue.put(audio_data)

    def _write_transcript(self, segments: List[dict]) -> None:
        if not segments:
            return

        with open(self.transcript_path, 'a') as f:
            for segment in segments:
                if segment["text"].strip():
                    start_time = datetime.fromtimestamp(segment["start"]).strftime('%H:%M:%S.%f')[:-4]
                    end_time = datetime.fromtimestamp(segment["end"]).strftime('%H:%M:%S.%f')[:-4]
                    line = f'[{start_time}-{end_time}] [{segment["speaker"]}]: {segment["text"]}\n'
                    print(line, end='')
                    f.write(line)

    def process_file(self, audio_path: str) -> None:
        logging.info(f"Processing {audio_path}...")
        
        try:
            for chunk in self.audio_processor.process_file(audio_path):
                if not self.is_silent(chunk):
                    if self.audio_buffer.add(chunk):
                        buffer_data = self.audio_buffer.get()
                        self.process_audio_chunk(buffer_data)
            
            # Add sentinel value to signal end of processing
            self.transcription_queue.put(None)
            self.diarization_queue.put(None)
            
            # Wait for the workers to finish
            self.transcription_worker.join()
            self.diarization_worker.join()
                
        except Exception as e:
            logging.error(f"Error processing file: {e}")
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
        if hasattr(self, 'diarization_worker'):
            self.diarization_worker.stop()
            self.diarization_queue.put(None)  # Add sentinel for diarization worker
            self.diarization_worker.join()
        if hasattr(self, 'transcription_worker'):
            self.transcription_worker.stop()
            self.transcription_queue.put(None)  # Ensure sentinel is added even if stop() is called
            self.transcription_worker.join()
        
        for q in [self.diarization_queue, self.transcription_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

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
