#!/usr/bin/env python
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Generator
from pyannote.audio import Pipeline
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
import whisper
import functools

# Configure logging
logging.basicConfig(level=logging.INFO)
for logger in ["speechbrain", "pyannote"]:
    logging.getLogger(logger).setLevel(logging.WARNING)

@dataclass
class AudioChunk:
    """Audio chunk with its timestamp for diarization"""
    waveform: torch.Tensor
    timestamp: float

class AudioBuffer:
    """Handles audio buffer processing for both file and stream input"""
    def __init__(self, sample_rate: int, buffer_duration: float = 5.0):
        self.sample_rate = sample_rate
        self.buffer_duration = buffer_duration
        self.buffer_size = int(sample_rate * buffer_duration)
        self.reset()
    
    def reset(self):
        """Reset buffer state"""
        self.buffer = []
        
    def add(self, data: np.ndarray) -> bool:
        """Add data to buffer and return True if buffer is full"""
        self.buffer.extend(data)
        return len(self.buffer) >= self.buffer_size
    
    def get(self) -> np.ndarray:
        """Get current buffer content and reset"""
        data = np.array(self.buffer[:self.buffer_size], dtype=np.int16)
        self.buffer = self.buffer[self.buffer_size:]
        return data

class AudioProcessor:
    """Handles both file and stream audio processing"""
    def __init__(self, sample_rate: int, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        
    def process_file(self, file_path: str) -> Generator[np.ndarray, None, None]:
        """Process audio file in chunks"""
        with wave.open(file_path, 'rb') as wf:
            # Validate audio format
            if wf.getframerate() != self.sample_rate:
                logging.warning(f"Sample rate mismatch: {wf.getframerate()} Hz vs {self.sample_rate} Hz")
            
            chunk_size = int(self.sample_rate)  # 1-second chunks
            while True:
                frames = wf.readframes(chunk_size)
                if not frames:
                    break
                    
                audio_chunk = np.frombuffer(frames, dtype=np.int16)
                if wf.getnchannels() == 2:
                    audio_chunk = audio_chunk.reshape(-1, 2).mean(axis=1).astype(np.int16)
                yield audio_chunk

class Speaker:
    """Thread-safe speaker state management"""
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
    """PyAnnote-based speaker diarization with optimized caching"""
    def __init__(self, auth_token: str, device: str = "cpu"):
        # Set up cache directory
        self.cache_dir = os.path.expanduser("~/.cache/pyannote")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize pipeline with caching
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=auth_token,
            cache_dir=self.cache_dir
        ).to(torch.device(device))
        
        # Store device for potential model movement
        self.device = device
        
        # Minimum duration for processing
        self.min_duration = 0.5  # seconds
        
        # Default parameters for diarization
        self.default_params = {
            "min_speakers": 1,
            "max_speakers": 5
        }
    
    def process(self, waveform: torch.Tensor, sample_rate: int) -> List[dict]:
        """Process audio for speaker diarization with optimizations"""
        # Check if audio is too short
        duration = waveform.shape[1] / sample_rate
        if duration < self.min_duration:
            return [{"speaker": "SPEAKER_00", "start": 0, "end": duration}]
            
        try:
            # Move waveform to correct device if needed
            if waveform.device != self.device:
                waveform = waveform.to(self.device)
            
            # Process with optimized parameters
            diarization = self.pipeline(
                {"waveform": waveform, "sample_rate": sample_rate},
                **self.default_params
            )
            
            return [{
                "speaker": label,
                "start": segment.start,
                "end": segment.end
            } for segment, _, label in diarization.itertracks(yield_label=True)]
            
        except Exception as e:
            logging.error(f"Diarization error: {e}")
            # Fallback to single speaker in case of error
            return [{"speaker": "SPEAKER_00", "start": 0, "end": duration}]

class DiarizationWorker(threading.Thread):
    """Background worker for continuous diarization"""
    def __init__(self, diarizer: Diarizer, audio_queue: queue.Queue, 
                 speaker: Speaker, sample_rate: int):
        super().__init__()
        self.diarizer = diarizer
        self.audio_queue = audio_queue
        self.speaker = speaker
        self.sample_rate = sample_rate
        self.running = True

    def run(self):
        while self.running:
            try:
                chunk = self.audio_queue.get(timeout=1.0)
                results = self.diarizer.process(
                    waveform=chunk.waveform,
                    sample_rate=self.sample_rate
                )
                
                if results:
                    latest = results[-1]
                    self.speaker.update(latest["speaker"], chunk.timestamp)
                
                self.audio_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Diarization error: {e}")

    def stop(self):
        self.running = False

class Whisperize:
    """Main application class for audio transcription and diarization"""
    def __init__(self, config_path: str = "config.json", input_source: Optional[str] = None):
        with open(config_path) as f:
            self.config = json.load(f)
            if input_source:
                self.config["input_source"] = input_source

        os.makedirs(self.config["output_folder"], exist_ok=True)

        self._init_models()
        self._init_audio_settings()
        self._init_diarization()
        self._create_transcript_file()
        self.last_prompt = None

    def _init_models(self):
        """Initialize Whisper and diarization models"""
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        whisper.torch.load = functools.partial(whisper.torch.load, weights_only=True)
        
        self.whisper = whisper.load_model(
            self.config.get("model", "base"), 
            device = device if not self.config.get("whisper_force_cpu", False) else "cpu"
        )
        
        token = self.config.get("huggingface_token")
        if not token:
            raise ValueError("HuggingFace token required in config.json")
            
        # Initialize diarizer
        self.diarizer = Diarizer(
            auth_token=token,
            device=device
        )

    def _init_audio_settings(self):
        """Initialize audio processing settings"""
        self.audio_config = {
            "format": pyaudio.paInt16,
            "channels": 1,
            "rate": 16000,
            "chunk": 1024,
            "buffer_duration": 5,  # 5 seconds buffer
        }
        
        self.audio_buffer = AudioBuffer(
            sample_rate=self.audio_config["rate"],
            buffer_duration=self.audio_config["buffer_duration"]
        )
        
        self.audio_processor = AudioProcessor(
            sample_rate=self.audio_config["rate"],
            channels=self.audio_config["channels"]
        )

    def _init_diarization(self):
        """Initialize diarization components"""
        self.audio_queue = queue.Queue(maxsize=100)
        self.speaker = Speaker()
        self.diarization_worker = DiarizationWorker(
            diarizer=self.diarizer,
            audio_queue=self.audio_queue,
            speaker=self.speaker,
            sample_rate=self.audio_config["rate"]
        )
        self.diarization_worker.start()

    def _create_transcript_file(self):
        """Set up transcript output file"""
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

    def transcribe(self, audio_buffer: np.ndarray) -> List[dict]:
        """Transcribe audio buffer with optimized Whisper parameters"""
        waveform = torch.from_numpy(audio_buffer.astype(np.float32) / 32768.0).unsqueeze(0)
        timestamp = datetime.now().timestamp()
        
        self.audio_queue.put(AudioChunk(waveform=waveform, timestamp=timestamp))
        speaker, _ = self.speaker.current
        
        result = self.whisper.transcribe(
            audio=waveform.numpy().squeeze(),
            temperature=(0.0, 0.2, 0.4),
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=True,
            initial_prompt=self.last_prompt,
            word_timestamps=True,
            fp16=self.whisper.device.type != "cpu",
            language=self.config.get('language', None)
        )
        
        text = result["text"].strip()
        
        if text:
            # Update the last prompt for better context in next transcription
            self.last_prompt = text[-200:] if len(text) > 200 else text
            return [{
                "speaker": speaker or "SPEAKER_UNKNOWN",
                "text": text,
                "start": timestamp,
                "end": timestamp + len(audio_buffer) / self.audio_config["rate"]
            }]
        
        return []

    def process_file(self, audio_path: str) -> None:
        """Process audio file using the same buffering approach as streaming"""
        logging.info(f"Processing {audio_path}...")
        
        try:
            for chunk in self.audio_processor.process_file(audio_path):
                if self.audio_buffer.add(chunk):
                    buffer_data = self.audio_buffer.get()
                    transcript = self.transcribe(buffer_data)
                    self._write_transcript(transcript)
            
            # Process any remaining audio in buffer
            if self.audio_buffer.buffer:
                buffer_data = self.audio_buffer.get()
                transcript = self.transcribe(buffer_data)
                self._write_transcript(transcript)
                
        except Exception as e:
            logging.error(f"Error processing file: {e}")
        finally:
            self.cleanup()

    def process_microphone(self) -> None:
        """Process real-time audio from microphone"""
        audio = pyaudio.PyAudio()
        stream = None
        
        try:
            stream = audio.open(
                format=self.audio_config["format"],
                channels=self.audio_config["channels"],
                rate=self.audio_config["rate"],
                input=True,
                frames_per_buffer=self.audio_config["chunk"]
            )
            
            logging.info("Listening... Press Ctrl+C to stop.")
            
            while True:
                data = stream.read(self.audio_config["chunk"], exception_on_overflow=False)
                chunk = np.frombuffer(data, dtype=np.int16)
                
                if self.audio_buffer.add(chunk):
                    buffer_data = self.audio_buffer.get()
                    transcript = self.transcribe(buffer_data)
                    self._write_transcript(transcript)
                    
        except KeyboardInterrupt:
            logging.info("\nStopping...")
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            audio.terminate()
            self.cleanup()

    def _write_transcript(self, segments: List[dict]) -> None:
        """Write transcribed segments to file"""
        if not segments:
            return

        with open(self.transcript_path, 'a') as f:
            for segment in segments:
                if segment["text"].strip():
                    line = f'[{segment["speaker"]}]: {segment["text"]}\n'
                    print(line, end='')
                    f.write(line)

    def cleanup(self) -> None:
        """Clean up resources"""
        if hasattr(self, 'diarization_worker'):
            self.diarization_worker.stop()
            self.diarization_worker.join()
        
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

def main():
    """Main entry point"""
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