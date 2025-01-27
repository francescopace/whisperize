#!/usr/bin/env python
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from typing import List, Dict, Optional
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

class TranscriptBuffer:
    """Manages transcript history and deduplication with smart merging"""
    def __init__(self, max_history: int = 5, merge_threshold: float = 0.5):
        self.history: List[Dict] = []
        self.max_history = max_history
        self.merge_threshold = merge_threshold
        self.pending_segment: Optional[Dict] = None
    
    def _similarity_ratio(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text segments"""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for comparison"""
        if not isinstance(text, str):
            return ""
        return ' '.join(text.strip().split())

    def _is_duplicate(self, new_text: str, threshold: float = 0.8) -> bool:
        """Check if text is too similar to recent history"""
        new_text = self._clean_text(new_text)
        for entry in self.history:
            if self._similarity_ratio(new_text, self._clean_text(entry["text"])) > threshold:
                return True
        return False
    
    def add_transcript(self, segment: Dict) -> Optional[Dict]:
        """Process and add a transcript segment, with deduplication"""
        if self._is_duplicate(segment["text"]):
            return None
            
        segment["text"] = self._clean_text(segment["text"])
        
        # Add new segment to history
        self.history.append(segment)
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
        return segment

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
    """PyAnnote-based speaker diarization"""
    def __init__(self, auth_token: str, device: str = "cpu"):
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=auth_token
        ).to(torch.device(device))
    
    def process(self, waveform: torch.Tensor, sample_rate: int) -> List[dict]:
        """Process audio for speaker diarization"""
        diarization = self.pipeline({"waveform": waveform, "sample_rate": sample_rate})
        
        return [
            {
                "speaker": label,
                "start": segment.start,
                "end": segment.end
            }
            for segment, _, label in diarization.itertracks(yield_label=True)
        ]

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
        
        self.transcript_buffer = TranscriptBuffer()

    def _init_models(self):
        """Initialize Whisper and diarization models"""
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        logging.info("Loading models...")

        whisper.torch.load = functools.partial(whisper.torch.load, weights_only=True)
        
        self.whisper = whisper.load_model(
            self.config.get("model", "base"), 
            device = device if not self.config.get("whisper_force_cpu", False) else "cpu"
        )
        
        token = self.config.get("huggingface_token")
        if not token:
            raise ValueError("HuggingFace token required in config.json")
        self.diarizer = Diarizer(auth_token=token, device=device)

    def _init_audio_settings(self):
        """Initialize audio processing settings"""
        self.audio_config = {
            "format": pyaudio.paInt16,
            "channels": 1,
            "rate": 16000,
            "chunk": 1024,
            "buffer_duration": 5,
            "buffer_overlap": 2
        }

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
        """Transcribe audio buffer and get speaker information"""
        waveform = torch.from_numpy(audio_buffer.astype(np.float32) / 32768.0).unsqueeze(0)
        timestamp = datetime.now().timestamp()
        
        self.audio_queue.put(AudioChunk(waveform=waveform, timestamp=timestamp))
        speaker, _ = self.speaker.current
        
        result = self.whisper.transcribe(
            audio=waveform.numpy().squeeze(), 
            fp16=self.whisper.device.type != "cpu"
        )
        
        text = result["text"].strip()
        
        if text:
            return [{
                "speaker": speaker or "SPEAKER_UNKNOWN",
                "text": text,
                "start": timestamp,
                "end": timestamp + len(audio_buffer) / self.audio_config["rate"]
            }]
        
        return []

    def process_file(self, audio_path: str) -> None:
        """Process a single audio file"""
        logging.info(f"Processing {audio_path}...")
        
        with wave.open(audio_path, 'rb') as wf:
            audio_buffer = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
            
            if wf.getframerate() != self.audio_config["rate"]:
                logging.warning(f"Sample rate mismatch: {wf.getframerate()} Hz vs {self.audio_config['rate']} Hz")
            
            if wf.getnchannels() == 2:
                audio_buffer = audio_buffer.reshape(-1, 2).mean(axis=1).astype(np.int16)
        
        transcript = self.transcribe(audio_buffer)
        self._write_transcript(transcript)
        
        self.audio_queue.join()
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
            
            buffer = []
            while True:
                data = stream.read(self.audio_config["chunk"], exception_on_overflow=False)
                buffer.extend(np.frombuffer(data, dtype=np.int16))
                
                if len(buffer) >= self.audio_config["buffer_duration"] * self.audio_config["rate"]:
                    transcript = self.transcribe(np.array(buffer))
                    self._write_transcript(transcript)
                    
                    overlap_samples = self.audio_config["buffer_overlap"] * self.audio_config["rate"]
                    buffer = buffer[-overlap_samples:]
                    
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
                processed_segment = self.transcript_buffer.add_transcript(segment)
                if processed_segment:
                    line = f'[{processed_segment["speaker"]}]: {processed_segment["text"]}\n'
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