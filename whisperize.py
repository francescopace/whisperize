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

    def _get_text_end(self, text: str) -> str:
        """Safely get the end of a text string for comparison"""
        if not isinstance(text, str) or not text:
            return ""
        words = text.split()
        if not words:
            return ""
        return words[-1]

    def _is_duplicate(self, new_text: str, threshold: float = 0.8) -> bool:
        """Check if text is too similar to recent history"""
        new_text = self._clean_text(new_text)
        for entry in self.history:
            if self._similarity_ratio(new_text, self._clean_text(entry["text"])) > threshold:
                return True
        return False
        
    def _should_merge_with_pending(self, new_segment: Dict) -> bool:
        """Determine if new segment should be merged with pending segment"""
        if not self.pending_segment or self.pending_segment["speaker"] != new_segment["speaker"]:
            return False
            
        # Check if new segment completes a sentence from pending segment
        pending_text = self._clean_text(self.pending_segment["text"])
        new_text = self._clean_text(new_segment["text"])
        
        if not pending_text or not new_text:
            return False
            
        # Check if pending text ends mid-word or without punctuation
        last_char = self._get_text_end(pending_text)
        ends_incomplete = last_char and not last_char[-1] in '.!?'
        
        # Check if texts overlap
        pending_words = pending_text.split()[-3:] if pending_text else []
        new_words = new_text.split()[:3] if new_text else []
        
        if not pending_words or not new_words:
            return False
            
        overlap_ratio = self._similarity_ratio(
            ' '.join(pending_words),
            ' '.join(new_words)
        )
        
        return ends_incomplete or overlap_ratio > self.merge_threshold
    
    def add_transcript(self, segment: Dict) -> Optional[Dict]:
        """
        Process and potentially add a transcript segment.
        Returns the segment to be written, or None if it should be skipped.
        """
        if self._is_duplicate(segment["text"]):
            return None
            
        # Clean the text
        segment["text"] = self._clean_text(segment["text"])
        
        # If text is too short, add to pending but don't output
        if len(segment["text"].split()) < 3:
            if self.pending_segment:
                # If we already have a pending segment, try to merge
                if self._should_merge_with_pending(segment):
                    self.pending_segment["text"] += " " + segment["text"]
                    return None
            self.pending_segment = segment.copy()
            return None
            
        # Check if we should merge with pending segment
        if self.pending_segment and self._should_merge_with_pending(segment):
            # Merge texts and use the longer timespan
            merged_segment = {
                "speaker": segment["speaker"],
                "text": self._clean_text(self.pending_segment["text"] + " " + segment["text"]),
                "start": min(self.pending_segment["start"], segment["start"]),
                "end": max(self.pending_segment["end"], segment["end"])
            }
            self.pending_segment = None
            
            if not self._is_duplicate(merged_segment["text"]):
                self.history.append(merged_segment)
                if len(self.history) > self.max_history:
                    self.history.pop(0)
                return merged_segment
            return None
            
        # Handle pending segment if it exists and we're not merging
        output_segment = None
        if self.pending_segment:
            if len(self.pending_segment["text"].split()) >= 3:
                output_segment = self.pending_segment
            self.pending_segment = None
            
        # Add new segment to history
        self.history.append(segment)
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
        return segment if not output_segment else output_segment

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
        # Load configuration
        with open(config_path) as f:
            self.config = json.load(f)
            if input_source:
                self.config["input_source"] = input_source

        # Create output directories
        for dir_key in ["output_folder", "temp_folder"]:
            os.makedirs(self.config[dir_key], exist_ok=True)

        # Initialize components
        self._init_models()
        self._init_audio_settings()
        self._init_diarization()
        self._create_transcript_file()
        
        # Initialize transcript buffer for deduplication
        self.transcript_buffer = TranscriptBuffer()

    def _init_models(self):
        """Initialize Whisper and diarization models"""

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

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
            "buffer_duration": 4,    # how often we analyze the audio
            "buffer_overlap": 2      # seconds of overlap between buffers
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
        # Prepare audio for Whisper (normalize to float32 in [-1, 1] range)
        waveform = torch.from_numpy(audio_buffer.astype(np.float32) / 32768.0).unsqueeze(0)
        timestamp = datetime.now().timestamp()
        
        # Queue for diarization
        self.audio_queue.put(AudioChunk(waveform=waveform, timestamp=timestamp))
        
        # Get current speaker
        speaker, _ = self.speaker.current
        
        # Pass the audio tensor directly to Whisper
        result = self.whisper.transcribe(
            audio=waveform.numpy().squeeze(), 
            fp16=self.whisper.device.type != "cpu",
            #language="it"
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
                    # Process the current buffer
                    transcript = self.transcribe(np.array(buffer))
                    self._write_transcript(transcript)
                    
                    # Keep only the overlap portion for the next iteration
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
        """Write transcribed segments to file with smart merging and deduplication"""
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