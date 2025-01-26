#!/usr/bin/env python
import json
import wave
import pyaudio
import os
import time
import argparse
import torch
import numpy as np
from typing import List
from pyannote.audio import Pipeline
from lightning_whisper_mlx import LightningWhisperMLX
from datetime import datetime
import logging

# Imposta il livello di logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("speechbrain").setLevel(logging.WARNING)
logging.getLogger("pyannote").setLevel(logging.WARNING)

class Whisperize:
    def __init__(self, config_path="config.json", input_source=None):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            if input_source:
                self.config["input_source"] = input_source
            
        # Create necessary directories
        os.makedirs(os.path.dirname(self.config["output_folder"]), exist_ok=True)
        os.makedirs(os.path.dirname(self.config["temp_folder"]), exist_ok=True)
        
        # Initialize Whisper model optimized for Apple Silicon
        logging.debug("Loading Whisper model...")
        self.whisper_model = LightningWhisperMLX(
            model=self.config.get("model", "tiny"),
            batch_size=self.config.get("batch_size", 12),
            quant=self.config.get("quant", None)
        )
        
        logging.debug("Model loaded successfully")
        
        # Initialize diarizer
        logging.debug("Loading diarization model...")
        auth_token = self.config.get("huggingface_token")
        if not auth_token:
            raise ValueError("Please add your HuggingFace token to config.json")
            
        self.diarizer = Diarizer(auth_token=auth_token)
        logging.debug(f"Diarization model loaded successfully with diarizer: {self.diarizer.__class__.__name__}")
        
        # Audio settings
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        self.buffer_duration = 5
        self.buffer_overlap = 2

        # Create a single transcript file for the session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_source = os.path.splitext(os.path.basename(self.config["input_source"]))[0] # Get filename without extension
        self.transcript_file = os.path.join(
            self.config["output_folder"],
            f"transcript_{input_source}_{timestamp}.txt"
        )
        logging.info(f"Writing transcript to: {self.transcript_file}")
        
    def transcribe_audio(self, audio_buffer):
        # Convert to float32 waveform for pyannote
        waveform = audio_buffer.astype(np.float32) / 32768.0  # Convert from int16 to float32 [-1,1]
        # Convert numpy array to PyTorch tensor
        waveform = torch.from_numpy(waveform)
        waveform = waveform.unsqueeze(0)  # Add batch dimension for pyannote
        
        # Get diarization results directly from buffer
        start_time = time.time()
        diarization_result = self.diarizer.diarize(waveform=waveform, sample_rate=self.rate)
        diarization_duration = time.time() - start_time
        logging.debug(f"Diarization took {diarization_duration:.2f} seconds.")

        # Process each speaker segment
        transcribed_segments = []
        
        # Process segments
        for segment in diarization_result:
            try:
                # Get segment info based on available attributes
                start = float(segment.start)
                end = float(segment.end)
                speaker = getattr(segment, 'speaker', 'SPEAKER')
                
                # Calculate samples for this segment
                start_sample = int(start * self.rate)
                end_sample = int(end * self.rate)
                
                # Ensure we don't exceed buffer boundaries
                if end_sample > len(audio_buffer):
                    end_sample = len(audio_buffer)
                if start_sample >= end_sample:
                    continue
                    
                segment_audio = audio_buffer[start_sample:end_sample]

                # Skip if segment is too short
                if len(segment_audio) < self.rate * 0.1:  # Skip segments shorter than 100ms
                    continue

                # Create temporary file only for whisper (since it requires file input)
                temp_segment_path = os.path.join(self.config["temp_folder"], f"temp_segment_{start}_{end}.wav")
                with wave.open(temp_segment_path, 'wb') as wf:
                    wf.setnchannels(self.channels)
                    wf.setsampwidth(2)  # 16-bit audio
                    wf.setframerate(self.rate)
                    wf.writeframes(segment_audio.tobytes())

                # Transcribe segment
                result = self.whisper_model.transcribe(temp_segment_path)
                transcribed_text = result["text"].strip()
                if transcribed_text:  # Only add if there's actual text
                    transcribed_segments.append({
                        "speaker": speaker,
                        "text": transcribed_text,
                        "start": start,
                        "end": end
                    })
                    
                # Cleanup segment file
                try:
                    os.remove(temp_segment_path)
                except OSError:
                    pass
                    
            except Exception as e:
                logging.error(f"Error processing segment: {e}")
                continue

        return transcribed_segments

    def _write_to_transcript_file(self, content):
        """Write content to a specified transcript file"""
        with open(self.transcript_file, 'a') as f:
            content = ''.join([f'[{entry["speaker"]}]: {entry["text"]}\n' for entry in content])  # Convert list of dicts to string
            if content:
                print(content)
                f.write(content)

    def process_file(self, audio_file):
        """Process a single audio file"""
        logging.info(f"Processing {audio_file}...")
        
        # Load the audio file
        with wave.open(audio_file, 'rb') as wf:
            # Read all frames at once
            frames = wf.readframes(wf.getnframes())
            # Convert to numpy array
            audio_buffer = np.frombuffer(frames, dtype=np.int16)
            
            # Ensure correct sample rate
            if wf.getframerate() != self.rate:
                logging.warning(f"Audio file sample rate ({wf.getframerate()} Hz) differs from expected rate ({self.rate} Hz)")
                # Here you might want to add resampling logic if needed
            
            # Ensure mono
            if wf.getnchannels() == 2:
                # Convert stereo to mono by averaging channels
                audio_buffer = audio_buffer.reshape(-1, 2).mean(axis=1).astype(np.int16)
        
        # Transcribe and get diarization
        transcript = self.transcribe_audio(audio_buffer)
        
        # Save transcript
        self._write_to_transcript_file(transcript)
        return transcript

    def process_microphone(self):
        """Process audio from microphone in real-time"""
        logging.info("Starting real-time transcription...")
        
        # Initialize PyAudio
        audio = pyaudio.PyAudio()
        stream = None
        
        try:
            # Open stream
            stream = audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            
            logging.info("Listening... Press Ctrl+C to stop.")
            
            audio_buffer = []
            while True:
                try:
                    # Read audio data
                    data = stream.read(self.chunk, exception_on_overflow=False)
                    audio_buffer.extend(np.frombuffer(data, dtype=np.int16))
                    
                    # Process every x seconds of audio
                    if len(audio_buffer) >= self.buffer_duration * self.rate:
                        audio_array = np.array(audio_buffer)

                        # Process the audio
                        transcript = self.transcribe_audio(audio_array)
                        
                        # Save and display results
                        self._write_to_transcript_file(transcript)
                        
                        # Keep only the last few seconds for overlap
                        audio_buffer = audio_buffer[-self.buffer_overlap*self.rate:]
                        
                except OSError as e:
                    if e.errno == -9981:  # Input overflow
                        logging.info("Input overflowed. Resetting buffer...")
                        audio_buffer = []
                    else:
                        raise
                    
        except KeyboardInterrupt:
            logging.info("\nStopping...")
        finally:
            # Cleanup
            if stream is not None:
                if stream.is_active():
                    stream.stop_stream()
                stream.close()
            audio.terminate()  

class DiarizationResult:
    """Class representing a diarization segment"""
    def __init__(self, speaker: str, start: float, end: float):
        self.speaker = speaker
        self.start = start
        self.end = end

class TextSegment:
    """Class representing a text segment with timing"""
    def __init__(self, text: str, start: float, end: float):
        self.text = text
        self.start = start
        self.end = end

class Diarizer:
    """PyAnnote-based diarization implementation"""
    
    def __init__(self, auth_token: str):
        """Initialize PyAnnote pipeline"""
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=auth_token
        )
        if torch.backends.mps.is_available():
            self.pipeline.to(torch.device("mps"))
    
    def diarize(self, waveform: torch.Tensor, sample_rate: int) -> List[DiarizationResult]:
        """Perform diarization using PyAnnote"""
        if not self.pipeline:
            raise RuntimeError("Diarizer not initialized. Call initialize() first.")

        audio_dict = {
            "waveform": waveform,
            "sample_rate": sample_rate
        }
        
        # Get diarization results
        diarization = self.pipeline(audio_dict)
        
        # Convert to our format
        results = []
        for segment, _, label in diarization.itertracks(yield_label=True):
            results.append(DiarizationResult(
                speaker=label,
                start=segment.start,
                end=segment.end
            ))
        
        return results

    def create_text_segments(text: str, avg_words_per_second: float = 2.5) -> List[TextSegment]:
        """Create time-aligned text segments from full text"""
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        text_segments = []
        current_time = 0
        
        for sentence in sentences:
            words = len(sentence.split())
            duration = words / avg_words_per_second
            text_segments.append(TextSegment(
                text=sentence,
                start=current_time,
                end=current_time + duration
            ))
            current_time += duration
        
        return text_segments

def main(whisperize):
    logging.info("Starting Whisperize...")
    
    input_source = whisperize.config.get("input_source")
    if input_source == "microphone":
        whisperize.process_microphone()
    else:
        whisperize.process_file(input_source)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Whisperize Application')
    parser.add_argument('--source', type=str, default="microphone", help='Path to the input audio file or use "microphone" for live input')
    args = parser.parse_args()
    whisperize = Whisperize(input_source=args.source)
    main(whisperize)