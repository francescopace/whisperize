#!/usr/bin/env python
import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Generator, List, Optional
import inspect
import json
import logging
import numpy as np
import os
import warnings
import pyaudio
import queue
import sys
import threading
import torch
import wave

# pyannote 4.x can emit a very verbose warning when torchcodec/ffmpeg
# integration is incomplete. Keep logs focused on actionable errors.
warnings.filterwarnings(
    "ignore",
    message=r".*torchcodec is not installed correctly so built-in audio decoding will fail.*",
    category=UserWarning
)
# pyannote internals can emit these NumPy warnings on short/edge segments.
# They are noisy and do not change transcription/diarization outputs.
warnings.filterwarnings(
    "ignore",
    message=r".*Mean of empty slice.*",
    category=RuntimeWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*invalid value encountered in divide.*",
    category=RuntimeWarning,
)

# Simplified logging configuration
logging.basicConfig(level=logging.INFO)
for logger in ["speechbrain", "pyannote"]:
    logging.getLogger(logger).setLevel(logging.WARNING)

DEFAULT_CONFIG = {
    "output_folder": "transcripts/",
    "output_format": "text",
    "model": "turbo",
    "language": None,
    "buffer_duration": 4.0,
    "temperature": [0.0, 0.2, 0.4],
    "model_cache_dir": ".model_cache",
    "diarization_min_speakers": None,
    "diarization_max_speakers": None,
}

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


@dataclass
class TranscribedWord:
    word: str
    start: float
    end: float
    probability: float


@dataclass
class TranscribedSegment:
    text: str
    words: List[TranscribedWord]
    start: Optional[float] = None
    end: Optional[float] = None


class MlxWhisperTranscriber:
    def __init__(self, model_repo: str, module: Any):
        self.model_repo = model_repo
        self.module = module

    def transcribe(self, audio_buffer: np.ndarray, initial_prompt: Optional[str], options: dict[str, Any]) -> List[TranscribedSegment]:
        audio_float = np.asarray(audio_buffer, dtype=np.float32) / 32768.0
        transcribe_kwargs = {
            "path_or_hf_repo": self.model_repo,
            "language": options.get("language"),
            "initial_prompt": initial_prompt,
            "word_timestamps": True,
            "temperature": options.get("temperature", (0.0, 0.2, 0.4)),
            "condition_on_previous_text": options.get("condition_on_previous_text", True),
            "no_speech_threshold": options.get("no_speech_threshold", 0.6),
            "compression_ratio_threshold": options.get("compression_ratio_threshold", 2.4),
        }

        result = self.module.transcribe(audio_float, **transcribe_kwargs)
        return _normalize_segments_from_dict(result.get("segments", []))


def _normalize_segments_from_dict(segments: List[dict]) -> List[TranscribedSegment]:
    normalized_segments: List[TranscribedSegment] = []
    for segment in segments:
        words: List[TranscribedWord] = []
        for word in segment.get("words", []):
            cleaned = (word.get("word", "") or "").strip()
            if not cleaned:
                continue
            words.append(
                TranscribedWord(
                    word=cleaned,
                    start=float(word.get("start", 0.0)),
                    end=float(word.get("end", 0.0)),
                    probability=float(word.get("probability", word.get("confidence", 0.0)) or 0.0),
                )
            )
        normalized_segments.append(
            TranscribedSegment(
                text=(segment.get("text", "") or "").strip(),
                words=words,
                start=float(segment.get("start", 0.0) or 0.0),
                end=float(segment.get("end", 0.0) or 0.0),
            )
        )
    return normalized_segments

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
        self._segments: list[tuple[float, float, str]] = []
        self._history_seconds = 300.0

    def update(self, speaker: str, timestamp: float) -> None:
        with self._lock:
            self._current = speaker
            self._timestamp = timestamp

    def add_diarization_segments(self, segments: List[dict], chunk_timestamp: float) -> None:
        with self._lock:
            for segment in segments:
                speaker = str(segment.get("speaker", "SPEAKER_00"))
                rel_start = float(segment.get("start", 0.0) or 0.0)
                rel_end = float(segment.get("end", rel_start) or rel_start)
                abs_start = chunk_timestamp + rel_start
                abs_end = chunk_timestamp + rel_end
                if abs_end < abs_start:
                    abs_start, abs_end = abs_end, abs_start

                self._segments.append((abs_start, abs_end, speaker))
                if abs_end >= self._timestamp:
                    self._current = speaker
                    self._timestamp = abs_end

            cutoff = chunk_timestamp - self._history_seconds
            self._segments = [seg for seg in self._segments if seg[1] >= cutoff]

    def get_speaker_for_timestamp(self, timestamp: float) -> str:
        with self._lock:
            for start, end, speaker in reversed(self._segments):
                if start <= timestamp <= end:
                    return speaker
            return self._current

    @property
    def current(self) -> tuple[str, float]:
        with self._lock:
            return self._current, self._timestamp

class Diarizer:
    def __init__(
        self,
        auth_token: Optional[str],
        device: str = "cpu",
        cache_dir: Optional[str] = None,
        pipeline_source: str = "pyannote/speaker-diarization-3.1",
        local_files_only: bool = False,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ):
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/pyannote")
        os.makedirs(self.cache_dir, exist_ok=True)

        from_pretrained_kwargs = {
            "token": auth_token,
            "cache_dir": self.cache_dir,
        }
        if local_files_only:
            # Avoid HF metadata/network checks when cache is already hydrated.
            from_pretrained_kwargs["local_files_only"] = True

        try:
            from pyannote.audio import Pipeline
            self.pipeline = Pipeline.from_pretrained(
                pipeline_source,
                **from_pretrained_kwargs
            ).to(torch.device(device))
        except TypeError:
            # Backward compatibility in case local_files_only is unsupported.
            from_pretrained_kwargs.pop("local_files_only", None)
            self.pipeline = Pipeline.from_pretrained(
                pipeline_source,
                **from_pretrained_kwargs
            ).to(torch.device(device))
        
        self.device = device
        self.min_duration = 0.5
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self._bounds_fallback_logged = False

    @staticmethod
    def _unwrap_annotation(diarization_output):
        """Normalize pyannote outputs to an Annotation-like object."""
        current = diarization_output

        # pyannote output shapes changed across versions. Walk through common
        # wrapper attributes/keys until we reach an object exposing itertracks().
        for _ in range(4):
            if hasattr(current, "itertracks"):
                return current

            unwrapped = None
            for attr_name in (
                "speaker_diarization",
                "exclusive_speaker_diarization",
                "diarization",
                "annotation",
            ):
                if hasattr(current, attr_name):
                    candidate = getattr(current, attr_name)
                    unwrapped = candidate() if callable(candidate) else candidate
                    break

            if unwrapped is None and isinstance(current, dict):
                for key in (
                    "speaker_diarization",
                    "exclusive_speaker_diarization",
                    "diarization",
                    "annotation",
                ):
                    if key in current:
                        unwrapped = current[key]
                        break

            if unwrapped is None:
                break

            current = unwrapped

        raise TypeError(
            "Unsupported diarization output type; expected an Annotation-like object "
            f"with itertracks(), got {type(current).__name__}."
        )
        
    def process(self, audio_buffer: np.ndarray, sample_rate: int) -> List[dict]:
        waveform = torch.from_numpy(audio_buffer.astype(np.float32) / 32768.0).unsqueeze(0).to(self.device)
        duration = waveform.shape[1] / sample_rate
        if duration < self.min_duration:
            return [{"speaker": "SPEAKER_00", "start": 0, "end": duration}]

        diarization_kwargs = {}
        constrained = self.min_speakers is not None or self.max_speakers is not None
        using_exact_speakers = (
            self.min_speakers is not None
            and self.max_speakers is not None
            and self.min_speakers == self.max_speakers
        )
        if using_exact_speakers:
            diarization_kwargs["num_speakers"] = self.min_speakers
        else:
            if self.min_speakers is not None:
                diarization_kwargs["min_speakers"] = self.min_speakers
            if self.max_speakers is not None:
                diarization_kwargs["max_speakers"] = self.max_speakers

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", category=UserWarning)
            diarization = self.pipeline(
                {"waveform": waveform, "sample_rate": sample_rate},
                **diarization_kwargs,
            )

        bounds_warning = any(
            "outside" in str(w.message).lower() and "given bounds" in str(w.message).lower()
            for w in caught_warnings
        )
        if constrained and not using_exact_speakers and bounds_warning:
            if not self._bounds_fallback_logged:
                logging.warning(
                    "Configured diarization speaker bounds are too strict for some chunks; "
                    "falling back to automatic speaker count on those chunks."
                )
                self._bounds_fallback_logged = True
            diarization = self.pipeline({"waveform": waveform, "sample_rate": sample_rate})

        diarization = self._unwrap_annotation(diarization)
        
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
                    self.speaker.add_diarization_segments(results, chunk.timestamp)
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
    def __init__(self, transcriber: Any, transcription_queue: queue.Queue, speaker: Speaker, config: dict, write_transcript_func):
        super().__init__()
        self.transcriber = transcriber
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

        return {
            "temperature": temperatures,
            "compression_ratio_threshold": 2.4,
            "no_speech_threshold": 0.6,
            "condition_on_previous_text": True,
            "word_timestamps": True,
            "language": self.config.get("language")
        }

    def _transcribe(self, audio_buffer: np.ndarray):
        return self.transcriber.transcribe(audio_buffer, self.last_prompt, self.transcribe_options)

    def _process_chunk(self, chunk: AudioChunk, retries: int) -> None:
        for attempt in range(retries):
            try:
                segments = self._transcribe(chunk.audio_buffer)
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
            words = segment.words or []
            if not words and segment.text.strip():
                # Fallback path for backends that only provide segment-level timestamps.
                segment_start = float(segment.start or 0.0)
                segment_end = float(segment.end or segment_start)
                words = [
                    TranscribedWord(
                        word=segment.text.strip(),
                        start=segment_start,
                        end=segment_end,
                        probability=1.0,
                    )
                ]
            if not words:
                continue

            current_group = None
            for word in words:
                word_start = buffer_timestamp + word.start
                word_end = buffer_timestamp + word.end
                cleaned_word = word.word.strip()
                if not cleaned_word:
                    continue

                midpoint = (word_start + word_end) / 2.0 if word_end >= word_start else word_start
                word_speaker = self.speaker.get_speaker_for_timestamp(midpoint)

                if current_group is None or current_group["speaker"] != word_speaker:
                    if current_group and current_group["words"]:
                        current_group["text"] = " ".join(
                            word_info["word"] for word_info in current_group["words"]
                        ).strip()
                        processed_segments.append(current_group)
                    current_group = {
                        "speaker": word_speaker,
                        "words": [],
                        "start": None,
                        "end": None,
                        "text": ""
                    }

                current_group["words"].append({
                    "word": cleaned_word,
                    "start": word_start,
                    "end": word_end,
                    "probability": word.probability
                })

                if current_group["start"] is None:
                    current_group["start"] = word_start
                current_group["end"] = word_end

            if current_group and current_group["words"]:
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
    def __init__(
        self,
        config_path: Optional[str] = None,
        input_source: str = None,
        force_hf_refresh: bool = False,
    ):
        self.config = DEFAULT_CONFIG.copy()
        self.force_hf_refresh = force_hf_refresh

        config_candidates = [config_path] if config_path else ["config.local.json", "config.json"]
        resolved_config_path = next((path for path in config_candidates if path and os.path.exists(path)), None)

        if resolved_config_path:
            with open(resolved_config_path) as f:
                loaded_config = json.load(f)
                if not isinstance(loaded_config, dict):
                    raise ValueError(f"Invalid config format in {resolved_config_path}. Expected JSON object.")
                self.config.update(loaded_config)
            logging.info(f"Loaded configuration from {resolved_config_path}")
        else:
            logging.info("No config file found. Using built-in defaults.")

        if input_source:
            self.config["input_source"] = input_source
        
        # Validate configuration
        self._validate_config()
        
        os.makedirs(self.config["output_folder"], exist_ok=True)
        self.model_cache_dir = os.path.abspath(self.config.get("model_cache_dir", ".model_cache"))
        self.hf_cache_dir = os.path.join(self.model_cache_dir, "huggingface")
        self.pyannote_cache_dir = os.path.join(self.model_cache_dir, "pyannote")
        os.makedirs(self.hf_cache_dir, exist_ok=True)
        os.makedirs(self.pyannote_cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = self.hf_cache_dir

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

        model_cache_dir = self.config.get("model_cache_dir")
        if not isinstance(model_cache_dir, str) or not model_cache_dir.strip():
            raise ValueError("model_cache_dir must be a non-empty string path")

        for key in ("diarization_min_speakers", "diarization_max_speakers"):
            value = self.config.get(key)
            if value is not None and (not isinstance(value, int) or value <= 0):
                raise ValueError(f"{key} must be a positive integer or null")
        min_speakers = self.config.get("diarization_min_speakers")
        max_speakers = self.config.get("diarization_max_speakers")
        if (
            min_speakers is not None
            and max_speakers is not None
            and min_speakers > max_speakers
        ):
            raise ValueError("diarization_min_speakers cannot be greater than diarization_max_speakers")
        
        # Validate output_format
        output_format = self.config.get("output_format", "text")
        if output_format not in ["text", "json"]:
            raise ValueError(f"Invalid output_format: {output_format}. Must be 'text' or 'json'.")

        # Validate model alias for MLX backend.
        model_name = self.config.get("model", "base")
        if model_name not in ["tiny", "base", "small", "medium", "large", "turbo"]:
            raise ValueError(
                f"Invalid model: {model_name}. Must be one of "
                "'tiny', 'base', 'small', 'medium', 'large', or 'turbo'."
            )

    def _init_models(self):
        default_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        diarizer_device = default_device
        self.transcription_backend = "mlx"
        use_local_only = not self.force_hf_refresh

        if use_local_only:
            # Keep this before any huggingface_hub/pyannote import in this run.
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            logging.info("Using strict local-only model mode (no HuggingFace requests).")
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            logging.info("Forced HuggingFace cache refresh enabled for this run.")

        model_name = self.config.get("model", "base")
        try:
            import mlx_whisper
        except ImportError as e:
            raise ImportError(
                "mlx-whisper is required. Install it with: pip install mlx-whisper"
            ) from e

        model_alias = {
            "tiny": "mlx-community/whisper-tiny-mlx",
            "base": "mlx-community/whisper-base-mlx",
            "small": "mlx-community/whisper-small-mlx",
            "medium": "mlx-community/whisper-medium-mlx",
            "large": "mlx-community/whisper-large-v3-mlx",
            "turbo": "mlx-community/whisper-large-v3-turbo",
        }
        mlx_repo = model_alias.get(model_name)
        token = self.config.get("huggingface_token")
        if not token:
            raise ValueError("HuggingFace token required from environment variables")

        pyannote_pipeline_repo = "pyannote/speaker-diarization-3.1"
        pyannote_dependencies = [
            "pyannote/segmentation-3.0",
            "pyannote/speaker-diarization-community-1",
            "pyannote/wespeaker-voxceleb-resnet34-LM",
        ]

        mlx_local_path = self._cache_hf_repo(mlx_repo, token=token)
        pyannote_pipeline_local_path = self._cache_hf_repo(pyannote_pipeline_repo, token=token)
        for repo_id in pyannote_dependencies:
            self._cache_hf_repo(repo_id, token=token)

        logging.info(f"Using mlx-whisper backend with local cache: {mlx_local_path}")
        self.transcriber = MlxWhisperTranscriber(model_repo=mlx_local_path, module=mlx_whisper)
        
        diarizer_token = token if self.force_hf_refresh else None
        diarization_min_speakers = self.config.get("diarization_min_speakers")
        diarization_max_speakers = self.config.get("diarization_max_speakers")
        try:
            self.diarizer = Diarizer(
                auth_token=diarizer_token,
                device=diarizer_device,
                cache_dir=self.pyannote_cache_dir,
                pipeline_source=pyannote_pipeline_local_path,
                local_files_only=use_local_only,
                min_speakers=diarization_min_speakers,
                max_speakers=diarization_max_speakers,
            )
        except Exception as e:
            error_text = str(e).lower()
            mps_related_error = "mps" in error_text or "unsupported device" in error_text
            if diarizer_device == "mps" and mps_related_error:
                logging.warning(
                    f"Diarization does not support MPS in this setup ({e}); "
                    "falling back to CPU."
                )
                self.diarizer = Diarizer(
                    auth_token=diarizer_token,
                    device="cpu",
                    cache_dir=self.pyannote_cache_dir,
                    pipeline_source=pyannote_pipeline_local_path,
                    local_files_only=use_local_only,
                    min_speakers=diarization_min_speakers,
                    max_speakers=diarization_max_speakers,
                )
            else:
                raise
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
            transcriber=self.transcriber,
            transcription_queue=self.transcription_queue,
            speaker=self.speaker,
            config=self.config,
            write_transcript_func=self._write_transcript
        )
        self.transcription_worker.start()

    def _cache_hf_repo(self, repo_id: str, token: str) -> str:
        """Resolve a model snapshot from local cache, downloading only when needed."""
        from huggingface_hub import snapshot_download

        if self.force_hf_refresh:
            logging.info(f"Refreshing HuggingFace cache for {repo_id}")
            return snapshot_download(
                repo_id=repo_id,
                token=token,
                cache_dir=self.hf_cache_dir,
                force_download=True,
                resume_download=True,
            )

        try:
            local_path = snapshot_download(
                repo_id=repo_id,
                token=token,
                cache_dir=self.hf_cache_dir,
                local_files_only=True,
            )
            logging.info(f"Loaded {repo_id} from local cache")
            return local_path
        except Exception:
            raise RuntimeError(
                f"Local cache missing for {repo_id}. "
                "Run once with --refresh-hf-cache to download required models."
            )

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
        if filtered_buffer.size == 0:
            return True
        rms = np.sqrt(np.mean(filtered_buffer ** 2))
        if not np.isfinite(rms):
            return True
        return rms <= threshold

    def process_audio_chunk(
        self,
        audio_chunk: np.ndarray,
        buffer_timestamp: Optional[float] = None,
        enqueue_diarization: bool = True,
    ):
        if buffer_timestamp is None:
            buffer_timestamp = datetime.now().timestamp()
        audio_data = AudioChunk(audio_buffer=audio_chunk, timestamp=buffer_timestamp)
        
        with self.queue_put_lock:
            if enqueue_diarization:
                if self.diarization_queue.full() or self.transcription_queue.full():
                    logging.warning("Queue full, dropping audio chunk to keep worker queues aligned")
                    return
            elif self.transcription_queue.full():
                logging.warning("Transcription queue full, dropping audio chunk")
                return
            try:
                if enqueue_diarization:
                    self.diarization_queue.put_nowait(audio_data)
                self.transcription_queue.put_nowait(audio_data)
            except queue.Full:
                logging.warning("Queue became full while enqueueing; dropped chunk to avoid desync")

    def _dispatch_audio_chunk(
        self,
        audio_chunk: np.ndarray,
        buffer_timestamp: float,
        enqueue_diarization: bool,
    ) -> None:
        # Test doubles may monkeypatch process_audio_chunk with an older
        # single-argument signature. Keep runtime behavior while preserving
        # backwards compatibility in tests.
        parameter_count = len(inspect.signature(self.process_audio_chunk).parameters)
        if parameter_count <= 1:
            self.process_audio_chunk(audio_chunk)
        elif parameter_count == 2:
            self.process_audio_chunk(audio_chunk, buffer_timestamp)
        else:
            self.process_audio_chunk(audio_chunk, buffer_timestamp, enqueue_diarization)

    def _load_file_audio_mono(self, audio_path: str) -> np.ndarray:
        """Load entire WAV file as int16 mono for file-level diarization."""
        with wave.open(audio_path, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16)
            if wf.getnchannels() == 2:
                audio = audio.reshape(-1, 2)
                audio = ((audio[:, 0].astype(np.float32) + audio[:, 1].astype(np.float32)) / 2.0).astype(np.int16)
            return audio

    def _prime_speaker_timeline_for_file(self, audio_path: str) -> Optional[List[dict]]:
        """
        Pre-compute diarization on full file to improve speaker consistency
        across chunked transcription.
        """
        try:
            audio_mono = self._load_file_audio_mono(audio_path)
            results = self.diarizer.process(audio_mono, sample_rate=16000)
            if not results:
                return None
            self.speaker.add_diarization_segments(results, self.timestamp_manager.start_time)
            logging.info("Primed speaker timeline using full-file diarization.")
            return results
        except Exception as exc:
            logging.warning(
                f"Full-file diarization prepass failed ({exc}); "
                "falling back to chunk-level diarization."
            )
            return None

    def _merge_diarization_segments(
        self,
        diarization_segments: List[dict],
        max_gap_seconds: float = 0.35,
    ) -> List[dict]:
        turns: List[dict] = []
        sorted_segments = sorted(
            diarization_segments,
            key=lambda item: float(item.get("start", 0.0) or 0.0),
        )

        for segment in sorted_segments:
            speaker = str(segment.get("speaker", "SPEAKER_00"))
            start = float(segment.get("start", 0.0) or 0.0)
            end = float(segment.get("end", start) or start)
            if end < start:
                start, end = end, start
            if end <= start:
                continue

            if not turns:
                turns.append({"speaker": speaker, "start": start, "end": end})
                continue

            previous = turns[-1]
            same_speaker = previous["speaker"] == speaker
            close_enough = start - float(previous["end"]) <= max_gap_seconds
            if same_speaker and close_enough:
                previous["end"] = max(float(previous["end"]), end)
            else:
                turns.append({"speaker": speaker, "start": start, "end": end})

        return turns

    def _build_turn_transcript_segments(
        self,
        raw_segments: List[TranscribedSegment],
        speaker: str,
        slice_start: float,
        turn_start: float,
        turn_end: float,
    ) -> List[dict]:
        processed_segments: List[dict] = []

        for segment in raw_segments:
            words = segment.words or []
            if not words and segment.text.strip():
                fallback_start = float(segment.start or 0.0)
                fallback_end = float(segment.end or fallback_start)
                words = [
                    TranscribedWord(
                        word=segment.text.strip(),
                        start=fallback_start,
                        end=fallback_end,
                        probability=1.0,
                    )
                ]

            if not words:
                continue

            kept_words = []
            for word in words:
                cleaned_word = word.word.strip()
                if not cleaned_word:
                    continue
                absolute_start = slice_start + float(word.start)
                absolute_end = slice_start + float(word.end)
                if absolute_end < absolute_start:
                    absolute_start, absolute_end = absolute_end, absolute_start
                midpoint = (absolute_start + absolute_end) / 2.0
                if midpoint < turn_start or midpoint > turn_end:
                    continue
                kept_words.append(
                    {
                        "word": cleaned_word,
                        "start": max(turn_start, absolute_start),
                        "end": min(turn_end, absolute_end),
                        "probability": float(word.probability),
                    }
                )

            if not kept_words:
                continue

            text = " ".join(word["word"] for word in kept_words).strip()
            if not text:
                continue
            processed_segments.append(
                {
                    "speaker": speaker,
                    "words": kept_words,
                    "start": kept_words[0]["start"],
                    "end": kept_words[-1]["end"],
                    "text": text,
                    "time_reference": "relative",
                }
            )

        return processed_segments

    def _process_file_with_diarization_segments(self, audio_path: str, diarization_segments: List[dict]) -> bool:
        sample_rate = 16000
        audio_mono = self._load_file_audio_mono(audio_path)
        turns = self._merge_diarization_segments(diarization_segments)
        if not turns:
            logging.warning("Full-file diarization did not produce usable speaker turns.")
            return False

        audio_duration = len(audio_mono) / sample_rate
        slice_padding = 0.2
        turn_prompt: Optional[str] = None
        wrote_any_segment = False
        transcribe_options = dict(self.transcription_worker.transcribe_options)
        transcribe_options["condition_on_previous_text"] = False

        for turn in turns:
            turn_start = float(turn["start"])
            turn_end = float(turn["end"])
            if turn_end <= turn_start:
                continue

            slice_start = max(0.0, turn_start - slice_padding)
            slice_end = min(audio_duration, turn_end + slice_padding)
            start_sample = int(slice_start * sample_rate)
            end_sample = int(np.ceil(slice_end * sample_rate))
            if end_sample <= start_sample:
                continue

            audio_slice = audio_mono[start_sample:end_sample]
            if audio_slice.size == 0:
                continue

            raw_segments = self.transcriber.transcribe(audio_slice, turn_prompt, transcribe_options)
            processed_segments = self._build_turn_transcript_segments(
                raw_segments=raw_segments,
                speaker=str(turn["speaker"]),
                slice_start=slice_start,
                turn_start=turn_start,
                turn_end=turn_end,
            )
            if not processed_segments:
                continue

            self._write_transcript(processed_segments)
            wrote_any_segment = True

            combined_text = " ".join(segment["text"] for segment in processed_segments).strip()
            if combined_text:
                turn_prompt = combined_text[-200:] if len(combined_text) > 200 else combined_text

        if not wrote_any_segment:
            logging.warning("Segment-guided transcription produced no transcript segments.")
        return wrote_any_segment

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
                    uses_relative_reference = segment.get("time_reference") == "relative"
                    if uses_relative_reference:
                        relative_start = float(segment["start"])
                        relative_end = float(segment["end"])
                    else:
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
        uses_relative_reference = segment.get("time_reference") == "relative"
        if uses_relative_reference:
            relative_start = float(segment["start"])
            relative_end = float(segment["end"])
        else:
            relative_start = self.timestamp_manager.get_relative_time(segment["start"])
            relative_end = self.timestamp_manager.get_relative_time(segment["end"])

        def to_relative_word_time(raw_value: float) -> float:
            if uses_relative_reference:
                return float(raw_value)
            return self.timestamp_manager.get_relative_time(raw_value)

        return {
            "speaker": segment["speaker"],
            "start": round(relative_start, 3),
            "end": round(relative_end, 3),
            "text": segment["text"],
            "words": [
                {
                    "word": word["word"],
                    "start": round(to_relative_word_time(word["start"]), 3),
                    "end": round(to_relative_word_time(word["end"]), 3),
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
            diarization_segments = self._prime_speaker_timeline_for_file(audio_path)
            use_file_level_diarization = diarization_segments is not None
            if use_file_level_diarization and self.diarization_worker.is_alive():
                self._try_put_sentinel(self.diarization_queue, "diarization")
                self.diarization_worker.join(timeout=10.0)

            if use_file_level_diarization and diarization_segments is not None:
                self._process_file_with_diarization_segments(audio_path, diarization_segments)
                self._try_put_sentinel(self.transcription_queue, "transcription")
                return

            emitted_samples = 0
            for chunk in self.audio_processor.process_file(audio_path):
                if not self.is_silent(chunk):
                    if self.audio_buffer.add(chunk):
                        buffer_data = self.audio_buffer.get()
                        buffer_timestamp = self.timestamp_manager.start_time + (emitted_samples / 16000.0)
                        self._dispatch_audio_chunk(
                            buffer_data,
                            buffer_timestamp,
                            not use_file_level_diarization,
                        )
                        emitted_samples += len(buffer_data)
            
            # Add sentinel value to signal end of processing
            self._try_put_sentinel(self.transcription_queue, "transcription")
            if not use_file_level_diarization:
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

def parse_args():
    parser = argparse.ArgumentParser(description="Whisperize transcription + diarization")
    parser.add_argument(
        "input_source",
        nargs="?",
        default="microphone",
        help="Audio input source: 'microphone' or path to WAV file",
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        default=None,
        help="Path to config JSON file (optional)",
    )
    parser.add_argument(
        "--refresh-hf-cache",
        action="store_true",
        help="Force refresh models from HuggingFace for this run",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_source = args.input_source
    
    try:
        app = Whisperize(
            config_path=args.config_path,
            input_source=input_source,
            force_hf_refresh=args.refresh_hf_cache,
        )
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
