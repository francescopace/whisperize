from __future__ import annotations

import json
import queue
import subprocess
import sys
import types
from pathlib import Path

import pytest

# Ensure repository root (where whisperize.py lives) is importable with plain `pytest`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Allow importing whisperize.py in lightweight test environments.
if "pyaudio" not in sys.modules:
    sys.modules["pyaudio"] = types.SimpleNamespace(PyAudio=object, paInt16=8)

if "pyannote.audio" not in sys.modules:
    pyannote_module = types.ModuleType("pyannote")
    pyannote_audio_module = types.ModuleType("pyannote.audio")

    class _DummyPipeline:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def to(self, *args, **kwargs):
            return self

    pyannote_audio_module.Pipeline = _DummyPipeline
    sys.modules["pyannote"] = pyannote_module
    sys.modules["pyannote.audio"] = pyannote_audio_module

import whisperize


class _DummyWorker:
    def is_alive(self) -> bool:
        return False

    def stop(self) -> None:
        return None

    def join(self, timeout: float | None = None) -> None:
        return None


class _InlineTranscriptionWorker(_DummyWorker):
    def __init__(self, transcriber, speaker, write_transcript_func) -> None:
        self._transcriber = transcriber
        self._speaker = speaker
        self._write_transcript_func = write_transcript_func

    def _process_chunk(self, chunk, retries: int = 1) -> None:
        segments = self._transcriber.transcribe(chunk.audio_buffer, None, {})
        current_speaker, _ = self._speaker.current

        processed_segments = []
        for segment in segments:
            words = segment.words or []
            if not words:
                continue

            processed_segments.append(
                {
                    "speaker": current_speaker,
                    "words": [
                        {
                            "word": word.word.strip(),
                            "start": chunk.timestamp + word.start,
                            "end": chunk.timestamp + word.end,
                            "probability": word.probability,
                        }
                        for word in words
                        if word.word.strip()
                    ],
                    "start": chunk.timestamp + words[0].start,
                    "end": chunk.timestamp + words[-1].end,
                    "text": segment.text.strip(),
                }
            )

        self._write_transcript_func(processed_segments)


class _FakeTranscriber:
    def __init__(self) -> None:
        self._idx = 0
        self._texts = [
            "Hello I am the first speaker and this is a diarization test.",
            "Now I speak as the second speaker and add a short response.",
            "Finally I join as the third speaker with a closing sentence.",
        ]

    def transcribe(self, audio_buffer, initial_prompt, options):
        text = self._texts[min(self._idx, len(self._texts) - 1)]
        self._idx += 1

        words = []
        cursor = 0.0
        for token in text.split():
            start = cursor
            end = start + 0.20
            cursor = end + 0.02
            words.append(
                whisperize.TranscribedWord(
                    word=token,
                    start=start,
                    end=end,
                    probability=0.99,
                )
            )

        return [whisperize.TranscribedSegment(text=text, words=words, start=0.0, end=cursor)]


def _repo_root() -> Path:
    return REPO_ROOT


def _ensure_english_audio_fixture() -> Path:
    audio_path = _repo_root() / "test_audio" / "diarization_3speakers_realistic.wav"
    if audio_path.exists():
        return audio_path

    script_path = _repo_root() / "script" / "generate_diarization_test_audio.py"
    subprocess.run([sys.executable, str(script_path)], check=True)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio fixture not created: {audio_path}")
    return audio_path


def _fake_init_models(self) -> None:
    self.transcriber = _FakeTranscriber()
    self.speaker = whisperize.Speaker()
    self.diarization_queue = queue.Queue(maxsize=100)
    self.transcription_queue = queue.Queue(maxsize=100)
    self.diarization_worker = _DummyWorker()
    self.transcription_worker = _InlineTranscriptionWorker(
        transcriber=self.transcriber,
        speaker=self.speaker,
        write_transcript_func=self._write_transcript,
    )


def _fake_process_audio_chunk(self, audio_chunk) -> None:
    # Keep deterministic speaker rotation for predictable transcript assertions.
    chunk_index = getattr(self, "_test_chunk_index", 0)
    speakers = ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]
    speaker_label = speakers[min(chunk_index, len(speakers) - 1)]
    self._test_chunk_index = chunk_index + 1

    timestamp = whisperize.datetime.now().timestamp()
    self.speaker.update(speaker_label, timestamp)

    chunk = whisperize.AudioChunk(audio_buffer=audio_chunk, timestamp=timestamp)
    self.transcription_worker._process_chunk(chunk, retries=1)


def test_whisperize_generates_expected_transcript_for_english_fixture_audio(tmp_path, monkeypatch):
    audio_path = _ensure_english_audio_fixture()

    output_dir = tmp_path / "transcripts"
    config_path = tmp_path / "config.test.json"
    config_path.write_text(
        json.dumps(
            {
                "output_folder": str(output_dir),
                "output_format": "text",
                "model": "base",
                "language": "en",
                "buffer_duration": 1.0,
                "temperature": [0.0, 0.2, 0.4],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("HUGGINGFACE_TOKEN", "test-token")
    monkeypatch.setattr(whisperize.Whisperize, "_init_models", _fake_init_models)
    monkeypatch.setattr(whisperize.Whisperize, "process_audio_chunk", _fake_process_audio_chunk)

    app = whisperize.Whisperize(config_path=str(config_path), input_source=str(audio_path))
    app.process_file(str(audio_path))

    transcript_path = Path(app.transcript_path)
    assert transcript_path.exists(), "Transcript file was not generated"

    transcript = transcript_path.read_text(encoding="utf-8")
    assert "first speaker" in transcript
    assert "second speaker" in transcript
    assert "third speaker" in transcript
    assert "[SPEAKER_00]" in transcript
    assert "[SPEAKER_01]" in transcript
    assert "[SPEAKER_02]" in transcript
