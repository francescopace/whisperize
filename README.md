# Whisperize

Real-time audio transcription with speaker diarization, optimized for Apple Silicon.
Uses [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) for transcription and [PyAnnote](https://github.com/pyannote/pyannote-audio) for speaker identification.

## Features

- Live microphone and WAV file transcription
- Speaker diarization (who said what)
- Multiple Whisper model sizes with MLX acceleration
- Text and JSON output formats
- Offline-first: runs from local model cache after initial download

## Requirements

- macOS (Apple Silicon)
- Python 3.12
- [FFmpeg](https://ffmpeg.org/) and [PortAudio](http://www.portaudio.com/) via Homebrew:

```bash
brew install ffmpeg portaudio
```

## Installation

```bash
git clone https://github.com/francescopace/whisperize.git
cd whisperize
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### HuggingFace token

A HuggingFace token is required to download the PyAnnote diarization model.

1. Create an account at [huggingface.co](https://huggingface.co/)
2. Generate a token at [Settings > Tokens](https://huggingface.co/settings/tokens)
3. Export it in your shell:

```bash
export HUGGINGFACE_TOKEN="your_token_here"
```

## Usage

```bash
# Microphone (default)
python whisperize.py

# WAV file
python whisperize.py path/to/audio.wav

# Force model cache refresh from HuggingFace
python whisperize.py --refresh-hf-cache
```

> WAV files must be 16-bit, 16 kHz, mono or stereo.

By default Whisperize runs in **local-only mode** — it loads models from the local cache without contacting HuggingFace. If a model is missing, run once with `--refresh-hf-cache` to download it.

## Configuration

Copy the example config to get started:

```bash
cp config.example.json config.local.json
```

`config.local.json` is git-ignored and takes precedence over `config.json`. If no config file is found, built-in defaults are used.

| Parameter | Default | Description |
|---|---|---|
| `output_folder` | `"transcripts/"` | Directory for saved transcripts |
| `output_format` | `"text"` | `"text"` or `"json"` |
| `model` | `"turbo"` | Whisper model size (see table below) |
| `language` | auto-detect | Language code (`"en"`, `"it"`, `"es"`, …) |
| `buffer_duration` | `4.0` | Audio buffer length in seconds |
| `temperature` | `[0.0, 0.2, 0.4]` | Whisper decoding temperature schedule |
| `model_cache_dir` | `".model_cache"` | Local directory for model snapshots |
| `diarization_min_speakers` | `null` | Optional lower bound for diarization speaker count |
| `diarization_max_speakers` | `null` | Optional upper bound for diarization speaker count |

For fixed-speaker test clips (like your generated 3-speaker sample), set both values to `3` to stabilize clustering.

## Models

| Alias | MLX Model | Notes |
|---|---|---|
| `tiny` | `mlx-community/whisper-tiny-mlx` | Fastest, lowest accuracy |
| `base` | `mlx-community/whisper-base-mlx` | Good speed/accuracy balance |
| `small` | `mlx-community/whisper-small-mlx` | Better accuracy |
| `medium` | `mlx-community/whisper-medium-mlx` | High accuracy |
| `large` | `mlx-community/whisper-large-v3-mlx` | Highest accuracy |
| `turbo` | `mlx-community/whisper-large-v3-turbo` | Optimized large model |

Speaker diarization uses **PyAnnote speaker-diarization-3.1**, loaded automatically via the HuggingFace token.

See the [Whisper docs](https://github.com/openai/whisper#available-models-and-languages) for language support details.

## Output

### Text format

```
# Transcript started at 2025-02-11 18:30:00

[00:00:02.500-00:00:05.300] [SPEAKER_00]: Hello, this is a test transcription.
[00:00:06.100-00:00:09.800] [SPEAKER_01]: Yes, I can hear you clearly.
```

### JSON format

Produces both a `.txt` file (for live monitoring) and a `.json` file with full metadata:

```json
{
  "metadata": {
    "start_time": "2025-02-11T18:30:00",
    "duration": 120.5,
    "model": "turbo",
    "language": "it",
    "source": "microphone"
  },
  "segments": [
    {
      "speaker": "SPEAKER_00",
      "start": 2.5,
      "end": 5.3,
      "text": "Hello, this is a test transcription.",
      "words": [
        { "word": "Hello", "start": 2.5, "end": 2.8, "probability": 0.95 }
      ]
    }
  ]
}
```

## Troubleshooting

**Duplicate FFmpeg Objective-C warnings at startup** — usually caused by multiple FFmpeg builds loaded simultaneously (Homebrew + wheel-bundled). Fix by rebuilding `av` against the system FFmpeg:

```bash
pip uninstall -y av
PKG_CONFIG_PATH="$(brew --prefix ffmpeg)/lib/pkgconfig" pip install --no-binary av av
```
