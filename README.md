# Whisperize

A Python application for real-time audio transcription and speaker diarization using Faster-Whisper and PyAnnote.

## Features
- Real-time audio transcription with Apple Silicon support (MPS)
- Advanced speaker diarization using PyAnnote
- Support for microphone and audio file input
- Multiple Whisper model sizes and quantization options
- Configurable via JSON with text or JSON output formats
- Thread-safe parallel processing of transcription and diarization

## Requirements

- Python 3.10
- FFmpeg (required for audio processing)
  ```bash
  # On macOS using Homebrew
  brew install ffmpeg

  # On Ubuntu/Debian
  sudo apt-get install ffmpeg

  # On Windows using Chocolatey
  choco install ffmpeg
  ```
- Apple Silicon Mac recommended for optimal performance with MPS acceleration

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/francescopace/whisperize.git
cd whisperize
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure the Application
1. Create a HuggingFace account at [https://huggingface.co/](https://huggingface.co/)
2. Generate an access token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Edit `config.json` and update with your settings:

```json
{
    "huggingface_token": "your_token_here",
    "output_folder": "transcripts/",
    "output_format": "text",
    "model": "turbo",
    "whisper_force_cpu": false,
    "language": "it",
    "buffer_duration": 4
}
```

#### Configuration Parameters

- **huggingface_token** (required): Your HuggingFace API token for accessing PyAnnote models
- **output_folder** (required): Directory where transcripts will be saved
- **output_format** (optional): Output format - `"text"` or `"json"` (default: `"text"`)
  - `text`: Creates a human-readable transcript with timestamps
  - `json`: Creates both a text file and a structured JSON file with metadata and word-level timestamps
- **model** (optional): Whisper model size - `"tiny"`, `"base"`, `"small"`, `"medium"`, `"large"`, `"turbo"` (default: `"base"`)
- **whisper_force_cpu** (optional): Force CPU usage even if GPU/MPS is available (default: `false`)
- **language** (optional): Language code (e.g., `"it"`, `"en"`, `"es"`). If not specified, language is auto-detected
- **buffer_duration** (optional): Audio buffer duration in seconds (default: `5.0`)

### Supported Models

#### Whisper Models
The application uses Faster-Whisper for transcription. Available models:
- `tiny` - Fastest, lowest accuracy
- `base` - Good balance of speed and accuracy
- `small` - Better accuracy, slower
- `medium` - High accuracy
- `large` - Highest accuracy, slowest
- `turbo` - Optimized large model

See [Whisper documentation](https://github.com/openai/whisper?tab=readme-ov-file#available-models-and-languages) for language support and model details.

#### Diarization Model
The application uses **PyAnnote speaker-diarization-3.1** for speaker identification. This model is automatically loaded and requires a HuggingFace token for access.

## Usage

### Basic Usage

**Microphone Input (default):**
```bash
python whisperize.py
# or explicitly
python whisperize.py microphone
```

**Audio File Input:**
```bash
python whisperize.py path/to/audio.wav
```

**Note:** Only WAV files (16-bit, mono or stereo) are currently supported.

### Output

Transcripts are saved in the `output_folder` specified in `config.json`:

**Text Format** (`output_format: "text"`):
```
# Transcript started at 2025-02-11 18:30:00

[00:00:02.500-00:00:05.300] [SPEAKER_00]: Hello, this is a test transcription.
[00:00:06.100-00:00:09.800] [SPEAKER_01]: Yes, I can hear you clearly.
```

**JSON Format** (`output_format: "json"`):
- Creates both a `.txt` file (for real-time monitoring) and a `.json` file
- JSON includes metadata, speaker labels, timestamps, and word-level details with confidence scores

Example JSON structure:
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
      "start": 2.500,
      "end": 5.300,
      "text": "Hello, this is a test transcription.",
      "words": [
        {
          "word": "Hello",
          "start": 2.500,
          "end": 2.800,
          "probability": 0.95
        }
      ]
    }
  ]
}
```
