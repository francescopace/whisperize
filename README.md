# Whisperize

A Python application for real-time audio transcription and speaker diarization using Lightning Whisper MLX (optimized for Apple Silicon) and PyAnnote.

## Features
- Real-time audio transcription optimized for Apple Silicon using MLX
- Advanced diarization using PyAnnote
- Support for various audio input sources
- Multiple model sizes and quantization options
- Configurable via JSON

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
- Apple Silicon Mac recommended for optimal performance with MLX

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/whisperize.git
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

The application uses a `config.json` file for configuration. You can specify various parameters, including the audio model and batch size. The `diarizer` parameter can be set via command line or will default to `embeddings`.

3. edit`config.json` and update with your settings:
   ```json
   {
       "huggingface_token": "your_token_here",
       "output_folder": "transcripts/",
       "temp_folder": "temp/",
       "model": "medium",
       "batch_size": 12,
       "quant": null
   }
   ```
### Models

The following models are supported:
["tiny", "small", "distil-small.en", "base", "medium", "distil-medium.en", "large", "large-v2", "distil-large-v2", "large-v3", "distil-large-v3"]

### Quantization

The following quantization options are available:
[None, "4bit", "8bit"]

### Batch Size
The default batch_size is 12, higher is better for throughput but you might run into memory issues. The heuristic is it really depends on the size of the model. If you are running the smaller models, then higher batch size, larger models, lower batch size. Also keep in mind your unified memory!

## Usage

To run the application, use the following command:
```bash
python whisperize.py --source <>
```
Replace `<source>` with the desired sourcey (default is `microphone`). It can be a file path.

The transcript will be saved with in the `output_folder` specified in `config.json`.

## License
MIT License
