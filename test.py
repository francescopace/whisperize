from lightning_whisper_mlx import LightningWhisperMLX
import json

# Models ["tiny", "small", "distil-small.en", "base", "medium", distil-medium.en", "large", "large-v2", "distil-large-v2", "large-v3", "distil-large-v3"]
# Quantization [None, "4bit", "8bit"]
# The default batch_size is 12, higher is better for throughput but you might run into memory issues. 
# The heuristic is it really depends on the size of the model. 
# If you are running the smaller models, then higher batch size, larger models, lower batch size. Also keep in mind your unified memory!

whisper = LightningWhisperMLX(model="medium", batch_size=12, quant=None)
result = whisper.transcribe(audio_path="inputs/sample.wav")
print(json.dumps(result, indent=4))