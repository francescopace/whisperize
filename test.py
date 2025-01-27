import sounddevice as sd
import numpy as np
import whisper
import queue
import threading

# Parametri audio
SAMPLE_RATE = 16000  # Whisper richiede audio a 16 kHz
BLOCK_SIZE = 1024    # Dimensione del blocco per l'acquisizione audio
CHANNELS = 1         # Canali audio (mono)
BUFFER_DURATION = 2   # Durata dei blocchi di trascrizione in secondi

# Coda per gestire i blocchi audio
audio_queue = queue.Queue()

# Modello Whisper
model = whisper.load_model("base")  # Sostituisci con "tiny", "small", ecc. per velocità maggiore

def audio_callback(indata, frames, time, status):
    """Callback per acquisire audio in tempo reale."""
    if status:
        print(f"Status: {status}", flush=True)
    # Inserisci l'audio grezzo nella coda
    audio_queue.put(indata.copy())

def transcribe_audio():
    """Thread per trascrivere audio dalla coda in tempo reale."""
    print("Inizio trascrizione in tempo reale...")
    while True:
        # Accumula audio dalla coda per BUFFER_DURATION secondi
        audio_buffer = []
        for _ in range(int(SAMPLE_RATE * BUFFER_DURATION / BLOCK_SIZE)):
            audio_buffer.append(audio_queue.get())
        audio_buffer = np.concatenate(audio_buffer, axis=0)
        
        # Normalizza l'audio per Whisper
        audio_data = audio_buffer.flatten()
        
        # Trascrizione
        result = model.transcribe(audio_data, fp16=False, language="it") # fp16=False se non hai GPU
        print(f"Sottotitoli: {result['text']}", flush=True)

def main():
    # Configura l'acquisizione audio
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        blocksize=BLOCK_SIZE,
        callback=audio_callback,
        dtype="float32"
    ):
        # Avvia il thread di trascrizione
        transcribe_thread = threading.Thread(target=transcribe_audio, daemon=True)
        transcribe_thread.start()

        print("Premi Ctrl+C per interrompere lo streaming.")
        try:
            while True:
                pass  # Mantieni il programma in esecuzione
        except KeyboardInterrupt:
            print("\nInterruzione dello streaming.")

if __name__ == "__main__":
    main()