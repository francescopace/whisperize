#!/usr/bin/env python3
"""Generate a realistic 3-speaker WAV sample for diarization tests on macOS."""

from __future__ import annotations

import subprocess
from pathlib import Path


OUTPUT_DIR = Path("test_audio")
OUTPUT_FILE = OUTPUT_DIR / "diarization_3speakers_realistic.wav"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def choose_voices(count: int = 3) -> list[str]:
    voices_raw = subprocess.check_output(["say", "-v", "?"], text=True)
    available = [line.split()[0] for line in voices_raw.splitlines() if line.strip()]

    preferred = ["Samantha", "Daniel", "Alex", "Alice", "Victoria", "Fred"]
    chosen: list[str] = [v for v in preferred if v in available][:count]

    if len(chosen) < count:
        for voice in available:
            if voice not in chosen:
                chosen.append(voice)
            if len(chosen) == count:
                break

    if len(chosen) < count:
        raise RuntimeError("Not enough system voices available to build test audio.")

    return chosen


def generate_speaker_track(voice: str, text: str, index: int) -> Path:
    aiff = OUTPUT_DIR / f"spk{index}.aiff"
    wav = OUTPUT_DIR / f"spk{index}.wav"

    run(["say", "-v", voice, text, "-o", str(aiff)])
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(aiff),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-sample_fmt",
            "s16",
            str(wav),
        ]
    )
    return wav


def build_final_mix(tracks: list[Path]) -> None:
    # Use separated turns to maximize diarization stability and speaker separability.
    filter_complex = (
        "[0:a]adelay=0|0[a0];"
        "[1:a]adelay=7000|7000[a1];"
        "[2:a]adelay=14000|14000[a2];"
        "[a0][a1][a2]amix=inputs=3:normalize=0[out]"
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(tracks[0]),
            "-i",
            str(tracks[1]),
            "-i",
            str(tracks[2]),
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-sample_fmt",
            "s16",
            str(OUTPUT_FILE),
        ]
    )


def cleanup_temporary_tracks() -> None:
    for pattern in ("spk*.aiff", "spk*.wav"):
        for path in OUTPUT_DIR.glob(pattern):
            path.unlink(missing_ok=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    voices = choose_voices(3)
    texts = [
        "Hello, I am the first speaker. I talk alone in this first section to help diarization.",
        "Now I am the second speaker. This turn starts later and stays isolated from the others.",
        "Finally I am the third speaker. My section is clearly separated so three speakers are easier to detect.",
    ]

    tracks = [
        generate_speaker_track(voices[idx], texts[idx], idx)
        for idx in range(3)
    ]
    build_final_mix(tracks)
    cleanup_temporary_tracks()

    print(f"Voices used: {', '.join(voices)}")
    print(f"Created: {OUTPUT_FILE}")
    print("Format: WAV, mono, 16kHz, 16-bit")


if __name__ == "__main__":
    main()
