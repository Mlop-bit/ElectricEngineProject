import csv
from pathlib import Path

import numpy as np
from scipy.io import wavfile

project_dir = Path(__file__).parent
dataset_dir = project_dir / "IDMT-ISA-ELECTRIC-ENGINE" / "train_cut"
output_path = project_dir / "train_cut_features.csv"
categories = {
    "Good": "engine1_good",
    "Broken": "engine2_broken",
    "Heavy load": "engine3_heavyload",
}


def to_mono_float(audio: np.ndarray) -> np.ndarray:
    if np.issubdtype(audio.dtype, np.integer):
        limits = np.iinfo(audio.dtype)
        audio = audio.astype(np.float64)
        if np.issubdtype(limits.dtype, np.unsignedinteger):
            midpoint = (limits.max + 1) / 2
            audio = (audio - midpoint) / midpoint
        else:
            audio /= max(abs(limits.min), limits.max)
    else:
        audio = audio.astype(np.float64)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio


def dominant_frequency(audio: np.ndarray, sample_rate: int) -> float:
    if len(audio) < 2:
        return 0.0

    centered_audio = audio - np.mean(audio)
    windowed_audio = centered_audio * np.hanning(len(centered_audio))
    magnitudes = np.abs(np.fft.rfft(windowed_audio))
    magnitudes[0] = 0

    if not np.any(magnitudes):
        return 0.0

    frequencies = np.fft.rfftfreq(len(audio), d=1 / sample_rate)
    return float(frequencies[np.argmax(magnitudes)])


rows = []
for label, folder_name in categories.items():
    audio_dir = dataset_dir / folder_name
    audio_files = sorted(
        audio_dir.glob("*.wav"),
        key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
    )
    if not audio_files:
        raise FileNotFoundError(f"No WAV files found in {audio_dir}")

    for audio_path in audio_files:
        sample_rate, raw_audio = wavfile.read(audio_path)
        audio = to_mono_float(raw_audio)
        if len(audio) == 0:
            raise ValueError(f"Audio file is empty: {audio_path}")

        rows.append(
            {
                "class": label,
                "file": audio_path.name,
                "peak": float(np.max(np.abs(audio))),
                "ptp": float(np.ptp(audio)),
                "rms": float(np.sqrt(np.mean(audio ** 2))),
                "dominant_frequency_hz": dominant_frequency(audio, sample_rate),
            }
        )

fieldnames = ["class", "file", "peak", "ptp", "rms", "dominant_frequency_hz"]
with output_path.open("w", newline="", encoding="utf-8") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Processed {len(rows)} audio files.")
print(f"Feature data saved to: {output_path}")
