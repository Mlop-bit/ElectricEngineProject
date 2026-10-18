"""Extract acoustic features from the training and test WAV files in the ZIP."""

from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

from read_wav import DEFAULT_ARCHIVE, PROJECT_ROOT, _read_audio_member


CATEGORIES = {
    "Good": "engine1_good",
    "Broken": "engine2_broken",
    "Heavy load": "engine3_heavyload",
}
FIELDNAMES = [
    "class", "file", "peak", "ptp", "rms", "dominant_frequency_hz",
    "wobble_frequency_hz", "wobble_strength",
]


def to_mono_float(audio: np.ndarray) -> np.ndarray:
    """Convert integer or floating-point WAV samples to mono float values."""
    if np.issubdtype(audio.dtype, np.integer):
        limits = np.iinfo(audio.dtype)
        audio = audio.astype(np.float64)
        if np.issubdtype(audio.dtype, np.unsignedinteger):
            midpoint = (limits.max + 1) / 2
            audio = (audio - midpoint) / midpoint
        else:
            audio /= max(abs(limits.min), limits.max)
    else:
        audio = audio.astype(np.float64)
    return audio.mean(axis=1) if audio.ndim == 2 else audio


def dominant_frequency(audio: np.ndarray, sample_rate: float) -> float:
    if len(audio) < 2:
        return 0.0
    centered = audio - np.mean(audio)
    magnitudes = np.abs(np.fft.rfft(centered * np.hanning(len(centered))))
    magnitudes[0] = 0
    if not np.any(magnitudes):
        return 0.0
    frequencies = np.fft.rfftfreq(len(audio), d=1 / sample_rate)
    return float(frequencies[np.argmax(magnitudes)])


def loudness_curve(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, float]:
    """Return block RMS values sampled at about 200 Hz."""
    block_size = max(1, sample_rate // 200)
    usable_length = len(audio) - len(audio) % block_size
    if usable_length == 0:
        return np.array([], dtype=np.float64), sample_rate / block_size
    blocks = audio[:usable_length].reshape(-1, block_size)
    return np.sqrt(np.mean(blocks ** 2, axis=1)), sample_rate / block_size


def extract_folder(archive: zipfile.ZipFile, folder: str, output_path: Path) -> int:
    rows = []
    for label, folder_name in CATEGORIES.items():
        prefix = f"{folder}/{folder_name}/"
        members = sorted(
            (name for name in archive.namelist()
             if name.startswith(prefix) and name.lower().endswith(".wav")),
            key=lambda name: int(PurePosixPath(name).stem.rsplit("_", 1)[-1]),
        )
        if not members:
            raise FileNotFoundError(f"No WAV files found under {prefix} in the dataset archive")
        for member in members:
            raw_audio, sample_rate = _read_audio_member(archive, member)
            audio = to_mono_float(raw_audio)
            if len(audio) == 0:
                raise ValueError(f"Audio file is empty: {member}")
            loudness, loudness_rate = loudness_curve(audio, sample_rate)
            mean_loudness = float(np.mean(loudness)) if len(loudness) else 0.0
            rows.append({
                "class": label,
                "file": PurePosixPath(member).name,
                "peak": float(np.max(np.abs(audio))),
                "ptp": float(np.ptp(audio)),
                "rms": float(np.sqrt(np.mean(audio ** 2))),
                "dominant_frequency_hz": dominant_frequency(audio, sample_rate),
                "wobble_frequency_hz": dominant_frequency(loudness, loudness_rate),
                "wobble_strength": float(np.std(loudness) / mean_loudness) if mean_loudness else 0.0,
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {output_path}")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE,
                        help=f"dataset ZIP (default: {DEFAULT_ARCHIVE})")
    parser.add_argument("--folder", choices=("train_cut", "test_cut"),
                        help="extract only this dataset split (default: both)")
    args = parser.parse_args()
    if not args.archive.is_file():
        parser.error(f"dataset archive does not exist: {args.archive}")

    folders = [args.folder] if args.folder else ["train_cut", "test_cut"]
    with zipfile.ZipFile(args.archive) as archive:
        for folder in folders:
            output_path = PROJECT_ROOT / f"{folder}_features.csv"
            extract_folder(archive, folder, output_path)


if __name__ == "__main__":
    main()
