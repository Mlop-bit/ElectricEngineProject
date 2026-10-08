"""Extract simple acoustic features from the training WAV files in the ZIP."""

from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

from read_wav import _read_audio_member


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARCHIVE = PROJECT_ROOT / "data" / "IDMT-ISA-ELECTRIC-ENGINE.zip"
DEFAULT_OUTPUT = PROJECT_ROOT / "train_cut_features.csv"
CATEGORIES = {
    "Good": "engine1_good",
    "Broken": "engine2_broken",
    "Heavy load": "engine3_heavyload",
}
FIELDNAMES = ["class", "file", "peak", "ptp", "rms", "dominant_frequency_hz"]


def to_mono_float(audio: np.ndarray) -> np.ndarray:
    """Convert integer or floating-point WAV samples to mono float values."""
    if np.issubdtype(audio.dtype, np.integer):
        limits = np.iinfo(audio.dtype)
        is_unsigned = np.issubdtype(audio.dtype, np.unsignedinteger)
        audio = audio.astype(np.float64)
        if is_unsigned:
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


def extract_features(archive_path: Path, output_path: Path) -> int:
    """Read training WAVs from the archive and write a labeled CSV."""
    rows = []
    with zipfile.ZipFile(archive_path) as archive:
        archive_members = set(archive.namelist())
        for label, folder_name in CATEGORIES.items():
            prefix = f"train_cut/{folder_name}/"
            audio_files = [
                member for member in archive_members
                if member.startswith(prefix) and member.lower().endswith(".wav")
            ]
            audio_files.sort(key=lambda member: int(PurePosixPath(member).stem.rsplit("_", 1)[-1]))
            if not audio_files:
                raise FileNotFoundError(f"No training WAV files found under {prefix} in {archive_path}")

            for member in audio_files:
                raw_audio, sample_rate = _read_audio_member(archive, member)
                audio = to_mono_float(raw_audio)
                if len(audio) == 0:
                    raise ValueError(f"Audio file is empty: {member}")

                rows.append({
                    "class": label,
                    "file": PurePosixPath(member).name,
                    "peak": float(np.max(np.abs(audio))),
                    "ptp": float(np.ptp(audio)),
                    "rms": float(np.sqrt(np.mean(audio ** 2))),
                    "dominant_frequency_hz": dominant_frequency(audio, sample_rate),
                })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE,
                        help=f"dataset ZIP (default: {DEFAULT_ARCHIVE})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"feature CSV output (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()
    if not args.archive.is_file():
        parser.error(f"dataset archive does not exist: {args.archive}")

    row_count = extract_features(args.archive, args.output)
    print(f"Processed {row_count} audio files.")
    print(f"Feature data saved to: {args.output}")


if __name__ == "__main__":
    main()
