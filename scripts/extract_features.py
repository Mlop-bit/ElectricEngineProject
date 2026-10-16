"""Extract simple features from the training audio and save them to a CSV file.

What this script does:
1. Opens the dataset zip in data/ (downloaded by download_dataset.py).
2. Finds every WAV file in the train_cut/ folder of the zip.
3. Reads each file with read_wav.py and computes four numbers (features).
4. Saves one row per file to train_cut_features.csv in the project folder.

Run with: python scripts/extract_features.py
"""

import csv
import zipfile

import numpy as np

from read_wav import DEFAULT_ARCHIVE, PROJECT_ROOT, read_audio

OUTPUT_CSV = PROJECT_ROOT / "train_cut_features.csv"
TRAINING_FOLDER = "train_cut"

# Folder name inside the zip -> class name written to the CSV.
CLASS_NAMES = {
    "engine1_good": "Good",
    "engine2_broken": "Broken",
    "engine3_heavyload": "Heavy load",
}

CSV_COLUMNS = ["class", "file", "peak", "ptp", "rms", "dominant_frequency_hz"]


def find_training_files(archive_path):
    """Return a list of (class name, path inside zip) for every training WAV.

    A training file looks like ".../train_cut/engine1_good/pure_0.wav".
    """
    training_files = []
    with zipfile.ZipFile(archive_path) as archive:
        for path in sorted(archive.namelist()):
            parts = path.split("/")
            if len(parts) < 3 or not path.lower().endswith(".wav"):
                continue

            folder = parts[-3]      # e.g. "train_cut"
            engine = parts[-2]      # e.g. "engine1_good"
            if folder == TRAINING_FOLDER and engine in CLASS_NAMES:
                training_files.append((CLASS_NAMES[engine], path))
    return training_files


def to_mono(samples):
    """Turn 16-bit samples into one channel of numbers between -1 and 1."""
    if samples.dtype != np.int16:
        raise ValueError(f"Expected 16-bit audio, got {samples.dtype}")

    audio = samples / 32768.0       # 16-bit values go from -32768 to 32767
    return audio.mean(axis=1)       # average the channels (stereo -> mono)


def dominant_frequency(audio, sample_rate):
    """Return the frequency (in Hz) that is loudest in the sound."""
    audio = audio - audio.mean()                    # remove any constant offset
    audio = audio * np.hanning(len(audio))          # soften the edges for a cleaner FFT
    strengths = np.abs(np.fft.rfft(audio))          # how strong each frequency is
    frequencies = np.fft.rfftfreq(len(audio), d=1 / sample_rate)
    strengths[0] = 0                                # ignore 0 Hz (no real sound)
    return float(frequencies[np.argmax(strengths)])


def compute_features(audio, sample_rate):
    """Return the four features for one sound."""
    return {
        "peak": float(np.max(np.abs(audio))),           # loudest single sample
        "ptp": float(np.max(audio) - np.min(audio)),    # peak-to-peak: highest minus lowest
        "rms": float(np.sqrt(np.mean(audio ** 2))),     # average loudness
        "dominant_frequency_hz": dominant_frequency(audio, sample_rate),
    }


def main():
    if not DEFAULT_ARCHIVE.is_file():
        raise SystemExit(
            f"Dataset not found: {DEFAULT_ARCHIVE}\n"
            "Download it first with: python scripts/download_dataset.py"
        )

    training_files = find_training_files(DEFAULT_ARCHIVE)
    if not training_files:
        raise SystemExit(f"No WAV files found in {TRAINING_FOLDER}/ inside the zip.")
    print(f"Found {len(training_files)} training files.")

    rows = []
    for number, (class_name, path) in enumerate(training_files, start=1):
        samples, sample_rate = read_audio(DEFAULT_ARCHIVE, path)
        audio = to_mono(samples)

        row = {"class": class_name, "file": path.split("/")[-1]}
        row.update(compute_features(audio, sample_rate))
        rows.append(row)
        print(f"[{number}/{len(training_files)}] {class_name}: {row['file']}")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
