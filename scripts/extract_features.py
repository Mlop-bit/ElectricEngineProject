"""    By Martin Allum & Edoardo Sileoni
Extract simple features from the audio and save them to CSV files.


Steps:
1. Find every WAV file in the train_cut/ and test_cut/ folders of the dataset zip.
2. Read each file with read_audio() from read_wav.py.
3. Compute six features for each file.
4. Save one row per file to train_cut_features.csv and test_cut_features.csv.

Run with: python scripts/extract_features.py
"""

import zipfile

import numpy as np
import pandas as pd

from read_wav import DEFAULT_ARCHIVE, PROJECT_ROOT, read_audio

# Folders inside the zip that we make a CSV file for
FOLDERS = ["train_cut", "test_cut"]

# Folder name inside the zip -> class name in the CSV
CLASS_NAMES = {
    "engine1_good": "Good",
    "engine2_broken": "Broken",
    "engine3_heavyload": "Heavy load",
}


def find_wav_files(folder):
    """Return a list of (class name, path) for every WAV file in the folder."""
    files = []
    with zipfile.ZipFile(DEFAULT_ARCHIVE) as archive:
        for path in sorted(archive.namelist()):
            # Example path: "train_cut/engine1_good/pure_0.wav"
            parts = path.split("/")
            if path.endswith(".wav") and parts[0] == folder:
                engine = parts[1]
                files.append((CLASS_NAMES[engine], path))
    return files


def to_mono(samples):
    """Turn 16-bit stereo samples into one channel with values between -1 and 1."""
    audio = samples / 32768.0   # 16-bit values go from -32768 to 32767
    return audio.mean(axis=1)   # average the channels


def dominant_frequency(audio, sample_rate):
    """Return the frequency (in Hz) that is strongest in the sound."""
    audio = audio - audio.mean()               # remove the DC offset
    audio = audio * np.hanning(len(audio))     # window to reduce FFT leakage
    strengths = np.abs(np.fft.rfft(audio))
    frequencies = np.fft.rfftfreq(len(audio), d=1 / sample_rate)
    strengths[0] = 0                           # ignore 0 Hz
    return frequencies[np.argmax(strengths)]


def loudness_curve(audio, sample_rate):
    """Cut the sound into 5 ms pieces and return the loudness (RMS) of each piece.

    The result is a new, much shorter signal that shows how the loudness
    changes over time. It has 200 values per second instead of 44100.
    """
    block_size = sample_rate // 200    # number of samples in 5 ms
    loudness = []
    for start in range(0, len(audio) - block_size + 1, block_size):
        block = audio[start:start + block_size]
        loudness.append(np.sqrt(np.mean(block ** 2)))
    return np.array(loudness), sample_rate / block_size


def extract_folder(folder):
    """Compute the features for every WAV file in the folder and save them to a CSV file."""
    files = find_wav_files(folder)
    print(f"Found {len(files)} files in {folder}/")

    rows = []
    for class_name, path in files:
        samples, sample_rate = read_audio(DEFAULT_ARCHIVE, path)
        audio = to_mono(samples)
        loudness, loudness_rate = loudness_curve(audio, sample_rate)

        rows.append({
            "class": class_name,
            "file": path.split("/")[-1],
            "peak": np.max(np.abs(audio)),                # loudest single sample
            "ptp": np.max(audio) - np.min(audio),         # peak-to-peak
            "rms": np.sqrt(np.mean(audio ** 2)),          # average loudness
            "dominant_frequency_hz": dominant_frequency(audio, sample_rate),
            # How many times per second the loudness goes up and down
            "wobble_frequency_hz": dominant_frequency(loudness, loudness_rate),
            # How big the loudness changes are compared to the average loudness
            # (0 = perfectly steady, bigger = more pulsing)
            "wobble_strength": np.std(loudness) / np.mean(loudness),
        })
        if len(rows) % 100 == 0:    # show progress every 100 files
            print(f"{len(rows)}/{len(files)} files done")

    output_csv = PROJECT_ROOT / f"{folder}_features.csv"
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Saved {len(rows)} rows to {output_csv}\n")


def main():
    for folder in FOLDERS:
        extract_folder(folder)


if __name__ == "__main__":
    main()
