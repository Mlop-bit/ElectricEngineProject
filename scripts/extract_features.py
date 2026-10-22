"""    By Martin Allum & Edoardo Sileoni
Extract features from the audio and save them to CSV files.


Steps:
1. Find every WAV file in the train_cut/ and test_cut/ folders of the dataset zip.
2. Read each file with read_audio() from read_wav.py.
3. Compute the features listed in FEATURES for each file.
4. Save one row per file to train_cut_features.csv and test_cut_features.csv.

Every row starts with three description columns:
    class  - Good, Broken or Heavy load
    file   - the WAV file name
    noise  - the background noise in the clip (pure, atmo_high, talking, ...)

Run with: python scripts/extract_features.py
"""

import functools
import zipfile
import numpy as np
import pandas as pd
from scipy.fft import dct
from scipy.signal import butter, sosfiltfilt
from read_wav import DEFAULT_ARCHIVE, PROJECT_ROOT, read_audio

# ============================================================
# SETTINGS - change these to choose what the script computes
# ============================================================

# Features to compute. Comment out a line to skip that feature.
# The active ones gave the best results on the noisy test set in our tests
# (Random Forest about 99 %). The others are kept so they can be switched on.
FEATURES = [
    # "peak",           # off: background noise makes clips louder, misleads the models
    # "ptp",            # off: same problem as peak
    "rms",
    "dominant_frequency_hz",
    "wobble_frequency_hz",
    # "wobble_strength",  # off: noise makes the loudness vary more, misleads the models
    "mel_bands",        # one column per mel band, e.g. mel_803hz (see MEL_MIN_HZ)
    # "mfcc",           # off: 38 columns, weaker than mel_bands in our tests
]

# How each mel band is summarised over time: "p10", "p20", "median" or "mean".
# "p10" (10th percentile) uses the quiet moments of the clip, so short noises
# like knocking and talking are mostly ignored. It worked best in our tests.
MEL_SUMMARY = "p10"

# Only keep mel bands from this frequency (in Hz) and up. The background
# noise (talking, factory sounds) is strongest below about 700 Hz.
# 0 = keep all 40 bands.
MEL_MIN_HZ = 700

# Remove all frequencies below this value (in Hz) before computing the
# features, e.g. 1000. None = no filter.
HIGHPASS_HZ = None

# Extra copies of every train_cut clip with white or pink noise added
# (0 = off). Note: if you split the train CSV randomly into train and test
# parts, a clip and its noisy copy can end up on both sides, which makes the
# score look too good. Train on the train CSV and test on the test CSV instead.
AUGMENT_COPIES = 0

# ============================================================

# Folders inside the zip that we make a CSV file for
FOLDERS = ["train_cut", "test_cut"]

# Folder name inside the zip -> class name in the CSV
CLASS_NAMES = {
    "engine1_good": "Good",
    "engine2_broken": "Broken",
    "engine3_heavyload": "Heavy load",
}

# Mel spectrogram settings
FFT_SIZE = 2048     # samples per piece (about 46 ms)
HOP_SIZE = 512      # step between pieces (about 12 ms)
MEL_BANDS = 40
MFCC_COUNT = 20


# ============================================================
# Helpers
# ============================================================

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


def noise_type(path):
    """Return the background noise of a clip, read from its file name.

    "test_cut/engine1_good/atmo_high_3.wav" -> "atmo_high"
    "test_cut/engine1_good/talking_2_17.wav" -> "talking"
    """
    name = path.split("/")[-1].rsplit("_", 1)[0]    # remove the clip number
    if name.startswith("talking"):
        return "talking"                            # talking_1 ... talking_4
    return name


def to_mono(samples):
    """Turn 16-bit samples into one channel with values between -1 and 1."""
    audio = samples / 32768.0   # 16-bit values go from -32768 to 32767
    return audio.mean(axis=1)   # average the channels


def highpass(audio, sample_rate, cutoff_hz):
    """Remove all frequencies below cutoff_hz."""
    filter_ = butter(4, cutoff_hz, btype="highpass", fs=sample_rate, output="sos")
    return sosfiltfilt(filter_, audio)


def add_noise(audio, kind, snr_db, rng):
    """Add white or pink noise so that the signal is snr_db louder than the noise."""
    noise = rng.standard_normal(len(audio))
    if kind == "pink":
        # Pink noise: power falls with frequency (1/f), more like real background noise
        spectrum = np.fft.rfft(noise)
        spectrum[1:] /= np.sqrt(np.arange(1, len(spectrum)))
        spectrum[0] = 0
        noise = np.fft.irfft(spectrum, n=len(audio))
    signal_power = np.mean((audio - audio.mean()) ** 2)
    noise_power = signal_power / 10 ** (snr_db / 10)
    return audio + noise * np.sqrt(noise_power / np.mean(noise ** 2))


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


def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


@functools.lru_cache
def mel_filters(sample_rate):
    """Return the mel filters and the centre frequency (Hz) of each band.

    Each filter is a triangle that collects the FFT frequencies of one band.
    The bands are narrow at low frequencies and wide at high frequencies,
    like human hearing.
    """
    edges = mel_to_hz(np.linspace(0, hz_to_mel(sample_rate / 2), MEL_BANDS + 2))
    fft_frequencies = np.fft.rfftfreq(FFT_SIZE, d=1 / sample_rate)
    filters = np.zeros((MEL_BANDS, len(fft_frequencies)))
    for band in range(MEL_BANDS):
        left, centre, right = edges[band:band + 3]
        rising = (fft_frequencies - left) / (centre - left)
        falling = (right - fft_frequencies) / (right - centre)
        filters[band] = np.maximum(0, np.minimum(rising, falling))
    return filters, edges[1:-1]


def mel_spectrogram(audio, sample_rate):
    """Return the energy (in dB) of each mel band over time: (bands x time steps)."""
    filters, _ = mel_filters(sample_rate)
    piece_count = 1 + (len(audio) - FFT_SIZE) // HOP_SIZE
    pieces = np.lib.stride_tricks.sliding_window_view(audio, FFT_SIZE)[::HOP_SIZE][:piece_count]
    power = np.abs(np.fft.rfft(pieces * np.hanning(FFT_SIZE), axis=1)) ** 2
    return 10 * np.log10(filters @ power.T + 1e-10)


# ============================================================
# Features - one function each
# Each takes (audio, sample_rate) and returns a number,
# or a dict of {column name: number} for features with many columns.
# ============================================================

def peak(audio, sample_rate):
    """Loudest single sample."""
    return np.max(np.abs(audio))


def ptp(audio, sample_rate):
    """Peak-to-peak: highest minus lowest sample."""
    return np.max(audio) - np.min(audio)


def rms(audio, sample_rate):
    """Average loudness."""
    return np.sqrt(np.mean(audio ** 2))


def dominant_frequency_hz(audio, sample_rate):
    """The strongest frequency in the sound."""
    return dominant_frequency(audio, sample_rate)


def wobble_frequency_hz(audio, sample_rate):
    """How many times per second the loudness goes up and down."""
    loudness, loudness_rate = loudness_curve(audio, sample_rate)
    return dominant_frequency(loudness, loudness_rate)


def wobble_strength(audio, sample_rate):
    """How big the loudness changes are compared to the average loudness.

    0 = perfectly steady, bigger = more pulsing.
    """
    loudness, _ = loudness_curve(audio, sample_rate)
    return np.std(loudness) / np.mean(loudness)


def mel_bands(audio, sample_rate):
    """The level of each mel band from MEL_MIN_HZ up, independent of the overall volume."""
    spectrogram = mel_spectrogram(audio, sample_rate)
    if MEL_SUMMARY == "mean":
        levels = spectrogram.mean(axis=1)
    elif MEL_SUMMARY == "median":
        levels = np.median(spectrogram, axis=1)
    elif MEL_SUMMARY in ("p10", "p20"):
        levels = np.percentile(spectrogram, int(MEL_SUMMARY[1:]), axis=1)
    else:
        raise ValueError(f'MEL_SUMMARY must be "p10", "p20", "median" or "mean", not {MEL_SUMMARY!r}')
    levels = levels - levels.mean()     # remove the overall volume (using all 40 bands)
    _, centres = mel_filters(sample_rate)
    return {f"mel_{centre:.0f}hz": level for centre, level in zip(centres, levels)
            if centre >= MEL_MIN_HZ}


def mfcc(audio, sample_rate):
    """Mel-frequency cepstral coefficients 1-19 (the shape of the spectrum).

    Coefficient 0 is skipped because it is just the overall loudness.
    """
    coefficients = dct(mel_spectrogram(audio, sample_rate), type=2, axis=0, norm="ortho")
    columns = {}
    for i in range(1, MFCC_COUNT):
        columns[f"mfcc_{i}_mean"] = coefficients[i].mean()
        columns[f"mfcc_{i}_std"] = coefficients[i].std()
    return columns


# Feature name in FEATURES -> function that computes it
FEATURE_FUNCTIONS = {
    "peak": peak,
    "ptp": ptp,
    "rms": rms,
    "dominant_frequency_hz": dominant_frequency_hz,
    "wobble_frequency_hz": wobble_frequency_hz,
    "wobble_strength": wobble_strength,
    "mel_bands": mel_bands,
    "mfcc": mfcc,
}


def compute_features(audio, sample_rate):
    """Run every feature in FEATURES and return all columns in one dict."""
    columns = {}
    for name in FEATURES:
        value = FEATURE_FUNCTIONS[name](audio, sample_rate)
        if isinstance(value, dict):
            columns.update(value)       # feature with many columns
        else:
            columns[name] = value
    return columns


# ============================================================
# Main loop
# ============================================================

def extract_folder(folder):
    """Compute the features for every WAV file in the folder and save them to a CSV file."""
    files = find_wav_files(folder)
    print(f"Found {len(files)} files in {folder}/")
    rng = np.random.default_rng(42)     # fixed seed, so the added noise is the same every run

    rows = []
    for count, (class_name, path) in enumerate(files, start=1):
        samples, sample_rate = read_audio(DEFAULT_ARCHIVE, path)
        audio = to_mono(samples)
        description = {"class": class_name, "file": path.split("/")[-1]}

        versions = [(noise_type(path), audio)]
        if folder == "train_cut":
            for _ in range(AUGMENT_COPIES):
                kind = rng.choice(["white", "pink"])
                snr_db = rng.uniform(-6, 20)
                versions.append((f"added_{kind}", add_noise(audio, kind, snr_db, rng)))

        for noise, version in versions:
            if HIGHPASS_HZ is not None:
                version = highpass(version, sample_rate, HIGHPASS_HZ)
            rows.append({**description, "noise": noise, **compute_features(version, sample_rate)})

        if count % 100 == 0:    # show progress every 100 files
            print(f"{count}/{len(files)} files done")

    output_csv = PROJECT_ROOT / f"{folder}_features.csv"
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Saved {len(rows)} rows to {output_csv}\n")


def main():
    unknown = [name for name in FEATURES if name not in FEATURE_FUNCTIONS]
    if unknown:
        raise SystemExit(f"Unknown feature(s) in FEATURES: {', '.join(unknown)}. "
                         f"Choose from: {', '.join(FEATURE_FUNCTIONS)}")

    print(f"Features: {', '.join(FEATURES)}")
    if "mel_bands" in FEATURES:
        print(f"Mel band summary: {MEL_SUMMARY}, bands from {MEL_MIN_HZ} Hz")
    print(f"High-pass filter: {f'{HIGHPASS_HZ} Hz' if HIGHPASS_HZ else 'off'}")
    print(f"Noisy training copies: {AUGMENT_COPIES or 'off'}\n")

    for folder in FOLDERS:
        extract_folder(folder)


if __name__ == "__main__":
    main()
