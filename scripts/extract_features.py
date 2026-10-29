import zipfile
import numpy as np
import pandas as pd
from scipy.fft import dct
from scipy.signal import butter, sosfiltfilt
from read_wav import DEFAULT_ARCHIVE, PROJECT_ROOT, read_audio


# Settings: feature extraction

# Choose which measurements to save and how to process the audio.
FEATURES = [
    "peak",
    "ptp",
    "rms",
    "dominant_frequency_hz",
    "wobble_frequency_hz",
    "wobble_strength",
    "mel_bands",
    "mfcc",
]

MEL_PERCENTILE = 10
MEL_MIN_HZ = 0

FFT_SIZE = 2048
HOP_SIZE = 512
MEL_BANDS = 40
MFCC_COUNT = 20


# Settings: noise alterations

HIGHPASS_HZ = None

# True = add noisy copies of every training clip.
ADD_NOISY_COPIES = True

# One extra copy of every training clip per kind in this list.
# "white", "pink" (uses make_pink) or "both" (white and pink mixed together).
NOISE_KINDS = ["white", "pink", "both"]

# How loud the sound is compared to the added noise, in dB.
# Each copy gets a random value between these two. Lower = more noise.
NOISE_SNR_MIN_DB = -15
NOISE_SNR_MAX_DB = 10


# Settings: files

FOLDERS = ["train_cut", "test_cut"]

CLASS_NAMES = {
    "engine1_good": "Good",
    "engine2_broken": "Broken",
    "engine3_heavyload": "Heavy load",
}


# Reading the files

def find_wav_files(folder):
    files = []
    # Keep only WAV files directly inside the requested archive folder.
    with zipfile.ZipFile(DEFAULT_ARCHIVE) as archive:
        for path in sorted(archive.namelist()):
            parts = path.split("/")
            if path.endswith(".wav") and parts[0] == folder:
                engine = parts[1]
                files.append((CLASS_NAMES[engine], path))
    return files


def noise_type(path):
    file_name = path.split("/")[-1]
    name = file_name.rsplit("_", 1)[0]
    if name.startswith("talking"):
        return "talking"
    return name


# Noise alterations

def highpass(audio, sample_rate, cutoff_hz):
    sound_filter = butter(4, cutoff_hz, btype="highpass", fs=sample_rate, output="sos")
    return sosfiltfilt(sound_filter, audio)


def make_pink(noise):
    # Reduce high-frequency noise to give it a pink-noise profile.
    spectrum = np.fft.rfft(noise)
    for i in range(1, len(spectrum)):
        spectrum[i] = spectrum[i] / np.sqrt(i)
    spectrum[0] = 0
    return np.fft.irfft(spectrum, n=len(noise))


def make_noise(kind, length, rng):
    if kind == "white":
        return rng.standard_normal(length)
    if kind == "pink":
        return make_pink(rng.standard_normal(length))
    if kind == "both":
        white = make_noise("white", length, rng)
        pink = make_noise("pink", length, rng)
        # Give both the same strength before mixing, so neither one drowns out the other.
        white = white / np.sqrt(np.mean(white ** 2))
        pink = pink / np.sqrt(np.mean(pink ** 2))
        return white + pink
    raise ValueError(f'Unknown noise kind: {kind}. Use "white", "pink" or "both".')


def add_noise(audio, kind, snr_db, rng):
    noise = make_noise(kind, len(audio), rng)

    signal_power = np.mean((audio - audio.mean()) ** 2)
    # Scale the added noise to match the requested signal-to-noise ratio.
    wanted_noise_power = signal_power / 10 ** (snr_db / 10)

    current_noise_power = np.mean(noise ** 2)
    scale = np.sqrt(wanted_noise_power / current_noise_power)
    return audio + noise * scale


# Feature extraction: helpers

def dominant_frequency(audio, sample_rate):
    audio = audio - audio.mean()
    audio = audio * np.hanning(len(audio))
    strengths = np.abs(np.fft.rfft(audio))
    frequencies = np.fft.rfftfreq(len(audio), d=1 / sample_rate)
    strengths[0] = 0
    strongest = np.argmax(strengths)
    return frequencies[strongest]


def loudness_curve(audio, sample_rate):
    block_size = sample_rate // 200
    loudness = []
    for start in range(0, len(audio) - block_size + 1, block_size):
        block = audio[start:start + block_size]
        loudness.append(np.sqrt(np.mean(block ** 2)))
    values_per_second = sample_rate / block_size
    return np.array(loudness), values_per_second


def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


def mel_band_edges(sample_rate):
    highest_mel = hz_to_mel(sample_rate / 2)
    mel_edges = np.linspace(0, highest_mel, MEL_BANDS + 2)
    return mel_to_hz(mel_edges)


def mel_filters(sample_rate):
    edges = mel_band_edges(sample_rate)
    fft_frequencies = np.fft.rfftfreq(FFT_SIZE, d=1 / sample_rate)

    filters = []
    # Build overlapping triangular filters spaced on the Mel scale.
    for band in range(MEL_BANDS):
        left = edges[band]
        centre = edges[band + 1]
        right = edges[band + 2]

        weights = np.zeros(len(fft_frequencies))
        for i in range(len(fft_frequencies)):
            frequency = fft_frequencies[i]
            if left < frequency <= centre:
                weights[i] = (frequency - left) / (centre - left)
            elif centre < frequency < right:
                weights[i] = (right - frequency) / (right - centre)
        filters.append(weights)
    return filters


def mel_spectrogram(audio, sample_rate):
    filters = mel_filters(sample_rate)
    spectrogram = []

    start = 0
    # Measure each short, overlapping audio segment.
    while start + FFT_SIZE <= len(audio):
        piece = audio[start:start + FFT_SIZE]
        piece = piece * np.hanning(FFT_SIZE)
        power = np.abs(np.fft.rfft(piece)) ** 2

        band_energies = []
        for band_filter in filters:
            energy = np.sum(band_filter * power)
            band_energies.append(10 * np.log10(energy + 1e-10))

        spectrogram.append(band_energies)
        start += HOP_SIZE

    return np.array(spectrogram).T


# Feature extraction: the features

def peak(audio):
    return np.max(np.abs(audio))


def ptp(audio):
    return np.max(audio) - np.min(audio)


def rms(audio):
    return np.sqrt(np.mean(audio ** 2))


def wobble_frequency(audio, sample_rate):
    loudness, values_per_second = loudness_curve(audio, sample_rate)
    return dominant_frequency(loudness, values_per_second)


def wobble_strength(audio, sample_rate):
    loudness, _ = loudness_curve(audio, sample_rate)
    return np.std(loudness) / np.mean(loudness)


def mel_bands(audio, sample_rate):
    spectrogram = mel_spectrogram(audio, sample_rate)

    levels = []
    # Summarize each band using a low percentile to reduce brief loud sounds.
    for band in range(MEL_BANDS):
        levels.append(np.percentile(spectrogram[band], MEL_PERCENTILE))

    average = np.mean(levels)

    edges = mel_band_edges(sample_rate)
    columns = {}
    for band in range(MEL_BANDS):
        centre = edges[band + 1]
        if centre >= MEL_MIN_HZ:
            column_name = f"mel_{centre:.0f}hz"
            columns[column_name] = levels[band] - average
    return columns


def mfcc(audio, sample_rate):
    spectrogram = mel_spectrogram(audio, sample_rate)
    coefficients = dct(spectrogram, type=2, axis=0, norm="ortho")

    columns = {}
    for i in range(1, MFCC_COUNT):
        columns[f"mfcc_{i}_mean"] = np.mean(coefficients[i])
        columns[f"mfcc_{i}_std"] = np.std(coefficients[i])
    return columns


def compute_features(audio, sample_rate):
    # Calculate only the features selected in FEATURES.
    features = {}

    if "peak" in FEATURES:
        features["peak"] = peak(audio)
    if "ptp" in FEATURES:
        features["ptp"] = ptp(audio)
    if "rms" in FEATURES:
        features["rms"] = rms(audio)
    if "dominant_frequency_hz" in FEATURES:
        features["dominant_frequency_hz"] = dominant_frequency(audio, sample_rate)
    if "wobble_frequency_hz" in FEATURES:
        features["wobble_frequency_hz"] = wobble_frequency(audio, sample_rate)
    if "wobble_strength" in FEATURES:
        features["wobble_strength"] = wobble_strength(audio, sample_rate)

    if "mel_bands" in FEATURES:
        bands = mel_bands(audio, sample_rate)
        for name in bands:
            features[name] = bands[name]
    if "mfcc" in FEATURES:
        coefficients = mfcc(audio, sample_rate)
        for name in coefficients:
            features[name] = coefficients[name]

    return features


# Main loop

def make_row(class_name, file_name, noise, audio, sample_rate):
    row = {}
    row["class"] = class_name
    row["file"] = file_name
    row["noise"] = noise

    if HIGHPASS_HZ is not None:
        audio = highpass(audio, sample_rate, HIGHPASS_HZ)

    # Combine the file labels with the selected audio measurements.
    features = compute_features(audio, sample_rate)
    for name in features:
        row[name] = features[name]
    return row


def extract_folder(folder):
    files = find_wav_files(folder)
    print(f"Found {len(files)} files in {folder}/")
    rng = np.random.default_rng(42)

    rows = []
    count = 0
    for class_name, path in files:
        samples, sample_rate = read_audio(DEFAULT_ARCHIVE, path)
        audio = samples[:, 0] / 32768.0
        file_name = path.split("/")[-1]

        rows.append(make_row(class_name, file_name, noise_type(path), audio, sample_rate))

        # Add optional noisy copies to the training set only.
        if folder == "train_cut" and ADD_NOISY_COPIES:
            for kind in NOISE_KINDS:
                snr_db = rng.uniform(NOISE_SNR_MIN_DB, NOISE_SNR_MAX_DB)
                noisy_audio = add_noise(audio, kind, snr_db, rng)
                rows.append(make_row(class_name, file_name, f"added_{kind}", noisy_audio, sample_rate))

        count += 1
        if count % 100 == 0:
            print(f"{count}/{len(files)} files done")

    output_csv = PROJECT_ROOT / f"{folder}_features.csv"
    # Save one row per original file or augmented copy.
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Saved {len(rows)} rows to {output_csv}\n")


def main():
    print("Features:", ", ".join(FEATURES))
    if "mel_bands" in FEATURES:
        print(f"Mel band summary: percentile {MEL_PERCENTILE}, bands from {MEL_MIN_HZ} Hz")
    if HIGHPASS_HZ is None:
        print("High-pass filter: off")
    else:
        print(f"High-pass filter: {HIGHPASS_HZ} Hz")
    if ADD_NOISY_COPIES:
        print("Noisy training copies:", ", ".join(NOISE_KINDS))
        print(f"Noise level: {NOISE_SNR_MIN_DB} to {NOISE_SNR_MAX_DB} dB SNR\n")
    else:
        print("Noisy training copies: off\n")

    for folder in FOLDERS:
        extract_folder(folder)


if __name__ == "__main__":
    main()
