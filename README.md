# Electric Engine Project

Utilities for downloading and exploring the IDMT-ISA Electric Engine dataset.

## Project layout

```text
ElectricEngineProject/
├── data/                 # Dataset archive (not tracked in Git)
├── scripts/              # Python utilities
│   ├── download_dataset.py
│   ├── extract_features.py
│   ├── read_wav.py
│   └── train_model.py
├── train_cut_features.csv # Extracted training features
├── models/                # Locally trained model (not tracked in Git)
├── requirements.txt
└── README.md
```

## Setup

Install the dependencies into the project's Python environment:

```powershell
python -m pip install -r requirements.txt
```

## Use

Download the dataset (the script skips an existing archive unless `--force` is used):

```powershell
python scripts/download_dataset.py
```

List WAV files in the archive:

```powershell
python scripts/read_wav.py --list
```

Read every WAV sequentially, without opening plots or retaining all audio in memory:

```powershell
python scripts/read_wav.py --all
```

Limit the scan to WAV files under `test/` in the ZIP:

```powershell
python scripts/read_wav.py --all --folder test
```

To display each file's waveform in sequence, close the current plot window to continue:

```powershell
python scripts/read_wav.py --all --folder test --plot
```

Plot the default audio sample, or choose a WAV path from the list:

```powershell
python scripts/read_wav.py
python scripts/read_wav.py train_cut/engine2_broken/pure_0.wav
```

The scripts locate the `data/` directory relative to their own location, so they can be launched from another current directory.

## Train the neural-network classifier

The initial model is a small multilayer perceptron (MLP) that classifies each audio segment as `Good`, `Broken`, or `Heavy load`. It uses four features: peak amplitude, peak-to-peak amplitude, RMS, and dominant frequency. The dataset's `train_cut` audio is read directly from the ZIP; it does not need to be extracted first.

Regenerate the feature CSV if needed, then train and evaluate the model:

```powershell
python scripts/extract_features.py
python scripts/train_model.py
```

The evaluation keeps neighboring groups of ten audio cuts together in the train/test split, scales features using training data only, prints an accuracy report and confusion matrix, and saves the trained pipeline to `models/engine_classifier.joblib`. This is a first baseline: each class currently comes from a specific engine recording, so a high score does not yet show that the model can diagnose a different physical motor.

Tips:

To close every plot : use Ctrl+C
