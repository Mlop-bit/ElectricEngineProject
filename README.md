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

Create a project-local environment and install the dependencies:

```powershell
uv venv .venv --python 3.11
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```
```MacOs
python -m pip install -r requirements.txt
```

Select `.venv\Scripts\python.exe` as the Python interpreter in VS Code before
running the scripts.

If a package fails with `DLL load failed ... En programkontrollpolicy har
blokkert denne filen`, Windows App Control is blocking native Python extension
modules. This can affect pandas, Matplotlib, and scikit-learn at paths such as
`<environment>\Lib\site-packages\pandas\_libs\*.pyd`,
`<environment>\Lib\site-packages\matplotlib\*.pyd`, and
`<environment>\Lib\site-packages\sklearn\*.pyd`. This cannot be fixed in the
Python script or by reinstalling packages without changing the policy. Ask
the administrator to approve this trusted project environment and its
dependencies, or run the project in an approved Python environment/machine.
Do not disable the security policy yourself.

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

The initial model is a small multilayer perceptron (MLP) that classifies each audio segment as `Good`, `Broken`, or `Heavy load`. It uses all numeric feature columns in the feature CSV, excluding the `class`, `file`, and `noise` metadata columns. The dataset's `train_cut` audio is read directly from the ZIP; it does not need to be extracted first.

Regenerate the feature CSV if needed, then train and evaluate the model:

```powershell
python scripts/extract_features.py
python scripts/train_model.py
```

The model trains on all rows in `train_cut_features.csv` and evaluates on `test_cut_features.csv`. It scales features using the training data only, prints an accuracy report and confusion matrix, and saves the trained pipeline to `models/engine_classifier.joblib`. To evaluate one test noise category at a time, pass `--test-noise`:

```powershell
python scripts/train_model.py --test-noise atmo_high
python scripts/train_model.py --test-noise atmo_low
python scripts/train_model.py --test-noise atmo_medium
python scripts/train_model.py --test-noise stresstest
python scripts/train_model.py --test-noise talking
python scripts/train_model.py --test-noise whitenoise_low
```

Each command uses the full training CSV and filters only the test CSV. This is a first baseline: each class currently comes from a specific engine recording, so a high score does not yet show that the model can diagnose a different physical motor.

Tips:

To close every plot : use Ctrl+C
