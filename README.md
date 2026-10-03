# Electric Engine Project

Utilities for downloading and exploring the IDMT-ISA Electric Engine dataset.

## Project layout

```text
ElectricEngineProject/
├── data/                 # Dataset archive (not tracked in Git)
├── scripts/              # Python utilities
│   ├── download_dataset.py
│   └── read_wav.py
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

Both scripts locate the `data/` directory relative to their own location, so these commands also work when launched from another current directory.

Tips:

To close every plot : use Ctrl+C