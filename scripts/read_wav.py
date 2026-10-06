"""List and plot WAV files stored in the project's dataset zip archive."""

from __future__ import annotations

import argparse
import struct
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARCHIVE = PROJECT_ROOT / "data" / "IDMT-ISA-ELECTRIC-ENGINE.zip"
DEFAULT_MEMBER = "train_cut/engine1_good/pure_0.wav"


def _read_audio_member(archive: zipfile.ZipFile, member: str) -> tuple[np.ndarray, int]:
    """Read one WAV member, returning its samples and sample rate."""
    try:
        import numpy as np
    except ModuleNotFoundError as error:
        if error.name == "numpy":
            raise SystemExit("NumPy is missing. Install the project dependencies from requirements.txt.") from error
        raise

    try:
        audio_bytes = archive.read(member)
    except KeyError as error:
        raise SystemExit(f"WAV file not found in archive: {member}") from error

    if audio_bytes[:4] != b"RIFF" or audio_bytes[8:12] != b"WAVE":
        raise ValueError(f"Unsupported WAV container in {member}")

    format_tag = channels = sample_rate = bits_per_sample = block_align = None
    audio_data = None
    offset = 12
    while offset + 8 <= len(audio_bytes):
        chunk_id = audio_bytes[offset:offset + 4]
        chunk_size = struct.unpack_from("<I", audio_bytes, offset + 4)[0]
        chunk_start = offset + 8
        chunk_end = chunk_start + chunk_size
        if chunk_end > len(audio_bytes):
            raise ValueError(f"Truncated WAV chunk in {member}")

        if chunk_id == b"fmt ":
            if chunk_size < 16:
                raise ValueError(f"Invalid WAV format chunk in {member}")
            format_tag, channels, sample_rate, _, block_align, bits_per_sample = struct.unpack_from(
                "<HHIIHH", audio_bytes, chunk_start
            )
            if format_tag == 0xFFFE and chunk_size >= 40:
                # WAVE_FORMAT_EXTENSIBLE: the subtype GUID starts at byte 24.
                format_tag = struct.unpack_from("<H", audio_bytes, chunk_start + 24)[0]
        elif chunk_id == b"data":
            audio_data = audio_bytes[chunk_start:chunk_end]

        offset = chunk_end + (chunk_size & 1)

    if format_tag is None or audio_data is None or not channels or not sample_rate:
        raise ValueError(f"WAV is missing format or audio data: {member}")

    sample_width = bits_per_sample // 8
    bytes_per_frame = block_align or channels * sample_width
    if sample_width * channels != bytes_per_frame:
        raise ValueError(f"Unsupported WAV frame layout in {member}")
    frame_count = len(audio_data) // bytes_per_frame
    frames = audio_data[:frame_count * bytes_per_frame]

    if format_tag == 3 and sample_width == 4:
        samples = np.frombuffer(frames, dtype="<f4")
    elif format_tag == 3 and sample_width == 8:
        samples = np.frombuffer(frames, dtype="<f8")
    elif format_tag != 1:
        raise ValueError(f"Unsupported WAV encoding {format_tag} in {member}")
    elif sample_width == 1:
        # 8-bit PCM WAV samples are unsigned.
        samples = np.frombuffer(frames, dtype=np.uint8).astype(np.int16) - 128
    elif sample_width == 2:
        samples = np.frombuffer(frames, dtype="<i2")
    elif sample_width == 3:
        # NumPy has no int24 type; sign-extend each little-endian sample.
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        values = raw[:, 0].astype(np.int32) | (raw[:, 1].astype(np.int32) << 8) | (raw[:, 2].astype(np.int32) << 16)
        samples = (values ^ 0x800000) - 0x800000
    elif sample_width == 4:
        samples = np.frombuffer(frames, dtype="<i4")
    else:
        raise SystemExit(f"Unsupported WAV sample width: {sample_width} bytes")

    return samples.reshape(-1, channels), sample_rate


def read_audio(archive_path: Path, member: str) -> tuple[np.ndarray, int]:
    """Return audio samples and sample rate for a WAV member in the archive."""
    with zipfile.ZipFile(archive_path) as archive:
        return _read_audio_member(archive, member)


def scan_all_audio(
    archive_path: Path, folder: str | None = None, plot: bool = False
) -> None:
    """Read matching WAV files one at a time and report overall details."""
    if plot:
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ModuleNotFoundError as error:
            if error.name in {"matplotlib", "numpy"}:
                raise SystemExit(
                    f"{error.name} is missing. Install the project dependencies from requirements.txt."
                ) from error
            raise

    total_files = 0
    total_seconds = 0.0
    prefix = folder.strip("/\\").replace("\\", "/") + "/" if folder else None
    with zipfile.ZipFile(archive_path) as archive:
        members = [
            name for name in archive.namelist()
            if name.lower().endswith(".wav")
            and (prefix is None or name.startswith(prefix))
        ]
        if not members:
            print(f"No WAV files found under {folder!r} in the archive." if folder else
                  "No WAV files found in the archive.")
            return

        scope = f" under {folder}/" if folder else ""
        plot_message = " with one plot at a time" if plot else " without plots"
        print(f"Reading {len(members)} WAV files{scope} one at a time{plot_message}.")
        for member in members:
            audio, sample_rate = _read_audio_member(archive, member)
            total_seconds += len(audio) / sample_rate
            total_files += 1
            if plot:
                time = np.arange(len(audio)) / sample_rate
                figure, axes = plt.subplots(figsize=(12, 4))
                for channel in range(audio.shape[1]):
                    label = f"Channel {channel + 1}" if audio.shape[1] > 1 else None
                    axes.plot(time, audio[:, channel], label=label, linewidth=0.5)
                axes.set_xlabel("Time (seconds)")
                axes.set_ylabel("Amplitude")
                axes.set_title(member)
                if audio.shape[1] > 1:
                    axes.legend()
                figure.tight_layout()
                plt.show()  # Close the window to continue to the next file.
                plt.close(figure)
            del audio  # Keep only one decoded file in memory at a time.
            if total_files % 100 == 0 or total_files == len(members):
                print(f"Read {total_files}/{len(members)} files...")

    print(f"Finished: {total_files} files, {total_seconds / 3600:.2f} total audio hours.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("member", nargs="?", default=DEFAULT_MEMBER,
                        help=f"WAV path inside the zip (default: {DEFAULT_MEMBER})")
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE,
                        help=f"dataset zip path (default: {DEFAULT_ARCHIVE})")
    parser.add_argument("--list", action="store_true", help="list WAV paths in the zip")
    parser.add_argument("--all", action="store_true",
                        help="read every WAV sequentially without plotting")
    parser.add_argument("--folder", help="limit --all to a folder inside the zip, e.g. test")
    parser.add_argument("--plot", action="store_true",
                        help="show and close a plot for each file (requires --all)")
    args = parser.parse_args()

    if not args.archive.is_file():
        parser.error(f"archive does not exist: {args.archive}")

    if args.list:
        with zipfile.ZipFile(args.archive) as archive:
            for name in archive.namelist():
                if name.lower().endswith(".wav"):
                    print(name)
        return

    if args.all:
        scan_all_audio(args.archive, args.folder, args.plot)
        return
    if args.folder or args.plot:
        parser.error("--folder and --plot can only be used together with --all")

    try:
        import numpy as np
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        if error.name == "numpy":
            raise SystemExit("NumPy is missing. Install the project dependencies from requirements.txt.") from error
        if error.name == "matplotlib":
            raise SystemExit("Matplotlib is missing. Install the project dependencies from requirements.txt.") from error
        raise

    audio, sample_rate = read_audio(args.archive, args.member)
    duration = len(audio) / sample_rate
    time = np.arange(len(audio)) / sample_rate
    print(f"{args.member}: {sample_rate} Hz, {audio.shape[1]} channel(s), {duration:.2f} seconds")

    plt.figure(figsize=(12, 4))
    for channel in range(audio.shape[1]):
        label = f"Channel {channel + 1}" if audio.shape[1] > 1 else None
        plt.plot(time, audio[:, channel], label=label, linewidth=0.5)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title(args.member)
    if audio.shape[1] > 1:
        plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
