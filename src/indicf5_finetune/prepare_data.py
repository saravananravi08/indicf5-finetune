"""
Download OpenSLR-104 Hindi-English code-switched dataset and convert it
to the F5-TTS CustomDatasetPath format (raw/ + duration.json).

Usage:
    # Via uv script
    uv run prepare-data --data-dir ./data

    # Or directly
    uv run python -m indicf5_finetune.prepare_data --data-dir ./data

Output structure:
    data_dir/
    ├── audio/              # resampled 24kHz mono WAV files (one per utterance)
    ├── raw/                # HuggingFace Dataset (arrow format)
    ├── duration.json       # {"duration": [d1, d2, ...]}
    └── vocab.txt           # IndicF5 vocabulary (Indic + Latin chars)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile

import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OPENSLR104_URL = "https://us.openslr.org/resources/104/Hindi-English_train.tar.gz"
OPENSLR104_TEST_URL = "https://us.openslr.org/resources/104/Hindi-English_test.tar.gz"
TARGET_SR = 24_000
MIN_DURATION = 0.3
MAX_DURATION = 30.0


def download_file(url: str, dest: str):
    """Download a file using wget or curl."""
    if os.path.exists(dest):
        print(f"  Already downloaded: {dest}")
        return
    print(f"  Downloading {url} ...")
    try:
        subprocess.run(
            ["wget", "--no-check-certificate", "-q", "--show-progress", "-O", dest, url],
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  wget failed, trying curl ...")
        subprocess.run(
            ["curl", "-k", "-L", "-o", dest, url, "--progress-bar"],
            check=True,
        )


def extract_tarball(tar_path: str, dest_dir: str):
    """Extract a tar.gz archive."""
    print(f"  Extracting {tar_path} ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=dest_dir)


def parse_kaldi_text(text_path: str) -> dict:
    """Parse Kaldi 'text' file: utt_id transcript"""
    transcripts = {}
    with open(text_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                transcripts[parts[0]] = parts[1]
    return transcripts


def parse_kaldi_segments(segments_path: str) -> list:
    """Parse Kaldi 'segments' file: utt_id recording_id start_time end_time

    Returns list of (utt_id, recording_id, start_sec, end_sec).
    """
    segments = []
    with open(segments_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 4:
                utt_id = parts[0]
                rec_id = parts[1]
                start = float(parts[2])
                end = float(parts[3])
                segments.append((utt_id, rec_id, start, end))
    return segments


def find_kaldi_files(extracted_dir: str):
    """Find Kaldi-format files in extracted directory.

    OpenSLR-104 structure:
        train/
        ├── *.wav                   (full recordings)
        └── transcripts/
            ├── text                (utt_id -> transcript)
            ├── segments            (utt_id -> recording_id start end)
            ├── utt2spk
            └── spk2utt
    """
    text_path = None
    segments_path = None
    wav_dir = None

    for root, dirs, files in os.walk(extracted_dir):
        for f in files:
            full = os.path.join(root, f)
            if f == "text" and "transcripts" in root:
                text_path = full
            elif f == "segments":
                segments_path = full
        wav_files = [f for f in files if f.endswith(".wav")]
        if wav_files and wav_dir is None:
            wav_dir = root

    return text_path, segments_path, wav_dir


def build_dataset(data_dir: str, split: str = "train", max_samples: int = 0):
    """Build F5-TTS dataset from downloaded OpenSLR-104 data.

    OpenSLR-104 uses Kaldi format with segments:
    - WAV files are full recordings (multiple utterances each)
    - 'segments' file maps utterance IDs to (recording_id, start, end)
    - 'text' file maps utterance IDs to transcripts
    We slice each recording into individual utterance WAVs at 24kHz.
    """
    raw_dir = os.path.join(data_dir, "raw_download")
    audio_out_dir = os.path.join(data_dir, "audio")
    os.makedirs(audio_out_dir, exist_ok=True)

    # Download
    archive_name = f"Hindi-English_{split}.tar.gz"
    archive_path = os.path.join(raw_dir, archive_name)
    os.makedirs(raw_dir, exist_ok=True)

    url = OPENSLR104_URL if split == "train" else OPENSLR104_TEST_URL
    download_file(url, archive_path)

    # Extract
    extracted_dir = os.path.join(raw_dir, f"extracted_{split}")
    if not os.path.exists(extracted_dir) or not os.listdir(extracted_dir):
        if os.path.exists(extracted_dir):
            os.rmdir(extracted_dir)
        os.makedirs(extracted_dir, exist_ok=True)
        extract_tarball(archive_path, extracted_dir)
    else:
        print(f"  Already extracted: {extracted_dir}")

    # Find Kaldi files
    text_path, segments_path, wav_dir = find_kaldi_files(extracted_dir)

    if text_path is None:
        print("  ERROR: Could not find transcripts/text file")
        sys.exit(1)
    if wav_dir is None:
        print("  ERROR: No WAV files found")
        sys.exit(1)

    print(f"  Transcripts: {text_path}")
    print(f"  WAV dir:     {wav_dir}")

    # Parse transcripts
    transcripts = parse_kaldi_text(text_path)
    print(f"  Loaded {len(transcripts)} transcripts")

    # Parse segments (if available)
    if segments_path:
        print(f"  Segments:    {segments_path}")
        segments = parse_kaldi_segments(segments_path)
        print(f"  Loaded {len(segments)} segments")
    else:
        segments = None

    # Cache loaded recordings to avoid re-reading the same WAV multiple times
    recording_cache = {}

    def load_recording(rec_id):
        if rec_id in recording_cache:
            return recording_cache[rec_id]
        wav_path = os.path.join(wav_dir, f"{rec_id}.wav")
        if not os.path.exists(wav_path):
            return None, None
        try:
            audio, sr = sf.read(wav_path)
        except Exception as e:
            print(f"    WARN: Failed to read {wav_path}: {e}")
            return None, None
        # Convert to mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        recording_cache[rec_id] = (audio, sr)
        return audio, sr

    audio_paths = []
    texts = []
    durations = []
    skipped = 0

    if segments:
        # Kaldi segment-based extraction
        print(f"  Extracting {len(segments)} utterance segments ...")

        for i, (utt_id, rec_id, start, end) in enumerate(segments):
            text = transcripts.get(utt_id)
            if text is None:
                skipped += 1
                continue

            seg_duration = end - start
            if seg_duration < MIN_DURATION or seg_duration > MAX_DURATION:
                skipped += 1
                continue

            audio, sr = load_recording(rec_id)
            if audio is None:
                skipped += 1
                continue

            # Extract segment
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            segment_audio = audio[start_sample:end_sample]

            if len(segment_audio) == 0:
                skipped += 1
                continue

            # Resample to target sample rate
            if sr != TARGET_SR:
                try:
                    import torch
                    import torchaudio
                    audio_tensor = torch.from_numpy(segment_audio).float()
                    resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
                    audio_tensor = resampler(audio_tensor)
                    segment_audio = audio_tensor.numpy()
                except ImportError:
                    import librosa
                    segment_audio = librosa.resample(segment_audio, orig_sr=sr, target_sr=TARGET_SR)

            actual_duration = len(segment_audio) / TARGET_SR

            # Save utterance WAV
            dst_path = os.path.join(audio_out_dir, f"{utt_id}.wav")
            sf.write(dst_path, segment_audio, TARGET_SR)

            audio_paths.append(os.path.abspath(dst_path))
            texts.append(text)
            durations.append(round(actual_duration, 4))

            if max_samples > 0 and len(audio_paths) >= max_samples:
                print(f"    Reached max_samples={max_samples}, stopping early")
                break

            if (i + 1) % 5000 == 0:
                print(f"    Processed {i+1}/{len(segments)} segments ({skipped} skipped)")

        # Free recording cache
        recording_cache.clear()

    else:
        # Fallback: each WAV is a standalone utterance
        wav_files = sorted([f for f in os.listdir(wav_dir) if f.endswith(".wav")])
        print(f"  Processing {len(wav_files)} audio files ...")

        for i, wav_file in enumerate(wav_files):
            utt_id = wav_file.replace(".wav", "")

            text = transcripts.get(utt_id)
            if text is None:
                for key in transcripts:
                    if key.endswith(utt_id) or utt_id.endswith(key):
                        text = transcripts[key]
                        break

            if text is None:
                skipped += 1
                continue

            src_path = os.path.join(wav_dir, wav_file)
            dst_path = os.path.join(audio_out_dir, wav_file)

            try:
                audio, sr = sf.read(src_path)
            except Exception as e:
                print(f"    WARN: Failed to read {src_path}: {e}")
                skipped += 1
                continue

            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            if sr != TARGET_SR:
                try:
                    import torch
                    import torchaudio
                    audio_tensor = torch.from_numpy(audio).float()
                    resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
                    audio_tensor = resampler(audio_tensor)
                    audio = audio_tensor.numpy()
                except ImportError:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

            duration = len(audio) / TARGET_SR
            if duration < MIN_DURATION or duration > MAX_DURATION:
                skipped += 1
                continue

            sf.write(dst_path, audio, TARGET_SR)
            audio_paths.append(os.path.abspath(dst_path))
            texts.append(text)
            durations.append(round(duration, 4))

            if max_samples > 0 and len(audio_paths) >= max_samples:
                print(f"    Reached max_samples={max_samples}, stopping early")
                break

            if (i + 1) % 1000 == 0:
                print(f"    Processed {i+1}/{len(wav_files)} files ({skipped} skipped)")

    print(f"  Done: {len(audio_paths)} utterances, {skipped} skipped")
    print(f"  Total duration: {sum(durations)/3600:.1f} hours")

    return audio_paths, texts, durations


def save_arrow_dataset(audio_paths, texts, durations, data_dir):
    """Save as HuggingFace Dataset in Arrow format."""
    from datasets import Dataset

    dataset = Dataset.from_dict({
        "audio_path": audio_paths,
        "text": texts,
        "duration": durations,
    })

    dataset.save_to_disk(os.path.join(data_dir, "raw"))
    print(f"  Saved arrow dataset to {data_dir}/raw/")

    duration_path = os.path.join(data_dir, "duration.json")
    with open(duration_path, "w") as f:
        json.dump({"duration": durations}, f)
    print(f"  Saved {duration_path}")

    return dataset


def copy_vocab(data_dir: str):
    """Copy the IndicF5 vocab.txt to the data directory.

    The existing vocab already contains Latin characters (A-Z, a-z),
    Devanagari, and all Indic scripts - no modification needed.
    """
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub/models--ai4bharat--IndicF5")
    vocab_src = None
    for root, dirs, files in os.walk(hf_cache):
        if "vocab.txt" in files:
            vocab_src = os.path.join(root, "vocab.txt")
            break

    if vocab_src is None:
        print("  IndicF5 vocab not found in cache. Downloading...")
        from huggingface_hub import hf_hub_download
        vocab_src = hf_hub_download("ai4bharat/IndicF5", filename="checkpoints/vocab.txt")

    dst = os.path.join(data_dir, "vocab.txt")
    shutil.copy2(vocab_src, dst)
    print(f"  Copied vocab.txt ({sum(1 for _ in open(dst))} entries)")


def print_stats(audio_paths, texts, durations):
    """Print dataset statistics."""
    total_hrs = sum(durations) / 3600
    avg_dur = sum(durations) / len(durations)

    hindi_chars = 0
    english_chars = 0
    for text in texts:
        for c in text:
            if "\u0900" <= c <= "\u097F":
                hindi_chars += 1
            elif c.isascii() and c.isalpha():
                english_chars += 1

    mix_ratio = english_chars / max(hindi_chars + english_chars, 1) * 100

    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print(f"  Utterances:     {len(audio_paths):,}")
    print(f"  Total duration: {total_hrs:.1f} hours")
    print(f"  Avg duration:   {avg_dur:.1f}s")
    print(f"  Min duration:   {min(durations):.1f}s")
    print(f"  Max duration:   {max(durations):.1f}s")
    print(f"  English chars:  {english_chars:,} ({mix_ratio:.1f}%)")
    print(f"  Hindi chars:    {hindi_chars:,} ({100-mix_ratio:.1f}%)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Prepare OpenSLR-104 for IndicF5 fine-tuning")
    parser.add_argument("--data-dir", default="./data", help="Output directory")
    parser.add_argument("--split", default="train", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Limit number of samples (0=all, use 20 for test run)")
    parser.add_argument("--skip-download", action="store_true", help="Skip download if already done")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Preparing OpenSLR-104 Hindi-English dataset")
    print(f"Output: {data_dir}")
    print(f"{'='*60}\n")

    if args.max_samples > 0:
        print(f"*** TEST MODE: limiting to {args.max_samples} samples ***\n")

    print("[1/3] Downloading and processing audio ...")
    audio_paths, texts, durations = build_dataset(data_dir, split=args.split, max_samples=args.max_samples)

    print("\n[2/3] Saving Arrow dataset ...")
    save_arrow_dataset(audio_paths, texts, durations, data_dir)

    print("\n[3/3] Copying vocabulary ...")
    copy_vocab(data_dir)

    print_stats(audio_paths, texts, durations)

    print(f"\nReady for fine-tuning! Run:")
    print(f"  uv run train --data-dir {data_dir}")


if __name__ == "__main__":
    main()
