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
    ├── audio/              # resampled 24kHz mono WAV files
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
    """Download a file using wget (handles large files better than Python)."""
    if os.path.exists(dest):
        print(f"  Already downloaded: {dest}")
        return
    print(f"  Downloading {url} ...")
    subprocess.run(
        ["wget", "-q", "--show-progress", "-O", dest, url],
        check=True,
    )


def extract_tarball(tar_path: str, dest_dir: str):
    """Extract a tar.gz archive."""
    print(f"  Extracting {tar_path} ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=dest_dir)


def find_transcript_file(extracted_dir: str) -> str | None:
    """Find the transcript/text file in extracted OpenSLR-104 data."""
    for root, dirs, files in os.walk(extracted_dir):
        for f in files:
            if f in ("text", "line_index.tsv", "transcription.tsv"):
                return os.path.join(root, f)
            if f.endswith(".txt") and "transcript" in f.lower():
                return os.path.join(root, f)
    for root, dirs, files in os.walk(extracted_dir):
        for f in files:
            if f == "text":
                return os.path.join(root, f)
    return None


def find_wav_dir(extracted_dir: str) -> str | None:
    """Find directory containing WAV files."""
    for root, dirs, files in os.walk(extracted_dir):
        wav_files = [f for f in files if f.endswith(".wav")]
        if wav_files:
            return root
    return None


def parse_transcripts(transcript_path: str) -> dict:
    """Parse transcript file into {utt_id: text} dict.

    OpenSLR-104 uses Kaldi-style 'text' format: utterance_id transcript_text
    """
    transcripts = {}
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                parts = line.split("\t", 1)
            else:
                parts = line.split(None, 1)
            if len(parts) == 2:
                utt_id, text = parts
                transcripts[utt_id] = text.strip()
    return transcripts


def resample_audio(src_path: str, dst_path: str, target_sr: int = TARGET_SR):
    """Load audio, convert to mono 24kHz, save as WAV."""
    try:
        audio, sr = sf.read(src_path)
    except Exception as e:
        print(f"    WARN: Failed to read {src_path}: {e}")
        return None, None

    # Convert to mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        try:
            import torch
            import torchaudio
            audio_tensor = torch.from_numpy(audio).float()
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            audio_tensor = resampler(audio_tensor)
            audio = audio_tensor.numpy()
        except ImportError:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    duration = len(audio) / target_sr

    if duration < MIN_DURATION or duration > MAX_DURATION:
        return None, None

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    sf.write(dst_path, audio, target_sr)
    return audio, duration


def build_dataset(data_dir: str, split: str = "train"):
    """Build F5-TTS dataset from downloaded OpenSLR-104 data."""
    raw_dir = os.path.join(data_dir, "raw_download")
    audio_out_dir = os.path.join(data_dir, "audio")
    os.makedirs(audio_out_dir, exist_ok=True)

    archive_name = f"Hindi-English_{split}.tar.gz"
    archive_path = os.path.join(raw_dir, archive_name)
    os.makedirs(raw_dir, exist_ok=True)

    url = OPENSLR104_URL if split == "train" else OPENSLR104_TEST_URL
    download_file(url, archive_path)

    extracted_dir = os.path.join(raw_dir, f"extracted_{split}")
    if not os.path.exists(extracted_dir):
        os.makedirs(extracted_dir, exist_ok=True)
        extract_tarball(archive_path, extracted_dir)
    else:
        print(f"  Already extracted: {extracted_dir}")

    transcript_path = find_transcript_file(extracted_dir)
    if transcript_path is None:
        print("  Could not find transcript file. Listing extracted contents:")
        for root, dirs, files in os.walk(extracted_dir):
            for f in files[:20]:
                print(f"    {os.path.join(root, f)}")
        sys.exit(1)

    print(f"  Found transcripts: {transcript_path}")
    transcripts = parse_transcripts(transcript_path)
    print(f"  Loaded {len(transcripts)} transcripts")

    wav_dir = find_wav_dir(extracted_dir)
    if wav_dir is None:
        print("  ERROR: No WAV files found in extracted data!")
        sys.exit(1)
    print(f"  Found audio dir: {wav_dir}")

    audio_paths = []
    texts = []
    durations = []
    skipped = 0

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
        audio, duration = resample_audio(src_path, dst_path)

        if audio is None:
            skipped += 1
            continue

        audio_paths.append(os.path.abspath(dst_path))
        texts.append(text)
        durations.append(round(duration, 4))

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
    parser.add_argument("--skip-download", action="store_true", help="Skip download if already done")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Preparing OpenSLR-104 Hindi-English dataset")
    print(f"Output: {data_dir}")
    print(f"{'='*60}\n")

    print("[1/3] Downloading and processing audio ...")
    audio_paths, texts, durations = build_dataset(data_dir, split=args.split)

    print("\n[2/3] Saving Arrow dataset ...")
    save_arrow_dataset(audio_paths, texts, durations, data_dir)

    print("\n[3/3] Copying vocabulary ...")
    copy_vocab(data_dir)

    print_stats(audio_paths, texts, durations)

    print(f"\nReady for fine-tuning! Run:")
    print(f"  uv run train --data-dir {data_dir}")


if __name__ == "__main__":
    main()
