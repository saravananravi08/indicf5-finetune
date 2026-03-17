"""
Evaluate a fine-tuned IndicF5 checkpoint on English/Hinglish/Hindi test sentences.

Generates audio for a set of test sentences and saves them as WAV files.
Optionally compares with the base IndicF5 model.

Usage:
    # Evaluate fine-tuned checkpoint
    uv run evaluate --checkpoint-dir ./checkpoints

    # Compare with base model
    uv run evaluate --checkpoint-dir ./checkpoints --compare-base

    # Use specific checkpoint step
    uv run evaluate --checkpoint-dir ./checkpoints --step 10000
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

from f5_tts.model import CFM, DiT
from f5_tts.model.utils import get_tokenizer
from f5_tts.infer.utils_infer import (
    infer_process,
    load_vocoder,
    preprocess_ref_audio_text,
)


SAMPLE_RATE = 24_000

INDICF5_MODEL_CFG = dict(
    dim=1024,
    depth=22,
    heads=16,
    ff_mult=2,
    text_dim=512,
    conv_layers=4,
)

# Test sentences covering different language mixing patterns
TEST_SENTENCES = {
    "hindi_pure": [
        "नमस्ते, आज का मौसम बहुत अच्छा है।",
        "भारत एक विशाल देश है जहाँ अनेक भाषाएँ बोली जाती हैं।",
        "कृपया अपना नाम और पता बताइए।",
    ],
    "hinglish_light": [
        "मैं आज office जा रहा हूँ, meeting है।",
        "यह project बहुत important है, deadline कल है।",
        "please मुझे email कर दीजिए, details भेज दूँगा।",
    ],
    "hinglish_heavy": [
        "basically मैं software engineer हूँ और machine learning पर काम करता हूँ।",
        "actually यह algorithm बहुत complex है, but performance improve हो गई है।",
        "server down है, database connection timeout हो रहा है, restart करो।",
    ],
    "english_pure": [
        "Hello, how are you doing today?",
        "The weather is beautiful this morning.",
        "Please send me the report by end of day.",
    ],
    "technical_hinglish": [
        "इस function में input parameter pass करो और output check करो।",
        "data preprocessing के बाद model training start करो।",
        "API endpoint पर GET request भेजो और response parse करो।",
    ],
}


def load_model_from_checkpoint(
    checkpoint_dir: str,
    vocab_path: str,
    step: int = None,
    device: str = "cuda",
):
    """Load a fine-tuned model from a training checkpoint."""
    if step is not None:
        ckpt_file = f"model_{step}.pt"
    else:
        ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
        if "model_last.pt" in ckpt_files:
            ckpt_file = "model_last.pt"
        elif ckpt_files:
            ckpt_file = sorted(
                ckpt_files,
                key=lambda x: int("".join(filter(str.isdigit, x)) or "0"),
            )[-1]
        else:
            print(f"ERROR: No checkpoints found in {checkpoint_dir}")
            sys.exit(1)

    ckpt_path = os.path.join(checkpoint_dir, ckpt_file)
    print(f"Loading checkpoint: {ckpt_path}")

    vocab_char_map, vocab_size = get_tokenizer(vocab_path, tokenizer="custom")

    backbone = DiT(
        **INDICF5_MODEL_CFG,
        text_num_embeds=vocab_size,
        mel_dim=100,
    )

    model = CFM(
        transformer=backbone,
        mel_spec_kwargs=dict(
            n_fft=1024, hop_length=256, win_length=1024,
            n_mel_channels=100, target_sample_rate=SAMPLE_RATE,
            mel_spec_type="vocos",
        ),
        odeint_kwargs=dict(method="euler"),
        vocab_char_map=vocab_char_map,
    )

    checkpoint = torch.load(ckpt_path, weights_only=True, map_location="cpu")

    if "ema_model_state_dict" in checkpoint:
        state_dict = checkpoint["ema_model_state_dict"]
        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith("ema_model."):
                cleaned[k[len("ema_model."):]] = v
            elif k in ("initted", "step"):
                continue
            else:
                cleaned[k] = v
        for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
            cleaned.pop(key, None)
        model.load_state_dict(cleaned, strict=False)
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model = model.to(device)
    model.eval()

    step_num = checkpoint.get("step", "unknown")
    print(f"  Loaded checkpoint at step {step_num}")
    return model, step_num


def load_base_indicf5(vocab_path: str, device: str = "cuda"):
    """Load the original IndicF5 model for comparison."""
    from f5_tts.infer.utils_infer import load_model
    model = load_model(
        DiT, INDICF5_MODEL_CFG,
        mel_spec_type="vocos", vocab_file=vocab_path, device=device,
    )
    return model


def synthesize(model, vocoder, text, ref_audio, ref_text, device, nfe_step=16):
    """Generate audio for a single text. Returns float32 numpy array."""
    t0 = time.time()
    audio, sr, _ = infer_process(
        ref_audio, ref_text, text,
        model, vocoder,
        mel_spec_type="vocos",
        speed=1.0, device=device, nfe_step=nfe_step,
        show_info=lambda *a: None,
    )
    elapsed = time.time() - t0
    duration = len(audio) / SAMPLE_RATE
    rtf = elapsed / max(duration, 0.01)
    return np.array(audio, dtype=np.float32), duration, elapsed, rtf


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned IndicF5")
    parser.add_argument("--checkpoint-dir", required=True, help="Fine-tuned checkpoint directory")
    parser.add_argument("--vocab-path", default=None, help="Path to vocab.txt (auto-detected)")
    parser.add_argument("--ref-audio", default=None, help="Reference audio WAV for voice cloning")
    parser.add_argument("--ref-text", default=None, help="Transcript of reference audio")
    parser.add_argument("--output-dir", default="./eval_output", help="Output directory")
    parser.add_argument("--step", type=int, default=None, help="Specific checkpoint step")
    parser.add_argument("--compare-base", action="store_true", help="Also generate with base model")
    parser.add_argument("--nfe-step", type=int, default=16, help="NFE steps (16=fast, 32=quality)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Auto-detect vocab path
    vocab_path = args.vocab_path
    if vocab_path is None:
        for candidate in [
            os.path.join(os.path.dirname(args.checkpoint_dir), "data", "vocab.txt"),
            os.path.join(args.checkpoint_dir, "..", "data", "vocab.txt"),
            "./data/vocab.txt",
        ]:
            if os.path.exists(candidate):
                vocab_path = candidate
                break
        if vocab_path is None:
            hf_cache = os.path.expanduser("~/.cache/huggingface/hub/models--ai4bharat--IndicF5")
            for root, dirs, files in os.walk(hf_cache):
                if "vocab.txt" in files:
                    vocab_path = os.path.join(root, "vocab.txt")
                    break
    if vocab_path is None:
        print("ERROR: Could not find vocab.txt. Use --vocab-path to specify.")
        sys.exit(1)
    print(f"Vocab: {vocab_path}")

    # Auto-detect reference audio
    ref_audio_path = args.ref_audio
    ref_text = args.ref_text
    if ref_audio_path is None:
        voices_dir = "./voices"
        if os.path.exists(voices_dir):
            meta_path = os.path.join(voices_dir, "voices.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    voices = json.load(f)
                for vid, meta in voices.items():
                    if meta.get("transcript"):
                        ref_audio_path = os.path.join(voices_dir, meta["file"])
                        ref_text = meta["transcript"]
                        print(f"Using reference voice: {vid}")
                        break

    if ref_audio_path is None or ref_text is None:
        print("ERROR: No reference audio found. Use --ref-audio and --ref-text.")
        print("Or place voice files in ./voices/ with a voices.json metadata file.")
        sys.exit(1)

    print(f"Reference audio: {ref_audio_path}")
    ref_audio, ref_text_processed = preprocess_ref_audio_text(
        ref_audio_path, ref_text, device=device, show_info=lambda *a: None
    )

    print("Loading vocoder ...")
    vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device)

    print("\nLoading fine-tuned model ...")
    ft_model, ft_step = load_model_from_checkpoint(
        args.checkpoint_dir, vocab_path, step=args.step, device=device
    )

    base_model = None
    if args.compare_base:
        print("\nLoading base IndicF5 model ...")
        base_model = load_base_indicf5(vocab_path, device=device)

    print(f"\n{'='*70}")
    print(f"Generating test sentences (nfe_step={args.nfe_step})")
    print(f"{'='*70}\n")

    results = []
    for category, sentences in TEST_SENTENCES.items():
        cat_dir = os.path.join(output_dir, f"finetuned_step{ft_step}")
        os.makedirs(cat_dir, exist_ok=True)

        if base_model:
            base_dir = os.path.join(output_dir, "base_model")
            os.makedirs(base_dir, exist_ok=True)

        print(f"\n--- {category} ---")
        for i, text in enumerate(sentences):
            print(f"  [{i+1}] {text[:60]}{'...' if len(text) > 60 else ''}")

            audio_ft, dur_ft, elapsed_ft, rtf_ft = synthesize(
                ft_model, vocoder, text, ref_audio, ref_text_processed,
                device, nfe_step=args.nfe_step,
            )
            fname = f"{category}_{i+1}.wav"
            sf.write(os.path.join(cat_dir, fname), audio_ft, SAMPLE_RATE)
            print(f"       Fine-tuned: {dur_ft:.2f}s audio, {elapsed_ft:.2f}s gen, RTF={rtf_ft:.2f}x")

            result = {
                "category": category,
                "text": text,
                "finetuned_duration": round(dur_ft, 3),
                "finetuned_rtf": round(rtf_ft, 3),
            }

            if base_model:
                audio_base, dur_base, elapsed_base, rtf_base = synthesize(
                    base_model, vocoder, text, ref_audio, ref_text_processed,
                    device, nfe_step=args.nfe_step,
                )
                sf.write(os.path.join(base_dir, fname), audio_base, SAMPLE_RATE)
                print(f"       Base model: {dur_base:.2f}s audio, {elapsed_base:.2f}s gen, RTF={rtf_base:.2f}x")
                result["base_duration"] = round(dur_base, 3)
                result["base_rtf"] = round(rtf_base, 3)

            results.append(result)

    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")

    for category in TEST_SENTENCES:
        cat_results = [r for r in results if r["category"] == category]
        avg_rtf = sum(r["finetuned_rtf"] for r in cat_results) / len(cat_results)
        avg_dur = sum(r["finetuned_duration"] for r in cat_results) / len(cat_results)
        print(f"  {category:25s}: avg_dur={avg_dur:.2f}s, avg_rtf={avg_rtf:.2f}x")

    results_path = os.path.join(output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {results_path}")
    print(f"Audio saved to: {output_dir}")


if __name__ == "__main__":
    main()
