"""
Fine-tune IndicF5 on OpenSLR-104 Hindi-English code-switched data.

Loads the pre-trained IndicF5 checkpoint (DiT 1024d/22L/16H),
optionally freezes layers, and trains using F5-TTS's Trainer with
dynamic frame-based batching.

Usage:
    # Full fine-tune (all layers trainable)
    uv run train --data-dir ./data

    # Adapter-style: freeze backbone, only train embeddings + output
    uv run train --data-dir ./data --freeze-backbone

    # Resume from checkpoint
    uv run train --data-dir ./data --resume

    # Multi-GPU with accelerate
    uv run accelerate launch -m indicf5_finetune.train --data-dir ./data
"""

import argparse
import os
import sys

import torch

from f5_tts.model import CFM, DiT
from f5_tts.model.trainer import Trainer
from f5_tts.model.dataset import load_dataset
from f5_tts.model.utils import get_tokenizer


# ---------------------------------------------------------------------------
# IndicF5 architecture config (must match pre-trained checkpoint)
# ---------------------------------------------------------------------------

INDICF5_MODEL_CFG = dict(
    dim=1024,
    depth=22,
    heads=16,
    ff_mult=2,
    text_dim=512,
    conv_layers=4,
)

MEL_SPEC_KWARGS = dict(
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    n_mel_channels=100,
    target_sample_rate=24_000,
    mel_spec_type="vocos",
)


def find_indicf5_checkpoint():
    """Find the IndicF5 model weights in HuggingFace cache."""
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub/models--ai4bharat--IndicF5")
    for root, dirs, files in os.walk(hf_cache):
        if "model.safetensors" in files:
            return os.path.join(root, "model.safetensors")
    print("IndicF5 checkpoint not found in cache. Downloading...")
    from huggingface_hub import hf_hub_download
    return hf_hub_download("ai4bharat/IndicF5", filename="model.safetensors")


def load_pretrained_weights(model: CFM, checkpoint_path: str, device: str = "cpu"):
    """Load pre-trained IndicF5 weights into the CFM model.

    The HF checkpoint contains the full INF5Model state dict.
    We extract just the ema_model (CFM) weights.
    """
    from safetensors.torch import load_file

    print(f"Loading pre-trained weights from {checkpoint_path} ...")
    state_dict = load_file(checkpoint_path, device=device)

    # Strip known prefixes from keys: ema_model., _orig_mod., vocoder.
    cleaned = {}
    for k, v in state_dict.items():
        key = k
        # Strip all known prefixes (can be nested like ema_model._orig_mod.)
        for prefix in ["ema_model.", "_orig_mod."]:
            while key.startswith(prefix):
                key = key[len(prefix):]
        if key.startswith("vocoder."):
            continue
        cleaned[key] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
        for k in missing[:5]:
            print(f"    {k}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
        for k in unexpected[:5]:
            print(f"    {k}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {total_params/1e6:.1f}M params total")

    return model


def freeze_backbone(model: CFM):
    """Freeze transformer backbone, keep text embeddings and output layers trainable.

    Adapter-style fine-tuning:
    - Frozen: transformer attention layers, FFN layers
    - Trainable: text embeddings, input/output projections, norms, mel_spec
    """
    frozen_count = 0
    trainable_count = 0

    trainable_patterns = [
        "text_embed",
        "proj_in",
        "proj_out",
        "mel_spec",
        "norm",
        "to_pred",
    ]

    for name, param in model.named_parameters():
        should_train = any(p in name for p in trainable_patterns)
        if not should_train:
            param.requires_grad = False
            frozen_count += param.numel()
        else:
            trainable_count += param.numel()

    total = frozen_count + trainable_count
    print(f"  Frozen params:    {frozen_count/1e6:.1f}M ({frozen_count/total*100:.1f}%)")
    print(f"  Trainable params: {trainable_count/1e6:.1f}M ({trainable_count/total*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune IndicF5 on Hindi-English data")
    parser.add_argument("--data-dir", required=True, help="Path to prepared dataset")
    parser.add_argument("--checkpoint-dir", default="./checkpoints",
                        help="Where to save training checkpoints")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze transformer backbone (adapter-style training)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate (1e-5 for full fine-tune, 5e-5 for adapter)")
    parser.add_argument("--batch-size", type=int, default=38400,
                        help="Batch size in frames (38400 ~ 10s of audio per batch)")
    parser.add_argument("--max-samples", type=int, default=16,
                        help="Max samples per batch")
    parser.add_argument("--grad-accum", type=int, default=2,
                        help="Gradient accumulation steps")
    parser.add_argument("--warmup-steps", type=int, default=1000,
                        help="Warmup steps (lower than default 20K for fine-tuning)")
    parser.add_argument("--save-every", type=int, default=2000,
                        help="Save checkpoint every N updates")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--wandb-project", default="indicf5-hinglish",
                        help="WandB project name")
    parser.add_argument("--wandb-run", default="hinglish-finetune",
                        help="WandB run name")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable WandB logging")
    parser.add_argument("--log-samples", action="store_true",
                        help="Log audio samples during training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--test-run", action="store_true",
                        help="Quick smoke test: 1 epoch, save at step 2, small batch. Works on CPU.")
    args = parser.parse_args()

    # Override settings for test run
    if args.test_run:
        args.epochs = 1
        args.save_every = 2
        args.warmup_steps = 2
        args.no_wandb = True
        args.batch_size = 9600  # ~2.5s of audio per batch
        args.max_samples = 4
        args.grad_accum = 1
        args.num_workers = 1
        print("\n*** TEST RUN MODE — 1 epoch, small batch, no wandb ***\n")

    data_dir = os.path.abspath(args.data_dir)
    ckpt_dir = os.path.abspath(args.checkpoint_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"IndicF5 Fine-Tuning")
    print(f"{'='*60}")
    print(f"  Device:      {device}")
    if device == "cuda":
        print(f"  GPU:         {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:        {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    print(f"  Data dir:    {data_dir}")
    print(f"  Checkpoint:  {ckpt_dir}")
    print(f"  Freeze:      {args.freeze_backbone}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  LR:          {args.lr}")
    print(f"  Batch (frames): {args.batch_size}")
    print(f"  Grad accum:  {args.grad_accum}")
    print(f"{'='*60}\n")

    # 1. Load vocabulary
    vocab_path = os.path.join(data_dir, "vocab.txt")
    if not os.path.exists(vocab_path):
        print(f"ERROR: vocab.txt not found at {vocab_path}")
        print("Run prepare-data first!")
        sys.exit(1)

    vocab_char_map, vocab_size = get_tokenizer(vocab_path, tokenizer="custom")
    print(f"Vocabulary: {vocab_size} tokens")

    # 2. Build model (same architecture as IndicF5)
    print("Building model ...")
    backbone = DiT(
        **INDICF5_MODEL_CFG,
        text_num_embeds=vocab_size,
        mel_dim=100,
    )

    model = CFM(
        transformer=backbone,
        mel_spec_kwargs=dict(
            n_fft=MEL_SPEC_KWARGS["n_fft"],
            hop_length=MEL_SPEC_KWARGS["hop_length"],
            win_length=MEL_SPEC_KWARGS["win_length"],
            n_mel_channels=MEL_SPEC_KWARGS["n_mel_channels"],
            target_sample_rate=MEL_SPEC_KWARGS["target_sample_rate"],
            mel_spec_type=MEL_SPEC_KWARGS["mel_spec_type"],
        ),
        odeint_kwargs=dict(method="euler"),
        vocab_char_map=vocab_char_map,
    )

    # 3. Load pre-trained IndicF5 weights
    if not args.resume:
        ckpt_path = find_indicf5_checkpoint()
        model = load_pretrained_weights(model, ckpt_path, device="cpu")

    # 4. Optionally freeze backbone
    if args.freeze_backbone:
        print("Freezing transformer backbone ...")
        freeze_backbone(model)
        if args.lr == 1e-5:
            args.lr = 5e-5
            print(f"  Auto-adjusted LR to {args.lr} for adapter training")

    # 5. Load dataset
    print(f"Loading dataset from {data_dir} ...")
    train_dataset = load_dataset(
        dataset_name="openslr104",
        dataset_type="CustomDatasetPath",
        audio_type="raw",
        data_dir=data_dir,
        mel_spec_kwargs=dict(
            n_mel_channels=MEL_SPEC_KWARGS["n_mel_channels"],
            target_sample_rate=MEL_SPEC_KWARGS["target_sample_rate"],
            hop_length=MEL_SPEC_KWARGS["hop_length"],
            n_fft=MEL_SPEC_KWARGS["n_fft"],
            win_length=MEL_SPEC_KWARGS["win_length"],
            mel_spec_type=MEL_SPEC_KWARGS["mel_spec_type"],
        ),
    )
    print(f"Dataset: {len(train_dataset)} utterances")

    # 6. Create trainer and start training
    print("Creating trainer ...")
    trainer = Trainer(
        model=model,
        epochs=args.epochs,
        learning_rate=args.lr,
        num_warmup_updates=args.warmup_steps,
        save_per_updates=args.save_every,
        checkpoint_path=ckpt_dir,
        batch_size=args.batch_size,
        batch_size_type="frame",
        max_samples=args.max_samples,
        grad_accumulation_steps=args.grad_accum,
        max_grad_norm=1.0,
        logger=None if args.no_wandb else "wandb",
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run,
        log_samples=args.log_samples,
        last_per_steps=args.save_every,
        mel_spec_type="vocos",
        ema_kwargs=dict(
            beta=0.9999,
            update_after_step=100,
            update_every=10,
        ),
    )

    print("\nStarting training ...\n")
    trainer.train(
        train_dataset,
        num_workers=args.num_workers,
        resumable_with_seed=args.seed,
    )

    print("\nTraining complete!")
    print(f"Checkpoints saved to: {ckpt_dir}")


if __name__ == "__main__":
    main()
