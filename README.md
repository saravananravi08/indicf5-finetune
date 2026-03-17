# IndicF5 Fine-Tuning for Hindi-English (Hinglish)

Fine-tune [IndicF5](https://huggingface.co/ai4bharat/IndicF5) TTS model to support English and Hindi-English code-switched (Hinglish) speech using the [OpenSLR-104](https://www.openslr.org/104/) dataset.

## Prerequisites

- Python 3.10 or 3.11
- NVIDIA GPU with CUDA (L40S 48GB recommended, minimum 24GB VRAM)
- [uv](https://docs.astral.sh/uv/) package manager
- `wget` (for dataset download)
- `ffmpeg` and `libsndfile1` (for audio processing)

## Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the project
cd ~/Documents/indicf5-finetune

# Create virtual environment and install dependencies
uv sync

# Optional: install with WandB for experiment tracking
uv sync --extra wandb
```

### System dependencies (Ubuntu/Debian)

```bash
sudo apt-get install -y wget ffmpeg libsndfile1
```

## Usage

### Step 1: Prepare Data

Downloads OpenSLR-104 (~7.3GB), resamples audio to 24kHz, creates F5-TTS arrow format.

```bash
uv run prepare-data --data-dir ./data
```

This takes ~30-60 minutes depending on your internet speed and CPU. Output:

```
data/
├── audio/          # 24kHz mono WAV files
├── raw/            # HuggingFace arrow dataset
├── duration.json   # Duration metadata
└── vocab.txt       # IndicF5 vocabulary (Indic + Latin characters)
```

### Step 2: Fine-Tune

```bash
# Full fine-tune (recommended for best Hinglish quality)
uv run train --data-dir ./data

# Adapter-style (faster, lower VRAM, preserves Indic quality)
uv run train --data-dir ./data --freeze-backbone

# Without WandB
uv run train --data-dir ./data --no-wandb

# Resume from checkpoint
uv run train --data-dir ./data --resume
```

#### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 50 | Training epochs |
| `--lr` | 1e-5 | Learning rate (auto 5e-5 for adapter) |
| `--batch-size` | 38400 | Batch size in frames (~10s audio) |
| `--grad-accum` | 2 | Gradient accumulation steps |
| `--warmup-steps` | 1000 | LR warmup steps |
| `--save-every` | 2000 | Checkpoint save interval |
| `--freeze-backbone` | off | Adapter-style training |

#### Estimated Training Time

| GPU | Full Fine-Tune | Adapter |
|-----|---------------|---------|
| L40S 48GB | ~8-12 hours | ~4-6 hours |
| A100 80GB | ~6-8 hours | ~3-4 hours |
| RTX 4090 24GB | ~15-20 hours | ~8-10 hours |

### Step 3: Evaluate

Generate test audio across Hindi, Hinglish, and English sentences:

```bash
# Evaluate latest checkpoint
uv run evaluate --checkpoint-dir ./checkpoints

# Compare with base IndicF5 model
uv run evaluate --checkpoint-dir ./checkpoints --compare-base

# Evaluate a specific checkpoint step
uv run evaluate --checkpoint-dir ./checkpoints --step 10000

# Use custom reference voice
uv run evaluate --checkpoint-dir ./checkpoints \
    --ref-audio ./voices/TAM_F_HAPPY_00001.wav \
    --ref-text "your reference transcript here"
```

### Multi-GPU Training

```bash
uv run accelerate launch -m indicf5_finetune.train --data-dir ./data
```

## Project Structure

```
indicf5-finetune/
├── pyproject.toml                          # uv/pip project config
├── README.md
├── .gitignore
├── voices/                                 # Reference audio for evaluation
│   ├── voices.json
│   └── *.wav
└── src/
    ├── f5_tts/                             # Bundled F5-TTS package (AI4Bharat fork)
    │   ├── model/                          # CFM, DiT, Trainer, Dataset
    │   └── infer/                          # Inference utilities
    └── indicf5_finetune/                   # Fine-tuning scripts
        ├── prepare_data.py                 # Data download + preprocessing
        ├── train.py                        # Training script
        └── evaluate.py                     # Evaluation script
```

## Dataset

**OpenSLR-104** — Hindi-English code-switched speech corpus:
- ~90 hours of conversational Hindi-English speech
- Dense code-switching (Hinglish words within Hindi sentences)
- License: CC-BY-SA 4.0 (open-source your fine-tuned model)

## Model Architecture

IndicF5 uses F5-TTS with a DiT (Diffusion Transformer) backbone:
- Parameters: ~400M
- DiT: dim=1024, depth=22, heads=16, text_dim=512
- Mel: 100 channels, 24kHz sample rate, vocos vocoder
- Training: Conditional Flow Matching (CFM) with random span masking

## License

- This fine-tuning code: MIT
- IndicF5 model: MIT
- F5-TTS code: MIT
- OpenSLR-104 dataset: CC-BY-SA 4.0
- **Fine-tuned model weights must be open-sourced** (CC-BY-SA ShareAlike requirement)
