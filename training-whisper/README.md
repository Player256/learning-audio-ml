# Mini-Whisper Training on SageMaker

Hindi → English speech translation model trained on Bhasaanuvaad/SeamlessAlign dataset.

## Architecture

- **Encoder**: Conv subsampling + Transformer ((8,12,12) layers, d_model=(512, 768, 768))
- **Decoder**: Transformer with cross-attention ((6, 8, 12) layers, d_model=(512, 768, 768))

## Features

✅ Mixed precision training (FP16)
✅ Streaming dataset (no download needed)
✅ Train/Val/Test splits (820k/100k/100k)
✅ WER/CER metrics
✅ W&B logging
✅ Gradient clipping
✅ Warmup + cosine scheduler

## Setup

1. **Install dependencies locally** (for launch script):
   ```bash
   pip install sagemaker boto3
   ```

2. **Set W&B API key** (optional but recommended):
   ```bash
   export WANDB_API_KEY=your_wandb_key_here
   ```
   Get your key from: https://wandb.ai/authorize
   
   If not set, training will still work but metrics won't be logged to W&B.

3. **Update launch.py** with your AWS role ARN (if not running on SageMaker notebook)

4. **Launch training**:
   ```bash
   python launch.py
   ```

   This launches 3 experiments in parallel:
   - `mini-whisper-small` (512 dim, 8+6 layers)
   - `mini-whisper-medium` (768 dim, 12+8 layers)
   - `mini-whisper-large` (768 dim, 12+12 layers)

## Files

```
training-whisper/
├── mini-whisper-train.py   # Training script (runs on SageMaker)
├── launch.py               # Launcher (runs locally)
├── requirements.txt        # Python deps
└── README.md              # This file
```

## Monitoring

### Quick Status Check
```bash
python check_jobs.py --job-name=<job-name>
```

## Output

Models saved to: `s3://home-ml/experiments/{experiment_name}/`

