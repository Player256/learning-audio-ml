"""
Entry point for SageMaker training that ensures PyTorch is available before importing transformers.
"""
import sys
import os

# Ensure torch is imported first
import torch
import torchaudio

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")

# Now that torch is loaded, we can import the actual training script
from train import main

if __name__ == "__main__":
    main()
