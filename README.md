# Open-RLHF-RAG

# TIRESRAG-R1 A100 SXM Installation Guide

## System Requirements
- NVIDIA A100 SXM (40GB or 80GB)
- CUDA 11.8+ or 12.1+
- Python 3.8-3.11
- 32GB+ System RAM recommended

## Step 1: Environment Setup

### Option A: Fresh Conda Environment (Recommended)
```bash
# Create new environment
conda create -n tiresrag python=3.10 -y
conda activate tiresrag

# Install CUDA toolkit (if not system-wide)
conda install nvidia/label/cuda-12.1.0::cuda-toolkit -y
```

### Option B: Virtual Environment
```bash
python3.10 -m venv tiresrag_env
source tiresrag_env/bin/activate  # Linux/Mac
# or
tiresrag_env\Scripts\activate     # Windows
```

## Step 2: Core Dependencies

### PyTorch with CUDA Support
```bash
# For CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1 (recommended for A100)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

### Transformers and Core ML Libraries
```bash
# Core transformers stack
pip install transformers==4.35.2
pip install accelerate==0.24.1
pip install datasets==2.14.6
pip install tokenizers==0.15.0

# Flash Attention 2 for A100 optimization
pip install flash-attn==2.3.3 --no-build-isolation

# Alternative if flash-attn fails
pip install flash-attn --no-build-isolation
```

### HuggingFace Hub and Authentication
```bash
pip install huggingface_hub==0.19.4
pip install safetensors==0.4.0

# Login to HuggingFace (for private models if needed)
huggingface-cli login
```

## Step 3: Evaluation Dependencies

### Text Processing and Metrics
```bash
# ROUGE scoring
pip install rouge-score==0.1.2

# BERT Score (optional but recommended)
pip install bert-score==0.3.13

# Text preprocessing
pip install nltk==3.8.1
python -c "import nltk; nltk.download('punkt')"
```

### Scientific Computing Stack
```bash
# NumPy/SciPy (optimized versions)
pip install numpy==1.24.4
pip install scipy==1.11.4

# Data analysis and visualization
pip install pandas==2.1.3
pip install matplotlib==3.8.2
pip install seaborn==0.13.0
```

### HTTP and API Libraries
```bash
# For retrieval services
pip install requests==2.31.0
pip install aiohttp==3.9.1
```

## Step 4: Memory Optimization Libraries

### For Large Model Loading
```bash
# Memory efficient loading
pip install bitsandbytes==0.41.3

# Model sharding and parallelism  
pip install deepspeed==0.12.3

# Alternative: FairScale for model parallelism
pip install fairscale==0.4.13
```

## Step 5: Optional Advanced Dependencies

### For FlashRAG Integration
```bash
# FAISS for vector similarity search
conda install -c conda-forge faiss-gpu -y
# or
pip install faiss-gpu==1.7.4

# Sentence transformers for embeddings
pip install sentence-transformers==2.2.2
```

### For Enhanced Evaluation
```bash
# BERTScore dependencies
pip install packaging==23.2

# Additional NLP metrics
pip install sacrebleu==2.3.1
pip install evaluate==0.4.1
```

## Step 6: System-Level Optimizations

### CUDA Environment Variables
```bash
# Add to ~/.bashrc or ~/.zshrc
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# A100-specific optimizations
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="8.0"  # A100 architecture
```

### Memory Management
```bash
# For handling large models
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_VISIBLE_DEVICES=0  # Use first GPU only
```

## Step 7: Verification Script

Create `verify_installation.py`:
```python
#!/usr/bin/env python3

import sys
import torch
import transformers
import datasets
import numpy as np
import pandas as pd

def check_installation():
    print("🔍 Checking TIRESRAG-R1 Installation...")
    
    # CUDA Check
    print(f"✓ CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ GPU Count: {torch.cuda.device_count()}")
        print(f"✓ GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA Version: {torch.version.cuda}")
    
    # Memory Check
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        print(f"✓ GPU Memory: {gpu_memory / 1e9:.1f} GB")
    
    # Package Versions
    packages = {
        'torch': torch.__version__,
        'transformers': transformers.__version__,
        'datasets': datasets.__version__,
        'numpy': np.__version__,
        'pandas': pd.__version__
    }
    
    print("\n📦 Package Versions:")
    for pkg, ver in packages.items():
        print(f"  {pkg}: {ver}")
    
    # Optional packages
    optional = ['rouge_score', 'bert_score', 'flash_attn']
    print("\n🔧 Optional Packages:")
    for pkg in optional:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}: Available")
        except ImportError:
            print(f"  ✗ {pkg}: Not installed")
    
    # Test model loading
    print("\n🤖 Testing Model Loading...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        print("✓ Model loading test passed")
    except Exception as e:
        print(f"✗ Model loading test failed: {e}")
    
    print("\n🎉 Installation verification complete!")

if __name__ == "__main__":
    check_installation()
```

Run verification:
```bash
python verify_installation.py
```

## Step 8: A100-Specific Configuration

### Model Loading Configuration
Add to your evaluation script:
```python
# A100-optimized model loading
model_kwargs = {
    "torch_dtype": torch.bfloat16,  # A100 supports bfloat16 natively
    "device_map": "auto",
    "trust_remote_code": True,
    "attn_implementation": "flash_attention_2",  # Use Flash Attention 2
    "low_cpu_mem_usage": True,
    "use_cache": True
}
```

### Batch Processing Optimization
```python
# A100 can handle larger batches
BATCH_SIZE = 8  # Adjust based on model size
MAX_LENGTH = 2048
GENERATION_KWARGS = {
    "max_new_tokens": 512,
    "temperature": 0.1,
    "top_p": 0.9,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "use_cache": True
}
```

## Troubleshooting

### Common Issues and Solutions

#### Flash Attention Installation Fails
```bash
# Alternative installation
pip install flash-attn --no-build-isolation --no-deps
pip install einops  # Required dependency
```

#### CUDA Out of Memory
```bash
# Reduce batch size or use gradient checkpointing
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
```

#### Dataset Loading Issues
```bash
# Increase timeout for large datasets
export HF_DATASETS_OFFLINE=0
export HF_HUB_TIMEOUT=300
```

## Expected Installation Size
- Total disk space: ~15-20 GB
- Model cache (varies by models used): 10-50 GB
- Dataset cache: 5-15 GB

## Performance Expectations on A100
- **Model Loading**: 30-60 seconds (depending on model size)
- **Inference Speed**: 50-200 tokens/second per sample
- **Memory Usage**: 15-35 GB VRAM (depending on model)
- **Full Evaluation (600 samples)**: 2-6 hours

## Final Installation Command
```bash
# Complete installation in one go
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121 && \
pip install transformers==4.35.2 accelerate==0.24.1 datasets==2.14.6 && \
pip install flash-attn==2.3.3 --no-build-isolation && \
pip install rouge-score==0.1.2 bert-score==0.3.13 && \
pip install numpy==1.24.4 scipy==1.11.4 pandas==2.1.3 matplotlib==3.8.2 seaborn==0.13.0 && \
pip install requests==2.31.0 bitsandbytes==0.41.3 && \
echo "✅ TIRESRAG-R1 installation complete!"
```
