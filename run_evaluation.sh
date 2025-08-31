#!/bin/bash

# Create directory structure
mkdir -p logs
mkdir -p results

# Log start time
echo "Starting OpenRLHF-RAG evaluation at $(date)"

# Run the comprehensive evaluation script
python run_comprehensive_eval.py \
  --model_name meta-llama/Meta-Llama-3-8B \
  --max_samples 200 \
  --output_dir ./results

# Log completion
echo "Completed OpenRLHF-RAG evaluation at $(date)"
