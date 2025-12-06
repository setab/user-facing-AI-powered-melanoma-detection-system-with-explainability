#!/bin/bash
# Start Gradio server with correct Python environment

# Get project root (parent of scripts directory)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Starting Melanoma Detection Gradio Server"
echo "=========================================="
echo ""
echo "Check your .env file for server configuration"
echo "Default port: 7860"
echo ""

# Use the correct dPython environment
/home/the/miniconda/envs/ml2/bin/python src/serve_gradio.py
