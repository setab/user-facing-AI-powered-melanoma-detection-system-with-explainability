#!/bin/bash
# Start Gradio server with correct Python environment

# Get project root (parent of scripts directory)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🚀 Starting Melanoma Detection Gradio Server"
echo "=============================================="
echo ""
echo "Server IP: SERVER_IP_HIDDEN"
echo "Port: 7860"
echo "Access from Mac: http://SERVER_IP_HIDDEN:7860"
echo "Starting server..."
echo ""

# Use the correct Python environment
/home/the/miniconda/envs/ml2/bin/python src/serve_gradio.py
