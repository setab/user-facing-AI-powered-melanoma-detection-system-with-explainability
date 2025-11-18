#!/bin/bash
# Start Gradio server with correct Python environment

# Get project root (parent of scripts directory)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🚀 Starting Melanoma Detection Gradio Server"
echo "=============================================="
echo ""
echo "Server IP: 192.168.0.207"
echo "Port: 7860"
echo "Access from Mac: http://192.168.0.207:7860"
echo ""
echo "Login credentials:"
echo "  Username: the"
echo "  Password: Iamsetab0071"
echo ""
echo "Starting server..."
echo ""

# Use the correct Python environment
/home/the/miniconda/envs/ml2/bin/python src/serve_gradio.py
