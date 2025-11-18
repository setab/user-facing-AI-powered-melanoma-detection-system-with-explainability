#!/bin/bash
# Quick setup script for running model comparison experiments

echo "🚀 Melanoma Detection - Model Comparison Setup"
echo "==============================================="
echo ""

# Check Python environment
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python found: $(python --version)"
echo ""

# Check for required packages
echo "📦 Checking dependencies..."
MISSING_PACKAGES=()

python -c "import torch" 2>/dev/null || MISSING_PACKAGES+=("torch")
python -c "import torchvision" 2>/dev/null || MISSING_PACKAGES+=("torchvision")
python -c "import sklearn" 2>/dev/null || MISSING_PACKAGES+=("scikit-learn")
python -c "import pandas" 2>/dev/null || MISSING_PACKAGES+=("pandas")
python -c "import matplotlib" 2>/dev/null || MISSING_PACKAGES+=("matplotlib")
python -c "import seaborn" 2>/dev/null || MISSING_PACKAGES+=("seaborn")
python -c "import gradio" 2>/dev/null || MISSING_PACKAGES+=("gradio")

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo ""
    echo "⚠️  Missing packages: ${MISSING_PACKAGES[*]}"
    echo ""
    echo "Install with:"
    echo "  pip install ${MISSING_PACKAGES[*]}"
    echo ""
    echo "Or install all requirements:"
    echo "  pip install -r requirements-train.txt"
    exit 1
else
    echo "✅ All required packages found"
fi

echo ""
echo "📁 Checking data..."

# Check for metadata
if [ ! -f "data/HAM10000_metadata.csv" ]; then
    echo "❌ data/HAM10000_metadata.csv not found"
    echo "   Please place your HAM10000 metadata CSV in data/"
    exit 1
fi

# Check for images
if [ ! -d "data/ds/img" ] || [ -z "$(ls -A data/ds/img)" ]; then
    echo "❌ data/ds/img/ is empty or doesn't exist"
    echo "   Please place HAM10000 images in data/ds/img/"
    exit 1
fi

IMAGE_COUNT=$(ls -1 data/ds/img/*.jpg 2>/dev/null | wc -l)
echo "✅ Found $IMAGE_COUNT images in data/ds/img/"

echo ""
echo "📊 Checking model artifacts..."

if [ -f "models/label_maps/label_map_nb.json" ]; then
    echo "✅ Label map found"
else
    echo "⚠️  Label map not found (will be created during training)"
fi

echo ""
echo "🔧 Configuration check..."

if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found"
    echo "   Copying .env.example to .env..."
    cp .env.example .env
    echo "   ✅ Created .env - please edit with your settings"
else
    echo "✅ .env file exists"
fi

echo ""
echo "==============================================="
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo ""
echo "1. Review configuration:"
echo "   nano .env"
echo ""
echo "2. Run model comparison (this will take several hours):"
echo "   python src/training/compare_models.py \\"
echo "     --output-dir experiments/model_comparison \\"
echo "     --epochs 20 --batch-size 32"
echo ""
echo "3. Generate visualizations for thesis:"
echo "   python src/training/visualize_comparison.py \\"
echo "     --results experiments/model_comparison/comparison_results.json"
echo ""
echo "4. Test Gradio web UI with chat Q&A:"
echo "   python src/serve_gradio.py"
echo ""
echo "For more details, see:"
echo "  - docs/MODEL_COMPARISON_GUIDE.md"
echo "  - docs/RECENT_UPDATES.md"
echo ""
