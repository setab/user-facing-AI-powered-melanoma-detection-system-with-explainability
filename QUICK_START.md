# 🚀 Quick Start Guide

## Current Status

✅ Server configured (check .env file)  
✅ Port: **7860**  
✅ Authentication: Configured in .env  
❌ Model weights: **MISSING** - Need to train first!

## Immediate Steps to Run

### Option 1: Train Model First (Recommended)

You need to train a model before using the web UI:

```bash
# Open the training notebook
jupyter notebook notebooks/01_train_baseline.ipynb

# Or use Jupyter Lab
jupyter lab notebooks/01_train_baseline.ipynb
```

**In the notebook:**
1. Run all cells sequentially (Shift+Enter)
2. This will create: `models/checkpoints/melanoma_resnet50_nb.pth`
3. Training takes 30-60 minutes depending on GPU

### Option 2: Quick Demo Mode (No Model)

For testing the interface without a model:

```bash
# Create a dummy model checkpoint for testing
python -c "
import torch
import torchvision.models as models

# Create a ResNet50 with 7 classes (HAM10000)
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 7)

# Save dummy checkpoint
torch.save({
    'state_dict': model.state_dict(),
    'architecture': 'resnet50',
}, 'models/checkpoints/melanoma_resnet50_nb.pth')

print('✅ Dummy model created for testing!')
"
```

### Option 3: Use CLI Instead (If You Have Images)

If you just want to test inference without web UI:

```bash
# Test with a sample image (replace with your image path)
python -m src.inference.cli \
  --image data/ds/img/ISIC_0024306.jpg \
  --weights models/checkpoints/melanoma_resnet50_nb.pth \
  --label-map models/label_maps/label_map_nb.json \
  --out test_result.jpg
```

## Access from Mac

Once model is ready and server is running:

### Step 1: Start Server (Ubuntu)

```bash
cd /home/the/Codes/Melanoma-detection
bash scripts/start_server.sh
# or directly:
python src/serve_gradio.py
```

You should see:
```
🔒 Authentication enabled for user: the
🚀 Starting Gradio on 0.0.0.0:7860
Running on local URL:  http://0.0.0.0:7860
```

### Step 2: Open on Mac

**In Mac browser, navigate to:**
```
http://SERVER_IP_HIDDEN:7860
```

**Login:**
- Use credentials from your .env file
- GRADIO_USERNAME and GRADIO_PASSWORD

### Step 3: Check Firewall (If Connection Fails)

```bash
# On Ubuntu server, check firewall status
sudo ufw status

# If port 7860 is not allowed, add it:
sudo ufw allow 7860/tcp
sudo ufw reload

# Verify port is listening
netstat -tlnp | grep 7860
```

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'src'"
**Fixed!** ✅ The import path has been corrected.

### Error: "FileNotFoundError: melanoma_resnet50_nb.pth"
**Solution:** Train model using `learning/day1.ipynb` or create dummy model (see Option 2 above)

### Error: "Cannot connect from Mac"
**Check:**
1. Server is running: `ps aux | grep serve_gradio`
2. Port is open: `sudo ufw allow 7860/tcp`
3. Using correct IP: `SERVER_IP_HIDDEN:7860`
4. Both devices on same network

### Error: "CUDA out of memory"
**Solution:** Use CPU mode or reduce batch size in training

## Quick Commands Reference

```bash
# Start server (after model is trained)
bash scripts/start_server.sh

# Or directly
python src/serve_gradio.py

# Run in background with tmux
tmux new -s gradio
bash scripts/start_server.sh
# Detach: Ctrl+B, then D
# Re-attach: tmux attach -t gradio

# Check server IP
hostname -I | awk '{print $1}'

# Check port is open
netstat -tlnp | grep 7860

# Stop server
# Press Ctrl+C in terminal, or:
pkill -f serve_gradio
```

## What to Do Next

1. **For Quick Test:**
   - Create dummy model (Option 2)
   - Start server
   - Access from Mac at `http://SERVER_IP_HIDDEN:7860`
   - Upload any image to test interface

2. **For Real Project:**
   - Train model using `notebooks/01_train_baseline.ipynb`
   - Verify model saved: `ls -lh models/checkpoints/*.pth`
   - Start server: `bash scripts/start_server.sh`
   - Upload skin lesion images
   - Test chat Q&A with uncertain cases

3. **For Thesis:**
   - Run model comparison: `python src/training/compare_models.py`
   - Generate visualizations for thesis writeup
   - Document results in thesis

## Files Status

```
✅ .env - Configured for remote access
✅ Configuration - Server: 0.0.0.0:7860
✅ Authentication - Username/password set
✅ Code - Import issues fixed
❌ Model weights - Need to train or download
✅ Network - IP: SERVER_IP_HIDDEN
```

---

**Mac URL:** http://YOUR_SERVER_IP:7860  
**Login:** See .env file for credentials  
**Status:** Ready after model training ✅
