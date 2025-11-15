# Server Deployment Guide: Ubuntu GPU Server + Mac Client

This guide covers deploying the melanoma detection system on your home Ubuntu server (RTX 5060 Ti 16GB) and accessing the Gradio UI from your Mac.

## Architecture

```
┌──────────────────────┐         Network         ┌──────────────────────┐
│   Ubuntu Server      │◄────────────────────────►│   Mac Client         │
│   (GPU: RTX 5060Ti)  │   SSH / HTTP/HTTPS      │   (Web Browser)      │
│                      │                          │                      │
│  • Python env        │                          │  • Access Gradio UI  │
│  • Model weights     │                          │  • Upload images     │
│  • Gradio server     │                          │  • View results      │
└──────────────────────┘                          └──────────────────────┘
```

## Prerequisites

### On Ubuntu Server
- Ubuntu 20.04+ with NVIDIA drivers installed
- CUDA-compatible PyTorch (already installed in `/home/the/miniconda/envs/ml2`)
- RTX 5060 Ti 16GB properly configured
- Network access (local or VPN/tailscale if remote)

### On Mac
- Modern web browser (Safari, Chrome, Firefox)
- Network access to the Ubuntu server
- Optional: SSH client for management

## Setup Steps

### 1. Server Configuration

#### a) Copy environment template and configure
```bash
cd /home/the/Codes/Melanoma-detection
cp .env.example .env
nano .env
```

Edit `.env` with your settings:
```bash
# Server binding (0.0.0.0 allows remote access)
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
GRADIO_SHARE=False

# Security: SET THESE FOR AUTHENTICATION
GRADIO_USERNAME=your_username
GRADIO_PASSWORD=your_secure_password

# Model paths (defaults should work if unchanged)
WEIGHTS_PATH=models/checkpoints/melanoma_resnet50_nb.pth
LABEL_MAP_PATH=models/label_maps/label_map_nb.json
TEMPERATURE_JSON_PATH=models/checkpoints/temperature.json
OPERATING_JSON_PATH=models/checkpoints/operating_points.json

LOG_LEVEL=INFO
```

#### b) Install python-dotenv for .env support
```bash
/home/the/miniconda/envs/ml2/bin/pip install python-dotenv
```

#### c) Verify GPU access
```bash
/home/the/miniconda/envs/ml2/bin/python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 5060 Ti
```

### 2. Firewall Configuration

#### Option A: Local Network Only (Recommended for home)
Allow port 7860 from your local subnet:
```bash
sudo ufw allow from 192.168.1.0/24 to any port 7860 comment 'Gradio melanoma detection'
sudo ufw status
```

#### Option B: Public Access (Use with authentication!)
```bash
sudo ufw allow 7860/tcp comment 'Gradio melanoma detection'
```

⚠️ **Security Warning**: If exposing publicly, ALWAYS set `GRADIO_USERNAME` and `GRADIO_PASSWORD` in `.env`!

### 3. Launch the Server

#### Using tmux (recommended for persistent sessions)
```bash
# Start a new tmux session
tmux new -s melanoma

# Activate conda env and launch
cd /home/the/Codes/Melanoma-detection
/home/the/miniconda/envs/ml2/bin/python src/serve_gradio.py

# Detach: Ctrl+B then D
# Reattach: tmux attach -t melanoma
```

#### Direct launch (for testing)
```bash
cd /home/the/Codes/Melanoma-detection
/home/the/miniconda/envs/ml2/bin/python src/serve_gradio.py
```

You should see:
```
🔒 Authentication enabled for user: your_username
🚀 Starting Gradio on 0.0.0.0:7860
Running on local URL:  http://0.0.0.0:7860
```

### 4. Access from Mac

#### Find your server's IP
On Ubuntu server:
```bash
ip addr show | grep "inet " | grep -v 127.0.0.1
```

Example output: `inet 192.168.1.100/24`

#### Open in browser on Mac
```
http://192.168.1.100:7860
```

If authentication is enabled, you'll be prompted for username/password.

## Usage

1. **Upload Image**: Click "Upload skin lesion image" and select a `.jpg` file
2. **View Results**:
   - Grad-CAM heatmap overlay
   - Predicted class
   - Calibrated probabilities (JSON)
   - Melanoma decision (probability vs threshold)

## Troubleshooting

### Cannot connect from Mac

**Check 1: Server is running**
```bash
# On Ubuntu
sudo netstat -tulpn | grep 7860
```

**Check 2: Firewall allows connection**
```bash
# On Ubuntu
sudo ufw status | grep 7860
```

**Check 3: Network connectivity**
```bash
# On Mac
ping 192.168.1.100
telnet 192.168.1.100 7860
```

### GPU not being used

Check CUDA availability in the running server logs. If CPU fallback occurs:
```bash
# Verify NVIDIA driver
nvidia-smi

# Check PyTorch CUDA
/home/the/miniconda/envs/ml2/bin/python -c "import torch; print(torch.cuda.is_available())"
```

### Gradio crashes or OOM

Monitor GPU memory:
```bash
watch -n 1 nvidia-smi
```

The model uses ~2-3GB VRAM. Your 16GB GPU has plenty of headroom. If issues persist, check for:
- Other processes using GPU
- Memory leaks (restart the server)

## Security Best Practices

1. ✅ **Always set authentication** (`GRADIO_USERNAME` and `GRADIO_PASSWORD`) if accessible beyond localhost
2. ✅ **Keep .env out of git** (already in `.gitignore`)
3. ✅ **Use strong passwords** (12+ chars, mixed case, numbers, symbols)
4. ✅ **Firewall rules**: Restrict to trusted IPs/subnets
5. ✅ **HTTPS**: For production, use reverse proxy (nginx) with SSL cert
6. ✅ **Monitor logs**: Check for unusual access patterns

## Advanced: Systemd Service (Auto-start on boot)

Create `/etc/systemd/system/melanoma-gradio.service`:
```ini
[Unit]
Description=Melanoma Detection Gradio Service
After=network.target

[Service]
Type=simple
User=the
WorkingDirectory=/home/the/Codes/Melanoma-detection
Environment="PATH=/home/the/miniconda/envs/ml2/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=/home/the/miniconda/envs/ml2/bin/python src/serve_gradio.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable melanoma-gradio
sudo systemctl start melanoma-gradio
sudo systemctl status melanoma-gradio
```

## Monitoring

View logs:
```bash
# If using systemd
sudo journalctl -u melanoma-gradio -f

# If using tmux
tmux attach -t melanoma
```

Check GPU utilization:
```bash
nvidia-smi -l 1
```

## Backup and Updates

Before pulling updates:
```bash
cd /home/the/Codes/Melanoma-detection
# Backup your .env
cp .env .env.backup
# Pull updates
git pull
# Restore .env if needed
cp .env.backup .env
```

## Support

- Check logs first: `sudo journalctl -u melanoma-gradio -n 100`
- Verify `.env` configuration
- Test CLI first: `python -m src.inference.cli --help`
- GPU issues: Run `nvidia-smi` and check driver/CUDA versions
