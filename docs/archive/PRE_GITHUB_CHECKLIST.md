# Pre-GitHub Checklist

Complete these steps before pushing to GitHub:

## 1. Create your .env file (DO THIS FIRST!)

```bash
cd /home/the/Codes/Melanoma-detection
cp .env.example .env
nano .env
```

Set at minimum:
```bash
GRADIO_USERNAME=your_username
GRADIO_PASSWORD=your_secure_password
```

## 2. Verify .env is gitignored

```bash
git status
# Should NOT see .env in the output
```

If you see `.env` in git status:
```bash
git rm --cached .env
git add .gitignore
git commit -m "fix: ensure .env is ignored"
```

## 3. Security audit

```bash
# Check for hardcoded secrets in code
grep -r "password\|api_key\|secret\|token" src/ --exclude-dir=__pycache__ --exclude="*.pyc"
# Should find references to env vars, not actual secrets

# Check staged files for secrets
git diff --cached | grep -E "(password|api_key|secret|token)" && echo "⚠️ Review staged changes!" || echo "✅ Safe"
```

## 4. Test the setup

```bash
# Test config validation
/home/the/miniconda/envs/ml2/bin/python -c "from src.config import Config; Config.validate(); print('✅ OK')"

# Test CLI (non-interactive)
/home/the/miniconda/envs/ml2/bin/python -m src.inference.cli \
  --image data/ds/img/ISIC_0027990.jpg \
  --weights models/checkpoints/melanoma_resnet50_nb.pth \
  --label-map models/label_maps/label_map_nb.json \
  --no-ask

# Run smoke test
/home/the/miniconda/envs/ml2/bin/python -m unittest tests/test_smoke_inference.py
```

## 5. Initialize git (if not done)

```bash
cd /home/the/Codes/Melanoma-detection
git init
git add .
git commit -m "Initial commit: melanoma detection with XAI"
```

## 6. Create GitHub repository

### Option A: Via GitHub CLI (gh)
```bash
gh repo create melanoma-detection --private --source=. --remote=origin
gh repo edit --description "Automated melanoma detection with calibrated probabilities, operating thresholds, and Grad-CAM explanations"
git push -u origin main
```

### Option B: Via web browser
1. Go to https://github.com/new
2. Repository name: `melanoma-detection`
3. **Set to Private** (recommended for medical data)
4. Do NOT initialize with README (you already have one)
5. Click "Create repository"
6. Follow the push instructions:

```bash
git remote add origin https://github.com/YOUR_USERNAME/melanoma-detection.git
git branch -M main
git push -u origin main
```

## 7. Post-push verification

```bash
# Clone to a temp directory and verify
cd /tmp
git clone https://github.com/YOUR_USERNAME/melanoma-detection.git test-clone
cd test-clone

# Check .env is NOT there
ls -la | grep .env
# Should only see .env.example

# Check .gitignore is working
cat .gitignore | grep "\.env"
```

## 8. Setup branch protection (optional, recommended)

On GitHub web:
1. Go to repo → Settings → Branches
2. Add branch protection rule for `main`
3. Enable:
   - Require pull request reviews before merging
   - Require status checks to pass (if you add CI later)

## 9. Add collaborators (if needed)

On GitHub web:
1. Go to repo → Settings → Collaborators
2. Add team members
3. Set permissions (read/write/admin)

## 10. Configure firewall on server

```bash
# Allow Gradio from your local network
sudo ufw allow from 192.168.1.0/24 to any port 7860 comment 'Gradio melanoma'

# Check status
sudo ufw status numbered
```

## 11. Test Gradio from Mac

```bash
# On Ubuntu server: Start Gradio
cd /home/the/Codes/Melanoma-detection
tmux new -s melanoma
/home/the/miniconda/envs/ml2/bin/python src/serve_gradio.py
# Press Ctrl+B then D to detach

# On Mac: Open browser
# Get server IP: run on Ubuntu: ip addr show | grep "inet " | grep -v 127.0.0.1
open http://192.168.1.XXX:7860
```

## ✅ Checklist Summary

- [ ] Created `.env` with username/password
- [ ] Verified `.env` is gitignored
- [ ] No hardcoded secrets in code
- [ ] All tests pass
- [ ] GitHub repo created (private recommended)
- [ ] Code pushed successfully
- [ ] `.env` NOT in remote repo (double check!)
- [ ] Firewall configured on server
- [ ] Gradio accessible from Mac with auth

## 🚨 If you accidentally pushed .env

```bash
# Remove from history (DESTRUCTIVE - coordinate with team first!)
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch .env" \
  --prune-empty --tag-name-filter cat -- --all

# Force push (WARNING: overwrites remote history)
git push origin --force --all

# Rotate all secrets that were in .env!
# Change passwords, regenerate API keys, etc.
```

## 📚 Next Steps

After GitHub setup:
- [ ] Add GitHub Actions for CI (run tests on push)
- [ ] Add issue/PR templates
- [ ] Setup Dependabot for dependency updates
- [ ] Add SECURITY.md with vulnerability reporting process
