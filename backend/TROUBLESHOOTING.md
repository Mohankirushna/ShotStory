# ShotStory Backend - Troubleshooting Guide

## Common Issues and Solutions

### 🚨 Issue: Request Timeout on First Upload

**Symptoms:**
- Upload times out after 30-180 seconds
- Error: `asyncio.exceptions.CancelledError`
- Browser shows "Request timeout" or "Server not responding"

**Root Cause:**
First request must download and load 3 large AI models (~5GB total):
- BLIP captioning (~990MB)
- BLIP VQA (~2GB)
- DETR object detection (~167MB)
- Flan-T5-Large story generation (~3GB)

**Solutions:**

#### ✅ Solution 1: Increase Timeout (Already Fixed!)
The server now has a 5-minute timeout:
```python
# run.py
timeout_keep_alive=300  # 5 minutes
```

#### ✅ Solution 2: Preload Models at Startup (Recommended)
Create `.env` file in `backend/` directory:
```bash
PRELOAD_MODELS=true
```

Then restart the server. Models will load during startup (takes 2-5 minutes), and all subsequent requests will be fast.

#### ✅ Solution 3: Use the Preload Endpoint
Before uploading images, call:
```bash
curl -X POST http://localhost:8000/api/preload
```

Wait for it to complete (2-5 minutes), then upload your image.

---

### 🚨 Issue: AsyncIO Cancellation Errors

**Symptoms:**
```
ERROR: Traceback (most recent call last):
  ...
asyncio.exceptions.CancelledError
```

**Cause:**
- Server interrupted during model loading (Ctrl+C)
- Request cancelled before completion

**Solution:**
1. Let the first request complete fully (up to 5 minutes)
2. OR use `PRELOAD_MODELS=true` to load during startup
3. Restart the server cleanly if models are partially loaded

---

### 🚨 Issue: Slow Model Downloads

**Symptoms:**
- First request takes > 10 minutes
- Downloads seem stuck

**Solutions:**

#### Check Download Progress
Models cache to: `C:\Users\<YourName>\.cache\huggingface\hub`

Monitor active downloads:
```powershell
Get-ChildItem "$env:USERPROFILE\.cache\huggingface\hub" -Recurse -File | 
  Where-Object { $_.LastWriteTime -gt (Get-Date).AddMinutes(-10) } | 
  Format-Table Name, @{Name="SizeMB";Expression={[math]::Round($_.Length/1MB,2)}}, LastWriteTime
```

#### Speed Up Downloads
Set a HuggingFace token for faster downloads:
```bash
# In .env file
HF_TOKEN=your_huggingface_token_here
```

Get a free token at: https://huggingface.co/settings/tokens

---

### 🚨 Issue: Out of Memory (RAM)

**Symptoms:**
- Server crashes during model loading
- `MemoryError` or system freeze

**Minimum Requirements:**
- **RAM:** 8GB minimum, 16GB recommended
- **Disk:** 10GB free space for model cache

**Solution:**
If you have limited RAM, use lighter models in `.env`:
```bash
# Lighter alternatives
STORY_MODEL=google/flan-t5-base        # Instead of flan-t5-large (3GB → 1GB)
CAPTION_MODEL=Salesforce/blip-image-captioning-base  # Already using base model
```

---

### 🚨 Issue: Server Won't Start

**Check:**
1. Port 8000 is not in use:
```powershell
Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
```

2. Virtual environment is activated:
```powershell
.\venv\Scripts\Activate.ps1
```

3. Dependencies are installed:
```powershell
pip install -r requirements.txt
```

---

### 🚨 Issue: CUDA Out of Memory (GPU)

**Symptoms:**
- `RuntimeError: CUDA out of memory`

**Solution:**
Force CPU mode in `.env`:
```bash
CUDA_VISIBLE_DEVICES=""
```

Or upgrade to a GPU with more VRAM (8GB+ recommended).

---

## Performance Tips

### First Request Timeline
| Step | Time | What's Happening |
|------|------|------------------|
| 0-30s | Model downloads start | Checking cache → downloading missing models |
| 30s-3m | Downloading | BLIP + DETR + Flan-T5 downloading (~5GB) |
| 3-5m | Loading to memory | Loading models to CPU/GPU |
| 5m+ | Processing | Analyzing image + generating story |

### Subsequent Requests
After models are cached and loaded:
- **Image analysis:** 5-15 seconds
- **Story generation:** 10-30 seconds
- **Total:** 15-45 seconds per image

### Best Practice: Preload on Startup
Add to `.env`:
```bash
PRELOAD_MODELS=true
```

**Benefits:**
- ✅ Server startup takes 2-5 minutes (one-time cost)
- ✅ All requests are fast immediately
- ✅ No timeouts on first upload
- ✅ Better user experience

**Trade-off:**
- ❌ Longer startup time
- ❌ Uses more RAM even when idle

---

## Quick Reference

### Restart Server
```powershell
# Kill existing server
Get-Process python | Where-Object { $_.Path -like "*ShotStory*" } | Stop-Process

# Start fresh
cd backend
.\venv\Scripts\Activate.ps1
python run.py
```

### Clear Model Cache (Force Re-download)
```powershell
Remove-Item "$env:USERPROFILE\.cache\huggingface\hub" -Recurse -Force
```

### Check Server Health
```bash
curl http://localhost:8000/api/health
```

### Test Analysis
```bash
curl -X POST -F "file=@path/to/image.jpg" http://localhost:8000/api/analyze
```

---

## Still Having Issues?

1. Check the server terminal for detailed error logs
2. Enable debug mode in `.env`:
   ```bash
   DEBUG=true
   ```
3. Check system resources (RAM, disk space, network)
4. Ensure all dependencies are installed: `pip install -r requirements.txt`
