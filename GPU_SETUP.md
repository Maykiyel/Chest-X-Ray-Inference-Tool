# GPU Setup Guide for Chest X-Ray Inference Tool

This guide will help you enable GPU acceleration for faster processing.

## Prerequisites Check

### 1. Check if you have an NVIDIA GPU

**Windows:**
```bash
# Open Command Prompt or PowerShell
nvidia-smi
```

**What to look for:**
- If you see GPU information → You have an NVIDIA GPU ✅
- If you get an error → Either no NVIDIA GPU or drivers not installed ❌

### 2. Check Current PyTorch Installation

```python
# Open Python and run:
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
```

**Expected Output (with GPU):**
```
PyTorch: 2.1.0+cu118
CUDA Available: True
CUDA Version: 11.8
```

**Current Output (CPU only):**
```
PyTorch: 2.1.0+cpu
CUDA Available: False
CUDA Version: N/A
```

## Installation Steps

### Step 1: Check CUDA Compatibility

First, find out which CUDA version your GPU supports:

```bash
nvidia-smi
```

Look for "CUDA Version" in the top-right corner (e.g., "CUDA Version: 12.1")

### Step 2: Uninstall CPU-Only PyTorch

```bash
# Activate your virtual environment first
cd C:\Users\David\Desktop\xray-inference-tool
xray_env\Scripts\activate

# Uninstall current PyTorch
pip uninstall torch torchvision torchaudio
```

### Step 3: Install GPU-Enabled PyTorch

Visit: https://pytorch.org/get-started/locally/

**For CUDA 11.8 (Most Common - RTX 30/40 series):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1 (Latest GPUs):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 11.7 (Older GPUs):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

### Step 4: Verify Installation

```python
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

**Expected Output:**
```
CUDA Available: True
CUDA Device: NVIDIA GeForce RTX 3080
```

### Step 5: Restart Streamlit App

```bash
streamlit run app.py
```

Check the sidebar - it should now show:
```
🖥️ Using: CUDA
GPU: NVIDIA GeForce RTX [Your Model]
```

## Troubleshooting

### Issue 1: "CUDA Available: False" after installation

**Solution 1: Check CUDA Toolkit Installation**
```bash
nvcc --version
```

If command not found, install CUDA Toolkit:
- Download from: https://developer.nvidia.com/cuda-downloads
- Install the version matching your PyTorch installation

**Solution 2: Update NVIDIA Drivers**
- Download latest drivers: https://www.nvidia.com/Download/index.aspx
- Restart computer after installation

**Solution 3: Environment Variables (Windows)**
Add these to System Environment Variables:
```
CUDA_PATH = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
PATH += C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
```

### Issue 2: "RuntimeError: CUDA out of memory"

**Solutions:**
1. **Reduce Batch Size** in the app sidebar (try 4 or 8)
2. **Close other GPU applications** (games, video editors, browsers with hardware acceleration)
3. **Clear CUDA Cache:**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```
4. **Check GPU Memory:**
   ```bash
   nvidia-smi
   ```
   Look at "Memory-Usage" - if it's already high, close other applications

### Issue 3: Wrong CUDA Version Installed

**Check what CUDA versions are compatible:**
```bash
nvidia-smi
```

The "CUDA Version" shown is the MAXIMUM supported. You can install any lower version.

**Reinstall with correct version:**
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu[VERSION]
```

Replace `[VERSION]` with: 118, 121, 117, etc.

### Issue 4: Multiple CUDA Versions Installed

**Find all CUDA installations:**
```bash
# Windows
where nvcc

# Check PyTorch's CUDA:
python -c "import torch; print(torch.version.cuda)"
```

**Solution:** Uninstall conflicting versions, keep only one.

## Performance Expectations

### Speed Comparison (Per Image):

| Hardware | Batch Size 1 | Batch Size 8 | Batch Size 16 |
|----------|--------------|--------------|---------------|
| CPU (i7-10700K) | 15-30s | 12-25s | 10-20s |
| RTX 3060 | 3-5s | 1-2s | 0.8-1.5s |
| RTX 3080 | 2-3s | 0.5-1s | 0.3-0.7s |
| RTX 4090 | 1-2s | 0.3-0.5s | 0.2-0.4s |

### VRAM Requirements:

| Batch Size | VRAM Needed |
|------------|-------------|
| 1 | ~2 GB |
| 4 | ~3 GB |
| 8 | ~4 GB |
| 16 | ~6 GB |
| 32 | ~10 GB |
| 64 | ~18 GB |

## Quick Reference Commands

### Check GPU Status:
```bash
nvidia-smi
```

### Check PyTorch CUDA:
```python
python -c "import torch; print(torch.cuda.is_available())"
```

### Monitor GPU in Real-Time:
```bash
# Windows PowerShell
nvidia-smi -l 1
```

### Clear GPU Memory:
```python
import torch
torch.cuda.empty_cache()
```

## Recommended Settings by GPU

### RTX 3060 (12GB):
- Batch Size: 16
- Expected Speed: ~1-2s per image
- Can handle: 3 models simultaneously

### RTX 3070/3080 (8-10GB):
- Batch Size: 16-24
- Expected Speed: ~0.5-1s per image
- Can handle: 3 models simultaneously

### RTX 4070/4080/4090 (12-24GB):
- Batch Size: 32-64
- Expected Speed: ~0.3-0.5s per image
- Can handle: All models with no issues

### GTX 1660/1070 (6GB):
- Batch Size: 8-12
- Expected Speed: ~2-3s per image
- Can handle: 2 models at a time (recommended)

## Alternative: Google Colab (Free GPU)

If you don't have an NVIDIA GPU, use Google Colab:

1. **Create Colab Notebook:** https://colab.research.google.com
2. **Enable GPU:** Runtime → Change runtime type → GPU
3. **Install Dependencies:**
   ```python
   !pip install torchxrayvision streamlit
   ```
4. **Upload Files:** Use Colab's file upload feature
5. **Run Inference:**
   ```python
   from inference import load_models, predict_single_image
   # Your code here
   ```

**Limitations:**
- 12-hour session limit
- Manual file uploads
- No persistent storage

## Still Having Issues?

### Get Detailed System Info:

```python
import torch
import sys

print("="*60)
print("SYSTEM INFORMATION")
print("="*60)
print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("No CUDA devices found!")
print("="*60)
```

### Contact Support:

Include the output above when asking for help.

## Summary Checklist

- [ ] NVIDIA GPU present
- [ ] Latest NVIDIA drivers installed
- [ ] CUDA Toolkit installed (matching PyTorch version)
- [ ] PyTorch with CUDA support installed
- [ ] `torch.cuda.is_available()` returns `True`
- [ ] Streamlit app shows "Using: CUDA"
- [ ] Performance improvement observed

---

**Note:** GPU acceleration is optional. The tool works perfectly fine on CPU, just slower.

**Estimated setup time:** 15-30 minutes