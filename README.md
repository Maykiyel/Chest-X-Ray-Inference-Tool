# Chest X-Ray Inference Tool 🫁

A professional, GPU-accelerated application for chest X-ray analysis using state-of-the-art deep learning models from TorchXRayVision.

![Python](https://img.shields.io/badge/python-3.8--3.14-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 🎯 Python Version Compatibility

**Tested with:** Python 3.11  
**Compatible with:** Python 3.8 - 3.14.2 (latest as of December 2025)

✅ **Confirmed working on:**
- Python 3.8, 3.9, 3.10, 3.11 (tested)
- Python 3.12, 3.13, 3.14 (compatible - dependencies support these versions)

⚠️ **Important:** Do NOT use Python 3.7 or older (not supported by PyTorch 2.0+)

---

## 🧱 Project Structure (for easier coding/debugging)

The app is split into focused modules so you can iterate faster:

- `app.py` → Streamlit UI composition (tabs, widgets, layout)
- `app_services.py` → model caching + upload inference orchestration + summary helpers
- `app_state.py` → session-state defaults/initialization
- `app_constants.py` → shared pathology/default lists
- `inference.py` / `metrics.py` / `utils.py` → inference pipeline, evaluation, utilities

### Debug mode

In the sidebar, enable **Developer Tools → Enable debug panel** to inspect:
- selected device/models/pathology count
- cache key and last run stats (mode, image count, rows)
- preview of current in-memory results dataframe

This is useful when debugging performance, filtering behavior, and caching effects.

---

## 🌟 Features

✅ **Multi-Model Support**
- NIH Model (DenseNet121 - ChestX-ray14)
- MIMIC Model (DenseNet121 - MIMIC-CXR)
- CheXpert Model (DenseNet121 - CheXpert)

✅ **Advanced Capabilities**
- Custom pathology selection before scanning
- Single image and batch folder processing (150+ images)
- Automatic ground truth label detection from folder structure
- Smooth ROC/AUC curves with interpolation for small datasets
- GPU acceleration with CUDA support
- Real-time progress tracking with time estimates
- CSV export with full metadata
- Model comparison on single graph

✅ **Smart Features**
- Automatic "Normal" X-ray detection
- Interactive visualizations with Plotly
- Model comparison and ensemble analysis
- Configurable batch sizes and thresholds
- Dark, readable text on all graphs

---

## 🚀 Installation

### ⚡ Quick Setup (Recommended)

**One command does everything:**

```bash
python setup_script.py
```

**What it does:**
- ✅ Creates isolated virtual environment (`xray_env/`)
- ✅ Detects GPU and installs appropriate PyTorch version
- ✅ Installs ALL dependencies in the venv (nothing global!)
- ✅ Verifies installation
- ⏱️ Takes 10-15 minutes (first time only)

**That's it!** Everything installs in `xray_env/` - your global Python is untouched.

---

### 🔄 After Installation

**1. Activate virtual environment:**

```bash
# Git Bash CLI
source xray_env/Scripts/activate

# Windows - Command Prompt
xray_env\Scripts\activate.bat

# Windows - PowerShell
xray_env\Scripts\Activate.ps1

# Linux / macOS
source xray_env/bin/activate
```

**2. Run the application:**

```bash
streamlit run app.py
```

**3. Open your browser:**
- Automatically opens to `http://localhost:8501`
- Or manually navigate to that address

**4. When done:**
```bash
deactivate  # Exits the virtual environment
```

---

## 📦 What Gets Installed (All in venv!)

**Location:** Everything goes into `xray_env/` folder

**Size:**
- CPU version: ~1.5 GB
- GPU version: ~4 GB (includes CUDA libraries)

**Key packages:**
```
✓ PyTorch 2.0+ (CPU or GPU version)
✓ TorchXRayVision 1.2+
✓ Streamlit 1.28+
✓ Plotly 5.17+
✓ scikit-learn, pandas, numpy
✓ Image processing: Pillow, scikit-image, opencv-python
```

**Your global Python:**
- ❌ NOT modified
- ❌ No packages installed globally
- ✅ Remains clean and unchanged

---

## 🐍 Python Version Requirements

### Minimum: Python 3.8

### Recommended: Python 3.10 - 3.11

### Maximum: Python 3.14.2 (latest)

**Check your Python version:**
```bash
python --version
# or
python3 --version
```

**If you have Python 3.14:**
```bash
python3.14 setup_script.py
```

**If you have multiple Python versions:**
```bash
# Use specific version
python3.11 setup_script.py
# or
py -3.11 setup_script.py  # Windows
```

**Don't have Python?**
- Download from: https://www.python.org/downloads/
- For Windows: Check "Add Python to PATH" during installation
- For Linux: `sudo apt install python3 python3-venv`
- For macOS: `brew install python@3.11`

---

## 🔧 Manual Installation (If Automated Fails)

### Step 1: Create Virtual Environment

```bash
# Create venv
python -m venv xray_env

# Activate it
# Windows (Command Prompt)
xray_env\Scripts\activate.bat

# Windows (PowerShell)
xray_env\Scripts\Activate.ps1

# Linux/macOS
source xray_env/bin/activate
```

### Step 2: Install PyTorch

**For GPU (with CUDA):**
```bash
pip install torch>=2.0.0 torchvision>=0.15.0 --index-url https://download.pytorch.org/whl/cu118
```

**For CPU only:**
```bash
pip install torch>=2.0.0 torchvision>=0.15.0
```

### Step 3: Install Other Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 📖 Quick Start Guide

### 1. Select Your Pathologies

In the sidebar:
- Click **"Select Common"** for the 6 most important pathologies
- Or click **"Select All"** for all 15 pathologies
- Or manually select specific ones

**Available pathologies:**
- Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion
- Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule
- Pleural_Thickening, Pneumonia, Pneumothorax, Normal

### 2. Upload Images

**Option A: Multiple Images (Quick)**
```
1. Drag and drop 1-10 X-ray images
2. Click "🔍 Analyze Images"
3. View results in ~30 seconds
```

**Option B: Batch Processing (Large Datasets)**
```
1. Organize 150+ images in folders
2. Enter folder path
3. Set batch size (16 for GPU, 4 for CPU)
4. Enable "Auto-detect labels"
5. Click "🚀 Process Batch"
6. Monitor real-time progress
```

### 3. View Results

- **Results & Analysis** tab: View all predictions, filter, download CSV
- **ROC/AUC Curves** tab: Compare model performance (needs labeled data)

---

## 📁 Folder Structure for Auto-Labeling

For automatic ground truth detection:

```
my_dataset/
├── pneumonia_positive/
│   ├── img001.png
│   ├── img002.png
│   └── ... (150+ images supported)
├── pneumonia_negative/
│   ├── img003.png
│   ├── img004.png
│   └── ...
├── effusion_1/
│   ├── img005.png
└── effusion_0/
    ├── img006.png
```

**Supported naming patterns:**
- `{pathology}_positive` / `{pathology}_negative`
- `{pathology}_1` / `{pathology}_0`
- `positive_{pathology}` / `negative_{pathology}`
- `normal` (for normal X-rays)

---

## 🎯 Usage Examples

### Example 1: Quick Screening (Multiple Images)

```
Time: ~1 minute
Images: 5-10 X-rays

1. Select "Common" pathologies in sidebar
2. Upload 5-10 X-ray images
3. Click "Analyze Images"
4. Review predictions
5. Done!
```

### Example 2: Large Dataset (150+ images)

```
Time: ~10-30 minutes (GPU) or 1-2 hours (CPU)
Images: 150+ per pathology

1. Organize images in labeled folders
2. Select pathologies to analyze
3. Enter folder path
4. Set batch size: 16 (GPU) or 4 (CPU)
5. Enable "Auto-detect labels"
6. Click "Process Batch"
7. Watch real-time progress
8. Export to CSV
9. Generate ROC curves
```

### Example 3: Model Comparison

```
1. Process images with all 3 models
2. Go to "ROC/AUC Curves" tab
3. Select "All Models" in dropdown
4. Choose a pathology
5. Click "Generate ROC Curve"
6. See all models on one graph with performance table
```

---

## ⚙️ Configuration

### Batch Size Guidelines

| Hardware | Batch Size | Speed per Image |
|----------|-----------|-----------------|
| **GPU (8GB+ VRAM)** | 16-32 | ~2-5 seconds |
| **GPU (4-6GB VRAM)** | 8-16 | ~2-5 seconds |
| **CPU** | 1-4 | ~15-30 seconds |

**Rule of thumb:**
- GPU: Start with 16, increase if no errors
- CPU: Use 4 or less

### Device Selection

**GPU (CUDA):**
- Automatically detected if available
- 10-30x faster than CPU
- Requires NVIDIA GPU with CUDA support

**CPU:**
- Always available as fallback
- Good for small batches (<50 images)
- No GPU required

---

## 🔧 Troubleshooting

### ❌ Installation Issues

**Issue: `python: command not found`**
```bash
# Try python3
python3 setup_script.py
```

**Issue: PowerShell execution policy error (Windows)**
```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then run setup again
python setup_script.py
```

**Issue: "venv module not found"**
```bash
# Install python3-venv (Linux)
sudo apt install python3-venv

# Or use your Python version
sudo apt install python3.11-venv
```

**Issue: GPU not detected**
```
1. Check NVIDIA drivers: nvidia-smi
2. Install CUDA toolkit
3. Reinstall PyTorch with GPU support
4. Restart computer
```

---

### ⚠️ Runtime Issues

**Issue: "Out of memory"**
```
Solutions:
1. Reduce batch size to 1-4
2. Use only 1 model at a time
3. Close other applications
4. Restart app (clears cache)
```

**Issue: "No images found"**
```
Check:
1. Folder path is correct
2. Images are .png, .jpg, .jpeg, or .dcm
3. Enable "Search subfolders recursively"
4. Verify file permissions
```

**Issue: Models won't download**
```
1. Check internet connection
2. Check firewall settings
3. Models download on first use (~500MB each)
4. Wait patiently (can take 5-10 minutes)
```

**Issue: ROC curve not available**
```
Requirements:
1. Need labeled data (organize in folders)
2. Need at least 10 samples per pathology
3. Enable "Auto-detect labels from folder names"
4. Ground truth must be detected
```

**Issue: App is slow**
```
Expected:
- CPU: 15-30s per image (normal)
- GPU: 2-5s per image (normal)

Solutions:
- Use GPU for 10-30x speedup
- Reduce batch size
- Process overnight for large datasets
- Check GPU utilization: nvidia-smi
```

**Issue: Jagged ROC curves**
```
Normal for small datasets (<50 samples)
- Each step = different threshold
- More samples = smoother curves
- Interpolation applied automatically
```

---

### 🖼️ Image Quality Issues

**Best results:**
- ✅ Frontal chest X-rays (PA or AP view)
- ✅ DICOM or high-quality PNG/JPG  
- ✅ Full chest visible in image
- ✅ Good contrast and exposure

**Avoid:**
- ❌ Lateral views
- ❌ Severely cropped images
- ❌ Low resolution (<224x224)
- ❌ Heavy compression artifacts

---

## 📊 Understanding Results

### Pathology Predictions

Each prediction shows:
- **Pathology name:** e.g., Pneumonia, Effusion, Normal
- **Probability:** 0.0 to 1.0 (higher = more confident)
- **Model:** Which model made the prediction (NIH/MIMIC/CheXpert)

### Probability Interpretation

| Range | Meaning |
|-------|---------|
| **0.0 - 0.3** | Low likelihood |
| **0.3 - 0.6** | Moderate likelihood |
| **0.6 - 1.0** | High likelihood |

⚠️ **IMPORTANT:** These are **research predictions**, not clinical diagnoses. **Always consult medical professionals.**

### Normal Detection

The app calculates "Normal" probability as:
```
Normal = 1.0 - max(all pathology probabilities)
```

**Interpretation:**
- High normal probability (>0.7) → Likely normal X-ray
- Low normal probability (<0.3) → Likely abnormal finding

### ROC/AUC Analysis

- **AUC Score:** 0.5 = random guess, 1.0 = perfect
- **Typical range:** 0.70-0.95 for common pathologies
- **Smooth curves:** Interpolated for better visualization
- **Jagged curves:** Normal for small datasets (<50 samples)

---

## 🏗️ Project Structure

```
xray-inference-tool/
├── app.py                  # Main Streamlit application
├── inference.py            # Model loading and inference
├── utils.py                # Helper functions
├── metrics.py              # ROC/AUC calculations
├── config.py               # Configuration settings
├── setup_script.py         # Automated setup script
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore rules
├── README.md               # This file
└── xray_env/               # Virtual environment (created by setup)
```

**Total size:** ~200-500 KB (without venv)  
**With venv:** ~1.5-4 GB depending on CPU/GPU version

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📝 Citation

If using TorchXRayVision in research, please cite:

```bibtex
@article{Cohen2020xrv,
  title={TorchXRayVision: A library of chest X-ray datasets and models},
  author={Joseph Paul Cohen and Joseph D. Viviano and Paul Bertin and Paul Morrison and Parsa Torabian and Matteo Guarrera and Matthew P Lungren and Akshay Chaudhari and Rupert Brooks and Mohammad Hashir and Hadrien Bertrand},
  journal={Medical Imaging with Deep Learning},
  year={2020},
  url={https://github.com/mlmed/torchxrayvision}
}
```

---

## 📄 License

This project is licensed under the MIT License.

The underlying TorchXRayVision library and models have their own licenses - see https://github.com/mlmed/torchxrayvision

---

## ⚠️ Disclaimer

**This tool is for research and educational purposes only.** 

It is **NOT** intended for:
- ❌ Clinical diagnosis
- ❌ Medical decision making  
- ❌ Patient care
- ❌ Regulatory approval

**Always consult qualified medical professionals for health-related decisions.**

---

## 🔗 Resources

- **TorchXRayVision:** https://github.com/mlmed/torchxrayvision
- **Streamlit:** https://streamlit.io/
- **PyTorch:** https://pytorch.org/
- **Python Downloads:** https://www.python.org/downloads/

---

## 🎉 Quick Reference

| Command | Purpose |
|---------|---------|
| `python setup_script.py` | Complete automated setup |
| `xray_env\Scripts\activate` | Activate (Windows) |
| `source xray_env/bin/activate` | Activate (Linux/macOS) |
| `streamlit run app.py` | Run the application |
| `deactivate` | Exit virtual environment |
| `pip list` | See installed packages (in venv) |

---

## 💡 Tips & Best Practices

1. **Always activate venv** before running the app
2. **Select pathologies** before scanning for focused results
3. **Use GPU** for 10-30x faster processing
4. **Organize folders** properly for auto-labeling
5. **Start small** - test with 10 images before processing 1000
6. **Batch size** - bigger is faster but needs more memory
7. **ROC curves** - need at least 10 labeled samples
8. **Model comparison** - use "All Models" for quick comparison

---

## 🆘 Getting Help

**Setup Issues:**
- Re-run `python setup_script.py`
- Check Python version: `python --version`
- Ensure you have 3.8 - 3.14

**Runtime Issues:**
- Check troubleshooting section above
- Verify venv is activated: `which python` should show `xray_env`
- Clear cache: Restart app

**Still stuck?**
- Check error messages carefully
- Look for similar issues in troubleshooting
- Create detailed bug report with:
  - Python version
  - OS (Windows/Linux/macOS)
  - Error messages
  - Steps to reproduce

---

**Built with ❤️ using Streamlit and PyTorch**

**Python 3.8 - 3.14 compatible** | **Developed on Python 3.11** | **Tested December 2025**

---

## ✅ Installation Verification Checklist

After setup, verify everything works:

```bash
# 1. Check Python version (should be 3.8-3.14)
python --version

# 2. Check venv is activated (path should include xray_env)
# Windows
where python
# Linux/macOS
which python

# 3. Check PyTorch installation
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# 4. Check TorchXRayVision
python -c "import torchxrayvision; print('TorchXRayVision: OK')"

# 5. Run the app
streamlit run app.py
```

If all commands succeed → **✅ Installation successful!**

---

**Questions?** Check troubleshooting section or create an issue on GitHub!