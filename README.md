# Chest X-Ray Inference Tool 🫁

A professional, GPU-accelerated application for chest X-ray analysis using state-of-the-art deep learning models from TorchXRayVision.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

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

✅ **Smart Features**
- Automatic "Normal" X-ray detection
- Interactive visualizations with Plotly
- Model comparison and ensemble analysis
- Configurable batch sizes and thresholds

---

## 🚀 Installation

### One-Command Setup (Recommended)

```bash
python install_pytorch.py
```

That's it! The script automatically:
- ✅ Creates virtual environment
- ✅ Detects GPU and installs appropriate PyTorch version
- ✅ Installs all dependencies
- ✅ Verifies installation

**Time:** 10-15 minutes (first time only)

### After Installation

**Activate virtual environment:**

```bash
# Windows (Command Prompt)
xray_env\Scripts\activate.bat

# Windows (PowerShell)
xray_env\Scripts\Activate.ps1

# Linux/macOS
source xray_env/bin/activate
```

**Run the application:**

```bash
streamlit run app.py
```

**Open browser:** `http://localhost:8501`

---

## 📖 Quick Start Guide

### 1. Select Your Pathologies

In the sidebar, choose which pathologies to analyze:
- **Quick buttons:** "Select Common" or "Select All"
- **Custom selection:** Pick specific pathologies from the list

**Available pathologies:** Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural_Thickening, Pneumonia, Pneumothorax, Normal

### 2. Upload Images

**Option A: Multiple Images**
- Drag and drop images or click "Browse files"
- Select one or multiple X-ray images (PNG, JPG, JPEG, DICOM)
- Click "🔍 Analyze Images"
- View results immediately

**Option B: Batch Processing**
- Organize images in folders (see folder structure below)
- Enter folder path
- Set batch size (16 for GPU, 4 for CPU)
- Enable "Auto-detect labels" for ground truth
- Click "🚀 Process Batch"

### 3. View Results

- **Results & Analysis tab:** View predictions, filter, download CSV
- **ROC/AUC Curves tab:** Compare model performance (requires labeled data)

---

## 📁 Folder Structure for Auto-Labeling

For automatic ground truth detection:

```
dataset/
├── pneumonia_positive/
│   ├── image1.png
│   ├── image2.png
├── pneumonia_negative/
│   ├── image3.png
│   ├── image4.png
├── effusion_1/
│   ├── image5.png
└── effusion_0/
    ├── image6.png
```

**Supported naming patterns:**
- `{pathology}_positive` / `{pathology}_negative`
- `{pathology}_1` / `{pathology}_0`
- `positive_{pathology}` / `negative_{pathology}`
- `normal` (for normal X-rays)

---

## 🎯 Usage Examples

### Example 1: Quick Screening (Multiple Images)

1. Select "Common" pathologies in sidebar
2. Upload 5-10 X-ray images
3. Click "Analyze Images"
4. Review top predictions for each image
5. Done! (~30-60 seconds)

### Example 2: Large Dataset Analysis (150+ images)

1. Organize images by condition in folders
2. Select pathologies to focus on
3. Enter folder path
4. Set batch size to 16 (GPU) or 4 (CPU)
5. Enable "Auto-detect labels"
6. Click "Process Batch"
7. Monitor real-time progress
8. Export results to CSV
9. Generate ROC curves for model evaluation

### Example 3: Model Comparison

1. Process same images with all 3 models
2. Go to "Results & Analysis" tab
3. Compare predictions side-by-side
4. Use "ROC/AUC Curves" tab to evaluate performance
5. Select "All Models" for direct comparison

---

## ⚙️ Configuration

### Batch Size Guidelines

**GPU (with 8GB+ VRAM):**
- Batch size: 16-32
- Speed: ~2-5 seconds per image

**GPU (with 4-6GB VRAM):**
- Batch size: 8-16
- Speed: ~2-5 seconds per image

**CPU:**
- Batch size: 1-4
- Speed: ~15-30 seconds per image

### Device Selection

- **GPU (CUDA):** Automatically detected, 10-30x faster
- **CPU:** Always available, good for small batches

Switch between devices in the sidebar.

---

## 📊 Understanding Results

### Pathology Predictions

Each prediction shows:
- **Pathology name:** e.g., Pneumonia, Effusion, Normal
- **Probability:** 0.0 to 1.0 (higher = more confident)
- **Model:** Which model made the prediction

### Probability Interpretation

- **0.0 - 0.3:** Low likelihood
- **0.3 - 0.6:** Moderate likelihood
- **0.6 - 1.0:** High likelihood

⚠️ **Important:** These are research predictions, not clinical diagnoses. Always consult medical professionals.

### Normal Detection

The app automatically calculates "Normal" probability as:
```
Normal Probability = 1.0 - max(all pathology probabilities)
```

High normal probability indicates the X-ray appears normal.

### ROC/AUC Analysis

- **AUC Score:** 0.5 = random, 1.0 = perfect
- **Typical range:** 0.70-0.95 for common pathologies
- **Smooth curves:** Interpolated from actual data points
- **Jagged curves:** Normal for small datasets (<50 samples)

---

## 🔧 Troubleshooting

### Installation Issues

**Issue: "python: command not found"**
```bash
# Use python3 instead
python3 install_pytorch.py
```

**Issue: PowerShell execution policy error**
```powershell
# Run as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Issue: GPU not detected**
1. Check NVIDIA drivers: `nvidia-smi`
2. Install CUDA toolkit matching PyTorch version
3. Restart computer

### Runtime Issues

**Issue: "Out of memory"**
- Reduce batch size to 1-4
- Use only 1 model at a time
- Close other applications
- Clear CUDA cache (automatic)

**Issue: "No images found"**
- Check folder path is correct
- Verify image extensions (.png, .jpg, .jpeg, .dcm)
- Enable "Search subfolders recursively"

**Issue: "Models won't download"**
- Check internet connection
- Check firewall settings
- Models download automatically on first use (~500MB each)

**Issue: ROC curve not available**
- Need labeled data (organize in folders)
- Need at least 10 samples per pathology
- Enable "Auto-detect labels from folder names"

**Issue: App is slow**
- Expected on CPU (15-30s per image)
- Use GPU for 10-30x speedup
- Reduce batch size
- Process overnight for large datasets

### Image Quality Issues

**Best results:**
- ✅ Frontal chest X-rays (PA or AP view)
- ✅ DICOM or high-quality PNG/JPG
- ✅ Full chest visible
- ❌ Avoid lateral views
- ❌ Avoid severely cropped images

---

## 🏗️ Project Structure

```
xray-inference-tool/
├── app.py                  # Main Streamlit application
├── inference.py            # Model loading and inference
├── utils.py                # Helper functions
├── metrics.py              # ROC/AUC calculations
├── config.py               # Configuration settings
├── install_pytorch.py      # Automated setup script
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

**Total:** 8 essential files (~200-500 KB)

---

## 🤝 Contributing

Contributions are welcome! Please:
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

This project is licensed under the MIT License. See LICENSE file for details.

The underlying TorchXRayVision library and models have their own licenses - see https://github.com/mlmed/torchxrayvision for details.

---

## ⚠️ Disclaimer

**This tool is for research and educational purposes only.** It is not intended for clinical use or medical diagnosis. Always consult qualified medical professionals for health-related decisions.

---

## 🔗 Resources

- **TorchXRayVision:** https://github.com/mlmed/torchxrayvision
- **Streamlit:** https://streamlit.io/
- **PyTorch:** https://pytorch.org/
- **Report Issues:** [GitHub Issues](https://github.com/YOUR_USERNAME/xray-inference-tool/issues)

---

## 🎉 Quick Reference

| Command | Purpose |
|---------|---------|
| `python install_pytorch.py` | Complete setup |
| `xray_env\Scripts\activate` | Activate (Windows) |
| `source xray_env/bin/activate` | Activate (Linux/macOS) |
| `streamlit run app.py` | Run the app |
| `deactivate` | Deactivate environment |

---

**Built with ❤️ using Streamlit and PyTorch**

**Questions?** Open an issue on GitHub or check the troubleshooting section above.