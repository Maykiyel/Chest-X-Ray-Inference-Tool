# Quick Start Guide 🚀

Get up and running with the Chest X-Ray Inference Tool in 5 minutes!

## 1. Installation (2 minutes)

```bash
# Create a virtual environment (recommended)
python -m venv xray_env
source xray_env/Scripts/activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Launch the Application (30 seconds)

```bash
streamlit run app.py
```

Your browser will automatically open to `http://localhost:8501`

## 3. First Analysis (2 minutes)

### Option A: Single or Multiple Images

1. **Select Models** (sidebar):
   - ✅ Check "NIH Model"
   - ✅ Check "MIMIC Model"

2. **Upload Image(s)**:
   - Click "Browse files" 
   - Select one or multiple chest X-ray images (PNG/JPG)
   - Or drag and drop multiple files

3. **Analyze**:
   - Click "🔍 Analyze Images"
   - Wait ~5-10 seconds for first-time model download
   - View predictions!

### Option B: Batch Folder Processing (Large Datasets)

1. **Prepare Your Data**:
   ```
   my_xrays/
   ├── pneumonia_positive/
   │   ├── img1.png
   │   ├── img2.png
   │   └── ... (150+ images supported)
   └── pneumonia_negative/
       ├── img3.png
       └── ... (150+ images supported)
   ```

2. **Process**:
   - Enter folder path: `/path/to/my_xrays`
   - Set batch size: 16 (for GPU) or 4 (for CPU)
   - ✅ Check "Search subfolders recursively"
   - ✅ Check "Auto-detect labels"
   - Click "🚀 Process Batch"

3. **View Results**:
   - Real-time progress with estimated time
   - Processing statistics (time per image, total time)
   - Switch to "Results & Analysis" tab
   - Download CSV if needed
   - Generate ROC curves in "ROC/AUC Curves" tab

## 4. Understanding Results

### Pathology Predictions

Each prediction shows:
- **Pathology name**: e.g., "Pneumonia", "Effusion"
- **Probability**: 0.0 to 1.0 (higher = more confident)
- **Model**: Which model made the prediction

### Common Pathologies Detected

- **Atelectasis**: Collapsed lung tissue
- **Cardiomegaly**: Enlarged heart
- **Effusion**: Fluid around lungs
- **Pneumonia**: Lung infection
- **Pneumothorax**: Collapsed lung
- **Consolidation**: Lung solidification
- **Edema**: Fluid in lungs

### Interpreting Probabilities

- **0.0 - 0.3**: Low likelihood
- **0.3 - 0.6**: Moderate likelihood
- **0.6 - 1.0**: High likelihood

*Note: These are research predictions, not clinical diagnoses*

## 5. Tips for Best Results

### Image Quality
- ✅ Use frontal chest X-rays (PA or AP view)
- ✅ DICOM or high-quality PNG/JPG
- ✅ Full chest visible in image
- ❌ Avoid lateral views
- ❌ Avoid severely cropped images

### Performance
- 🔥 **GPU**: ~2-5 seconds per image
- 🐢 **CPU**: ~10-30 seconds per image

### Batch Size Guidelines
- **16-32 images**: Optimal for GPUs with 8GB+ VRAM
- **8-16 images**: Good for GPUs with 4-6GB VRAM
- **1-4 images**: Recommended for CPU processing
- **Large datasets**: Tool handles 150+ images per pathology efficiently

## 6. Example Workflows

### Workflow 1: Quick Screening (Multiple Images)
```
1. Upload 5-10 X-rays at once
2. Select all 3 models
3. Check top predictions for each
4. Review high-risk findings
5. Done! (~30-60 seconds)
```

### Workflow 2: Large Dataset Analysis (150+ images)
```
1. Organize 150+ images by condition in folders
2. Set batch size to 16 (GPU) or 4 (CPU)
3. Process batch with all models
4. Monitor real-time progress and statistics
5. Export to CSV
6. Generate ROC curves
7. Analyze in Excel/Python
```

### Workflow 3: Model Comparison
```
1. Process same images with different models
2. Go to "Results & Analysis"
3. Compare model predictions side-by-side
4. Use ROC curves to evaluate performance
```

## 7. Common Questions

**Q: Do I need a GPU?**  
A: No, but it's 5-10x faster. The app works fine on CPU.

**Q: How long does first run take?**  
A: 1-2 minutes to download models (one-time only).

**Q: What image formats work?**  
A: PNG, JPG, JPEG, and DICOM files.

**Q: Can I upload multiple images at once?**  
A: Yes! You can drag and drop or select multiple images for quick analysis.

**Q: Can I process 150+ images per pathology?**  
A: Yes! The tool is optimized for large datasets with automatic memory management.

**Q: How long does batch processing take?**  
A: ~2-5 seconds per image on GPU, ~10-30 seconds on CPU. The app shows real-time progress and estimated completion time.

**Q: Are predictions clinically reliable?**  
A: No. This is a research tool. Always consult medical professionals.

**Q: Can I use my own trained models?**  
A: Currently only TorchXRayVision models are supported.

## 8. Keyboard Shortcuts

- **R**: Rerun the app
- **Ctrl/Cmd + Enter**: Submit form
- **Ctrl/Cmd + Shift + R**: Clear cache

## 9. Next Steps

- 📖 Read the full [README.md](README.md) for detailed documentation
- 🔧 Adjust batch size in sidebar for your hardware
- 📊 Explore visualization options in Results tab
- 📈 Try ROC/AUC analysis with labeled datasets

## 10. Getting Help

**Application Issues:**
- Check `requirements.txt` versions
- Clear browser cache
- Restart the application

**Model Issues:**
- Ensure internet connection for first download
- Check CUDA installation for GPU support

**Still stuck?** 
- Review the troubleshooting section in README.md

---

**Happy analyzing! 🫁✨**