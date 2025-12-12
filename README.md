# Chest X-Ray Inference Tool 🫁

A professional, GPU-accelerated application for chest X-ray analysis using state-of-the-art deep learning models from TorchXRayVision.

## Features

✅ **Multi-Model Support**
- NIH Model (DenseNet121)
- MIMIC Model
- CheXpert Baseline Model

✅ **Advanced Capabilities**
- Single image and batch folder processing
- GPU acceleration with CUDA support
- Automatic ground truth label detection from folder structure
- ROC/AUC analysis with interactive plots
- CSV export of all predictions
- Real-time progress tracking

✅ **Performance Optimizations**
- Model caching for fast repeated inference
- Batch processing with PyTorch DataLoader
- Efficient preprocessing pipeline
- Memory-optimized inference with `torch.no_grad()`

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional, but recommended for performance)

### Setup

1. **Clone or download this repository**

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Verify installation**
```bash
python -c "import torchxrayvision; print('Installation successful!')"
```

## Usage

### Starting the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Multiple Image Analysis

1. Navigate to the **"Single/Batch Inference"** tab
2. Select the models you want to use in the sidebar
3. Upload one or more X-ray images (PNG, JPG, JPEG, DICOM)
   - Drag and drop multiple files
   - Or click "Browse files" to select multiple images
4. Click **"Analyze Images"**
5. View results and top predictions

### Batch Folder Processing

1. Navigate to the **"Single/Batch Inference"** tab
2. Select the models you want to use
3. Enter the path to your image folder
4. Adjust batch size (recommended: 16 for GPU, 4 for CPU)
5. Enable "Search subfolders recursively" if needed
6. Enable "Auto-detect labels from folder names" for ground truth detection
7. Click **"Process Batch"**
8. View real-time progress, estimated time, and statistics
9. Results available in the **"Results & Analysis"** tab

**Note:** The tool can efficiently handle large datasets (150+ images per pathology)

### Folder Structure for Auto-Labeling

To enable automatic ground truth detection, organize your images like this:

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

Supported naming patterns:
- `{pathology}_positive` / `{pathology}_negative`
- `{pathology}_1` / `{pathology}_0`
- `positive_{pathology}` / `negative_{pathology}`

### ROC/AUC Analysis

1. Process images with ground truth labels
2. Navigate to the **"ROC/AUC Curves"** tab
3. Select a model and pathology
4. Click **"Generate ROC Curve"**
5. View AUC score and interactive ROC plot

### Exporting Results

1. Navigate to the **"Results & Analysis"** tab
2. Apply filters if desired
3. Click **"Download Results CSV"**
4. The CSV contains all predictions with timestamps

## File Structure

```
xray-inference-tool/
├── app.py              # Main Streamlit application
├── inference.py        # Model loading and inference logic
├── utils.py            # Helper functions and utilities
├── metrics.py          # ROC/AUC calculations and plotting
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Supported Models

### NIH Model
- **Architecture**: DenseNet121
- **Training Data**: NIH ChestX-ray14 dataset
- **Pathologies**: 14 thoracic diseases

### MIMIC Model
- **Architecture**: DenseNet121
- **Training Data**: MIMIC-CXR dataset
- **Pathologies**: Multiple thoracic conditions

### CheXpert Model
- **Architecture**: DenseNet121
- **Training Data**: CheXpert dataset
- **Pathologies**: 14 observations

## Performance Tips

### For Best Performance:

1. **Use GPU**: The application automatically detects and uses CUDA if available
2. **Batch Processing**: Process multiple images at once for better efficiency
3. **Adjust Batch Size**: 
   - GPU with 8GB+ VRAM: Use batch size 16-32
   - GPU with 4-6GB VRAM: Use batch size 8-16
   - CPU: Use batch size 1-4
4. **Model Caching**: Models are cached after first load - keep the app running for repeated inference
5. **Large Datasets**: The tool efficiently handles 150+ images per pathology with automatic memory management

### Memory Requirements:

- **CPU Mode**: ~2-4 GB RAM
- **GPU Mode**: ~4-8 GB VRAM (depending on batch size)
- **Large batches (150+ images)**: Handled efficiently with automatic CUDA cache clearing

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size in the sidebar settings

### Issue: "No images found"
**Solution**: 
- Check the folder path is correct
- Ensure images have supported extensions (.png, .jpg, .jpeg, .dcm)
- Enable "Search subfolders recursively" if images are in subdirectories

### Issue: Models fail to load
**Solution**: 
- Ensure you have an internet connection (models download on first use)
- Check that `torchxrayvision` is properly installed
- Try reinstalling: `pip install --upgrade torchxrayvision`

### Issue: ROC curve not available
**Solution**: 
- Ensure images have ground truth labels
- Check folder naming follows supported patterns
- Enable "Auto-detect labels from folder names"

## Technical Details

### Preprocessing Pipeline
1. Images loaded with scikit-image
2. Converted to grayscale if needed
3. Normalized to [0, 1] range
4. Resized to 224x224 pixels
5. Normalized using dataset statistics: (x - 0.5) / 0.5

### Inference Pipeline
1. Images batched using PyTorch DataLoader
2. Batch moved to GPU (if available)
3. Forward pass with `torch.no_grad()`
4. Sigmoid activation for probabilities
5. Results collected and formatted

### Metrics Calculations
- **ROC/AUC**: Computed using `sklearn.metrics.roc_curve` and `roc_auc_score`
- **Optimal Threshold**: Determined using Youden's J statistic
- All metrics follow scikit-learn standards

## Citation

If you use this tool in your research, please cite TorchXRayVision:

```bibtex
@article{Cohen2020xrv,
  title={TorchXRayVision: A library of chest X-ray datasets and models},
  author={Joseph Paul Cohen and Joseph D. Viviano and Paul Bertin and Paul Morrison and Parsa Torabian and Matteo Guarrera and Matthew P Lungren and Akshay Chaudhari and Rupert Brooks and Mohammad Hashir and Hadrien Bertrand},
  journal={Medical Imaging with Deep Learning},
  year={2020},
  url={https://github.com/mlmed/torchxrayvision}
}
```

## License

This tool is provided as-is for research and educational purposes. The underlying TorchXRayVision library and models have their own licenses - please refer to the [TorchXRayVision repository](https://github.com/mlmed/torchxrayvision) for details.

## Support

For issues related to:
- **This tool**: Check the troubleshooting section above
- **TorchXRayVision**: Visit https://github.com/mlmed/torchxrayvision
- **Streamlit**: Visit https://docs.streamlit.io

## Acknowledgments

- TorchXRayVision team for the excellent models and library
- Streamlit for the amazing web framework
- The medical imaging community for open datasets

---

**Built with ❤️ using Streamlit and PyTorch**