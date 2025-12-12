# Troubleshooting Guide

## Common Issues and Solutions

### Issue: KeyError 'model' with ChexPert Model

**Error Message:**
```
KeyError: 'model'
File "app.py", line 145, in <module>
    model_results = df[df['model'] == model_name].nlargest(5, 'probability')
```

**Cause:** The ChexPert model has a different API in TorchXRayVision.

**Solution:** 

The issue has been fixed in the updated `inference.py`. The ChexPert model now uses:
```python
model = xrv.models.DenseNet(weights="densenet121-res224-all")
```

This loads a DenseNet model trained on all available datasets, which includes ChexPert data.

**Alternative Models:**

If you want to use a different ChexPert configuration, you can modify `inference.py`:

1. **Option 1: Use the "all" model (recommended)**
   ```python
   elif name == 'chexpert':
       model = xrv.models.DenseNet(weights="densenet121-res224-all")
   ```

2. **Option 2: Use PC (PadChest) model**
   ```python
   elif name == 'chexpert':
       model = xrv.models.DenseNet(weights="densenet121-res224-pc")
   ```

3. **Option 3: Use NIH+PC model**
   ```python
   elif name == 'chexpert':
       model = xrv.models.DenseNet(weights="densenet121-res224-nih-pc")
   ```

### Issue: CUDA Out of Memory

**Error Message:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce Batch Size**: In the sidebar, decrease the batch size slider (try 4 or 2)

2. **Clear CUDA Cache**: Add this to your script:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

3. **Use Fewer Models**: Uncheck some models to reduce memory usage

4. **Use CPU Instead**: The app will automatically fall back to CPU if CUDA isn't available

### Issue: Models Won't Download

**Error Message:**
```
ConnectionError: Failed to download model weights
```

**Solutions:**

1. **Check Internet Connection**: Ensure you have a stable internet connection

2. **Check Firewall**: Some corporate firewalls block model downloads

3. **Manual Download**: Download models manually and place in cache:
   ```bash
   python -c "import torchxrayvision as xrv; xrv.models.DenseNet(weights='densenet121-res224-nih')"
   ```

4. **Use Proxy**: If behind a proxy, set environment variables:
   ```bash
   export HTTP_PROXY=http://proxy.example.com:8080
   export HTTPS_PROXY=http://proxy.example.com:8080
   ```

### Issue: No Images Found in Folder

**Error Message:**
```
⚠️ No images found in the specified folder!
```

**Solutions:**

1. **Check Path**: Ensure the folder path is correct and exists

2. **Check File Extensions**: Verify images have supported extensions:
   - .png
   - .jpg
   - .jpeg
   - .dcm
   - .dicom

3. **Enable Recursive Search**: Check "Search subfolders recursively" if images are in subdirectories

4. **Check Permissions**: Ensure you have read permissions for the folder

### Issue: Ground Truth Labels Not Detected

**Error Message:**
```
⚠️ No ground truth labels found in the data.
```

**Solutions:**

1. **Check Folder Naming**: Ensure folders follow supported patterns:
   ```
   dataset/
   ├── pneumonia_positive/
   ├── pneumonia_negative/
   ├── effusion_1/
   └── effusion_0/
   ```

2. **Enable Auto-Label**: Check "Auto-detect labels from folder names" option

3. **Supported Patterns**:
   - `{pathology}_positive` / `{pathology}_negative`
   - `{pathology}_1` / `{pathology}_0`
   - `positive_{pathology}` / `negative_{pathology}`

4. **Manual Labels**: Alternatively, provide a CSV with ground truth labels

### Issue: Streamlit Cache Errors

**Error Message:**
```
StreamlitAPIException: Cannot cache this object
```

**Solutions:**

1. **Clear Cache**: Press 'C' in the Streamlit app, then click "Clear cache"

2. **Restart App**: Stop the app (Ctrl+C) and restart:
   ```bash
   streamlit run app.py
   ```

3. **Delete Cache Folder**:
   ```bash
   rm -rf ~/.streamlit/cache/
   ```

### Issue: Import Errors

**Error Message:**
```
ModuleNotFoundError: No module named 'torchxrayvision'
```

**Solutions:**

1. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Upgrade pip**:
   ```bash
   pip install --upgrade pip
   ```

3. **Use Virtual Environment**:
   ```bash
   python -m venv xray_env
   source xray_env/bin/activate  # Windows: xray_env\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Install Specific Package**:
   ```bash
   pip install torchxrayvision
   ```

### Issue: Slow Performance on CPU

**Symptom:** Each image takes 30+ seconds to process

**Solutions:**

1. **Reduce Batch Size**: Set to 1 or 2 for CPU processing

2. **Use Fewer Models**: Process with one model at a time

3. **Upgrade Hardware**: Consider using a machine with GPU

4. **Process Overnight**: For large batches, let it run overnight

### Issue: Incorrect Predictions

**Symptom:** Predictions seem wrong or inconsistent

**Solutions:**

1. **Check Image Quality**: Ensure images are actual chest X-rays
   - Frontal view (PA or AP)
   - Full chest visible
   - Good contrast

2. **Check Image Format**: Some DICOM files may need special handling

3. **Multiple Models**: Compare predictions across all three models

4. **Remember Limitations**: These are research models, not clinical tools

### Issue: ROC Curve Won't Generate

**Error Message:**
```
⚠️ Not enough data points for ROC analysis (need at least 10)
```

**Solutions:**

1. **More Images**: Process at least 10 images with the same pathology label

2. **Check Labels**: Ensure ground truth labels are detected correctly

3. **Balance Classes**: Have both positive and negative examples

4. **Check Pathology Name**: Ensure the pathology name matches your folder names

### Issue: CSV Export is Empty

**Symptom:** Downloaded CSV has only headers, no data

**Solutions:**

1. **Run Inference First**: Make sure you've processed images before exporting

2. **Check Filters**: Reset filters in the "Results & Analysis" tab

3. **Refresh App**: Try rerunning the inference

### Issue: Memory Leak (RAM Usage Keeps Increasing)

**Symptom:** Application gets slower over time

**Solutions:**

1. **Restart App**: Restart Streamlit app periodically for large batches

2. **Clear Session State**: Press 'C' and clear cache

3. **Process in Chunks**: Instead of processing 1000 images at once, do 100 at a time

4. **Monitor Resources**:
   ```bash
   # Linux/Mac
   htop
   
   # Windows
   Task Manager > Performance
   ```

### Issue: Windows-Specific Path Errors

**Error Message:**
```
FileNotFoundError: [WinError 3] The system cannot find the path specified
```

**Solutions:**

1. **Use Forward Slashes**: 
   ```
   C:/Users/David/Desktop/xrays
   ```
   Instead of:
   ```
   C:\Users\David\Desktop\xrays
   ```

2. **Use Raw Strings** (if editing code):
   ```python
   path = r"C:\Users\David\Desktop\xrays"
   ```

3. **Use Path Object**:
   ```python
   from pathlib import Path
   path = Path("C:/Users/David/Desktop/xrays")
   ```

## Getting More Help

### Debug Mode

Add this to the top of `app.py` to enable debug output:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Versions

```python
import torch
import torchxrayvision as xrv
import streamlit as st

print(f"PyTorch: {torch.__version__}")
print(f"TorchXRayVision: {xrv.__version__}")
print(f"Streamlit: {st.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
```

### Report an Issue

When reporting issues, include:

1. Full error message and traceback
2. Python version: `python --version`
3. Operating system
4. GPU info (if applicable)
5. Steps to reproduce

### Useful Links

- **TorchXRayVision GitHub**: https://github.com/mlmed/torchxrayvision
- **Streamlit Docs**: https://docs.streamlit.io
- **PyTorch Docs**: https://pytorch.org/docs

## Prevention Tips

1. **Keep Dependencies Updated**:
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **Use Virtual Environments**: Always use a clean virtual environment

3. **Test on Small Dataset First**: Before processing 1000 images, test with 10

4. **Monitor Resources**: Keep an eye on CPU/GPU/RAM usage

5. **Regular Restarts**: Restart the app every few hours for large processing jobs

---

**Last Updated:** December 2024

If your issue isn't listed here, check the README.md or create an issue on GitHub.