import torch
import torchxrayvision as xrv
import numpy as np
from PIL import Image
import streamlit as st
from pathlib import Path
from datetime import datetime
import skimage
from torch.utils.data import Dataset, DataLoader
from utils import extract_folder_label

@st.cache_resource
def load_models(model_names, device='cpu'):
    """
    Load and cache TorchXRayVision models.
    
    Args:
        model_names: List of model names ('nih', 'mimic', 'chexpert')
        device: 'cuda' or 'cpu'
    
    Returns:
        Dictionary of {model_name: model}
    """
    models = {}
    
    for name in model_names:
        try:
            if name == 'nih':
                model = xrv.models.DenseNet(weights="densenet121-res224-nih")
            elif name == 'mimic':
                model = xrv.models.DenseNet(weights="densenet121-res224-mimic_nb")
            elif name == 'chexpert':
                # ChexPert model uses different loading method
                model = xrv.models.DenseNet(weights="densenet121-res224-all")
                # Alternative: if you want the actual baseline model
                # import torchxrayvision.baseline_models as baseline_models
                # model = baseline_models.chexpert.DenseNet(weights="densenet121-res224-chexpert")
            else:
                continue
            
            model = model.to(device)
            model.eval()
            models[name] = model
            st.success(f"✓ Loaded {name.upper()} model")
            
        except Exception as e:
            st.error(f"Failed to load {name} model: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
    
    return models


def preprocess_image(img_path, target_size=224):
    """
    Preprocess image for TorchXRayVision models.
    
    Args:
        img_path: Path to image file
        target_size: Target size for resizing (default 224)
    
    Returns:
        Preprocessed image tensor
    """
    # Read image
    img = skimage.io.imread(str(img_path))
    
    # Convert to grayscale if needed
    if len(img.shape) > 2:
        img = skimage.color.rgb2gray(img)
    
    # Normalize to [0, 1]
    img = img.astype(np.float32)
    if img.max() > 1:
        img = img / 255.0
    
    # Resize
    img = skimage.transform.resize(img, (target_size, target_size), mode='constant')
    
    # Normalize using dataset statistics
    # TorchXRayVision uses specific normalization
    img = (img - 0.5) / 0.5
    
    # Add channel dimension and convert to tensor
    img = torch.from_numpy(img).unsqueeze(0)  # [1, H, W]
    
    return img


def predict_single_image(img_path, model, model_name, device='cpu'):
    """
    Run inference on a single image.
    
    Args:
        img_path: Path to image
        model: Loaded model
        model_name: Name of the model
        device: Device to use
    
    Returns:
        List of prediction dictionaries
    """
    # Preprocess
    img_tensor = preprocess_image(img_path).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(img_tensor)
    
    # Get predictions
    predictions = torch.sigmoid(outputs).cpu().numpy()[0]
    
    # Format results
    results = []
    pathology_names = model.pathologies
    
    # Add pathology predictions
    for pathology, prob in zip(pathology_names, predictions):
        results.append({
            'filename': Path(img_path).name,
            'filepath': str(img_path),
            'model': model_name,
            'pathology': pathology,
            'probability': float(prob),
            'timestamp': datetime.now().isoformat()
        })
    
    # Calculate "Normal" probability
    # Normal = 1 - max(all pathology probabilities)
    # This represents the likelihood that no significant pathology is present
    max_pathology_prob = float(np.max(predictions))
    normal_prob = 1.0 - max_pathology_prob
    
    # Add Normal as a prediction
    results.append({
        'filename': Path(img_path).name,
        'filepath': str(img_path),
        'model': model_name,
        'pathology': 'Normal',
        'probability': normal_prob,
        'timestamp': datetime.now().isoformat()
    })
    
    return results


class XRayDataset(Dataset):
    """Custom dataset for batch processing of X-ray images."""
    
    def __init__(self, image_paths, target_size=224):
        self.image_paths = image_paths
        self.target_size = target_size
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            img_tensor = preprocess_image(img_path, self.target_size)
            return img_tensor.squeeze(0), str(img_path), True
        except Exception as e:
            # Return dummy data if image fails to load
            return torch.zeros(self.target_size, self.target_size), str(img_path), False


def predict_batch(image_paths, model, model_name, device='cpu', 
                 batch_size=8, auto_label=True, progress_callback=None):
    """
    Run batch inference on multiple images.
    
    Args:
        image_paths: List of image paths
        model: Loaded model
        model_name: Name of the model
        device: Device to use
        batch_size: Batch size for inference
        auto_label: Whether to extract labels from folder names
        progress_callback: Callback function for progress updates
    
    Returns:
        List of prediction dictionaries
    """
    # Create dataset and dataloader
    dataset = XRayDataset(image_paths)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 for Streamlit compatibility
        pin_memory=(device == 'cuda')
    )
    
    results = []
    total_batches = len(dataloader)
    
    # Process batches
    for batch_idx, (images, paths, valid_flags) in enumerate(dataloader):
        # Filter valid images
        valid_indices = [i for i, flag in enumerate(valid_flags) if flag]
        
        if len(valid_indices) == 0:
            continue
        
        valid_images = images[valid_indices].to(device)
        valid_paths = [paths[i] for i in valid_indices]
        
        # Inference
        with torch.no_grad():
            outputs = model(valid_images)
            predictions = torch.sigmoid(outputs).cpu().numpy()
        
        # Process predictions
        pathology_names = model.pathologies
        
        for i, (pred, img_path) in enumerate(zip(predictions, valid_paths)):
            img_path_obj = Path(img_path)
            
            # Extract ground truth label if auto_label is enabled
            ground_truth = None
            if auto_label:
                ground_truth = extract_folder_label(img_path_obj)
            
            # Add pathology predictions
            for pathology, prob in zip(pathology_names, pred):
                result_dict = {
                    'filename': img_path_obj.name,
                    'filepath': str(img_path),
                    'model': model_name,
                    'pathology': pathology,
                    'probability': float(prob),
                    'timestamp': datetime.now().isoformat()
                }
                
                if ground_truth is not None:
                    result_dict['ground_truth'] = ground_truth.get(pathology, 0)
                
                results.append(result_dict)
            
            # Calculate and add "Normal" probability
            max_pathology_prob = float(np.max(pred))
            normal_prob = 1.0 - max_pathology_prob
            
            normal_dict = {
                'filename': img_path_obj.name,
                'filepath': str(img_path),
                'model': model_name,
                'pathology': 'Normal',
                'probability': normal_prob,
                'timestamp': datetime.now().isoformat()
            }
            
            if ground_truth is not None:
                normal_dict['ground_truth'] = ground_truth.get('Normal', 0)
            
            results.append(normal_dict)
        
        # Clear CUDA cache periodically to prevent memory buildup
        if device == 'cuda' and batch_idx % 10 == 0:
            torch.cuda.empty_cache()
        
        # Update progress
        if progress_callback:
            progress = (batch_idx + 1) / total_batches
            progress_callback(progress)
    
    # Final cleanup
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    return results


def get_model_info(model):
    """Get information about a model."""
    info = {
        'pathologies': model.pathologies,
        'num_pathologies': len(model.pathologies),
        'architecture': type(model).__name__
    }
    return info