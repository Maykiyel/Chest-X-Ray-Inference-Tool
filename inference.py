import torch
import torchxrayvision as xrv
import numpy as np
from PIL import Image
import streamlit as st
from pathlib import Path
from datetime import datetime
import torchvision.transforms
from torch.utils.data import Dataset, DataLoader
from utils import extract_folder_label
import gc


def resolve_ground_truth_for_pathology(ground_truth: dict, pathology: str):
    """Resolve per-pathology GT and infer binary negatives where appropriate."""
    if ground_truth is None:
        return None
    if pathology in ground_truth:
        return ground_truth.get(pathology)

    if ground_truth.get('Normal') == 1 and pathology != 'Normal':
        return 0

    positive_non_normal = [k for k, v in ground_truth.items() if k != 'Normal' and v == 1]
    if len(positive_non_normal) == 1 and pathology != positive_non_normal[0] and pathology != 'Normal':
        return 0

    return None

def load_models(model_names, device='cpu'):
    """
    Load TorchXRayVision models using official API.
    
    Args:
        model_names: List of model names ('nih', 'mimic', 'chexpert')
        device: 'cuda' or 'cpu'
    
    Returns:
        Dictionary of {model_name: model}
    """
    # Clear any existing models and cache
    if device == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    models = {}
    
    # Mapping of our names to official weight names
    weight_map = {
        'nih': 'densenet121-res224-nih',
        'mimic': 'densenet121-res224-mimic_nb',
        'chexpert': 'densenet121-res224-chex'
    }
    
    for name in model_names:
        try:
            if name not in weight_map:
                continue
            
            # Use official model getter
            model = xrv.models.get_model(weight_map[name])
            
            model = model.to(device)
            model.eval()
            
            # Disable gradient computation
            for param in model.parameters():
                param.requires_grad = False
            
            models[name] = model
            st.success(f"✓ Loaded {name.upper()} model ({weight_map[name]})")
            
        except Exception as e:
            st.error(f"Failed to load {name} model: {str(e)}")
    
    return models


def preprocess_image(img_path, target_size=224):
    """
    Preprocess image using OFFICIAL TorchXRayVision pipeline.
    Follows the exact same process as the official example.
    
    Args:
        img_path: Path to image file
        target_size: Target size for resizing (default 224)
    
    Returns:
        Preprocessed image tensor with shape [1, H, W] (no channel dimension)
    """
    # Use official XRV image loader - handles DICOM and regular images correctly
    img = xrv.utils.load_image(str(img_path))
    
    # Use official preprocessing transforms
    # These are specifically designed for chest X-rays
    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(target_size)
    ])
    
    img = transform(img)
    
    # Convert to tensor - EXACTLY as in official example
    # Shape after transform: [H, W]
    # Shape after unsqueeze: [1, H, W] (batch dimension only, NO channel dimension)
    img_tensor = torch.from_numpy(img).unsqueeze(0)
    
    return img_tensor


def predict_single_image(img_path, model, model_name, device='cpu'):
    """
    Run inference on a single image using OFFICIAL pipeline.
    Follows the exact process from the official TorchXRayVision example.
    
    Args:
        img_path: Path to image
        model: Loaded model
        model_name: Name of the model
        device: Device to use
    
    Returns:
        List of prediction dictionaries
    """
    # Force model to eval mode
    model.eval()
    
    # Preprocess using official pipeline
    img_tensor = preprocess_image(img_path).to(device)
    
    # Clear any cached computations
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    # Inference with no gradient - EXACTLY as in official example
    with torch.no_grad():
        # Run inference - returns RAW LOGITS
        outputs = model(img_tensor)
        
        # Move to CPU immediately
        outputs = outputs.cpu()
        
        # Extract predictions (raw logits, not probabilities yet)
        predictions = outputs[0].detach().numpy()
    
    # Immediate cleanup
    del img_tensor, outputs
    if device == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    # Get pathology names from official source
    pathology_names = xrv.datasets.default_pathologies
    
    # Format results - IMPORTANT: These are LOGITS, not probabilities
    # Higher values = higher likelihood, but not bounded 0-1
    # For display/threshold purposes, we'll apply sigmoid to get probabilities
    results = []
    
    # Convert logits to probabilities using sigmoid
    probs = 1.0 / (1.0 + np.exp(-predictions))  # Sigmoid function
    
    # Add pathology predictions
    for pathology, logit, prob in zip(pathology_names, predictions, probs):
        results.append({
            'filename': Path(img_path).name,
            'filepath': str(img_path),
            'model': model_name,
            'pathology': pathology,
            'logit': float(logit),  # Raw model output
            'probability': float(prob),  # Sigmoid-transformed probability
            'timestamp': datetime.now().isoformat()
        })
    
    # Calculate "Normal" probability
    # Normal = low probability of ALL pathologies
    max_pathology_prob = float(np.max(probs))
    normal_prob = 1.0 - max_pathology_prob
    
    results.append({
        'filename': Path(img_path).name,
        'filepath': str(img_path),
        'model': model_name,
        'pathology': 'Normal',
        'logit': float(-np.log(max_pathology_prob / (1 - max_pathology_prob + 1e-10))),  # Inverse sigmoid
        'probability': normal_prob,
        'timestamp': datetime.now().isoformat()
    })
    
    return results


class XRayDataset(Dataset):
    """
    Custom dataset for batch processing using official preprocessing.
    Follows the exact same preprocessing as the official TorchXRayVision example.
    """
    
    def __init__(self, image_paths, target_size=224):
        self.image_paths = image_paths
        self.target_size = target_size
        
        # Official transforms - EXACTLY as in official example
        self.transform = torchvision.transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(target_size)
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            # Use official loader and transforms - EXACTLY as in official example
            img = xrv.utils.load_image(str(img_path))
            img = self.transform(img)
            
            # Convert to tensor - EXACTLY as in official example
            # Shape after transform: [H, W]
            # Only ONE unsqueeze (batch dim added by DataLoader)
            img_tensor = torch.from_numpy(img)
            
            return img_tensor, str(img_path), True
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            # Return a valid tensor with correct shape [H, W]
            return torch.zeros(self.target_size, self.target_size), str(img_path), False


def predict_batch(image_paths, model, model_name, device='cpu', 
                 batch_size=8, auto_label=True, progress_callback=None):
    """
    Run batch inference using OFFICIAL pipeline.
    Follows the exact process from the official TorchXRayVision example.
    
    Args:
        image_paths: List of image paths
        model: Loaded model
        model_name: Name of the model
        device: Device to use
        batch_size: Batch size for inference
        auto_label: Whether to extract labels from folder/filename
        progress_callback: Callback function for progress updates
    
    Returns:
        List of prediction dictionaries
    """
    # Force model to eval mode
    model.eval()
    
    # Clear cache before batch
    if device == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    # Create dataset and dataloader
    dataset = XRayDataset(image_paths)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == 'cuda')
    )
    
    results = []
    total_batches = len(dataloader)
    
    # Get pathology names
    pathology_names = xrv.datasets.default_pathologies
    
    # Process batches
    for batch_idx, (images, paths, valid_flags) in enumerate(dataloader):
        # Filter valid images
        valid_indices = [i for i, flag in enumerate(valid_flags) if flag]
        
        if len(valid_indices) == 0:
            continue
        
        valid_images = images[valid_indices].to(device)
        valid_paths = [paths[i] for i in valid_indices]
        
        # Inference with no gradient - EXACTLY as in official example
        with torch.no_grad():
            # Clear cache
            if device == 'cuda':
                torch.cuda.empty_cache()
            
            # Run inference - returns RAW LOGITS
            outputs = model(valid_images)
            
            # Move to CPU
            outputs = outputs.cpu()
            
            # Get predictions (logits)
            predictions = outputs.detach().numpy()
        
        # Immediate cleanup
        del valid_images, outputs
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        # Process predictions
        for i, (pred, img_path) in enumerate(zip(predictions, valid_paths)):
            img_path_obj = Path(img_path)
            
            # Extract ground truth label from filename or folder
            ground_truth = None
            if auto_label:
                ground_truth = extract_folder_label(img_path_obj)
            
            # Convert logits to probabilities
            probs = 1.0 / (1.0 + np.exp(-pred))  # Sigmoid
            
            # Add pathology predictions
            for pathology, logit, prob in zip(pathology_names, pred, probs):
                result_dict = {
                    'filename': img_path_obj.name,
                    'filepath': str(img_path),
                    'model': model_name,
                    'pathology': pathology,
                    'logit': float(logit),
                    'probability': float(prob),
                    'timestamp': datetime.now().isoformat()
                }
                
                if ground_truth is not None:
                    result_dict['ground_truth'] = resolve_ground_truth_for_pathology(ground_truth, pathology)
                
                results.append(result_dict)
            
            # Calculate and add "Normal" probability
            max_pathology_prob = float(np.max(probs))
            normal_prob = 1.0 - max_pathology_prob
            
            normal_dict = {
                'filename': img_path_obj.name,
                'filepath': str(img_path),
                'model': model_name,
                'pathology': 'Normal',
                'logit': float(-np.log(max_pathology_prob / (1 - max_pathology_prob + 1e-10))),
                'probability': normal_prob,
                'timestamp': datetime.now().isoformat()
            }
            
            if ground_truth is not None:
                normal_dict['ground_truth'] = resolve_ground_truth_for_pathology(ground_truth, 'Normal')
            
            results.append(normal_dict)
        
        # Update progress
        if progress_callback:
            progress = (batch_idx + 1) / total_batches
            progress_callback(progress)
        
        # Cleanup after each batch
        gc.collect()
    
    # Final cleanup
    if device == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    return results


def get_model_info(model):
    """Get information about a model."""
    info = {
        'pathologies': xrv.datasets.default_pathologies,
        'num_pathologies': len(xrv.datasets.default_pathologies),
        'architecture': type(model).__name__
    }
    return info
