import pandas as pd
from pathlib import Path
import re
from typing import List, Dict, Optional

# Supported image extensions
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.dcm', '.dicom'}


PATHOLOGY_NAME_MAP = {
    'atelectasis': 'Atelectasis',
    'cardiomegaly': 'Cardiomegaly',
    'consolidation': 'Consolidation',
    'edema': 'Edema',
    'effusion': 'Effusion',
    'emphysema': 'Emphysema',
    'fibrosis': 'Fibrosis',
    'hernia': 'Hernia',
    'infiltration': 'Infiltration',
    'mass': 'Mass',
    'nodule': 'Nodule',
    'pleural_thickening': 'Pleural_Thickening',
    'pneumonia': 'Pneumonia',
    'pneumothorax': 'Pneumothorax',
    'covid': 'Covid',
    'fracture': 'Fracture',
    'normal': 'Normal',
}


def extract_label_from_filename(filename: str) -> Optional[Dict[str, int]]:
    """
    Extract pathology + binary label from filename when present.

    Supported examples:
    - atelectasis_1.jpeg -> {'Atelectasis': 1}
    - atelectasis-0.png -> {'Atelectasis': 0}
    - positive_pneumonia_22.jpg -> {'Pneumonia': 1}
    - nodule_negative_case5.png -> {'Nodule': 0}

    If only pathology is present (no explicit 0/1 or positive/negative marker),
    this defaults to positive (1) for backward compatibility.
    """
    normalized = re.sub(r'[^a-z0-9]+', '_', filename.lower()).strip('_')
    if not normalized:
        return None

    for pathology_key, canonical_name in PATHOLOGY_NAME_MAP.items():
        pathology_pattern = pathology_key.replace('_', '[_]?')

        negative_pattern = rf'(?:^|_)(?:negative|neg|0)_?{pathology_pattern}(?:_|$)|(?:^|_){pathology_pattern}_?(?:negative|neg|0)(?:_|$)'
        positive_pattern = rf'(?:^|_)(?:positive|pos|1)_?{pathology_pattern}(?:_|$)|(?:^|_){pathology_pattern}_?(?:positive|pos|1)(?:_|$)'
        pathology_only_pattern = rf'(?:^|_){pathology_pattern}(?:_|$)'

        if re.search(negative_pattern, normalized):
            return {canonical_name: 0}
        if re.search(positive_pattern, normalized):
            return {canonical_name: 1}
        if re.search(pathology_only_pattern, normalized):
            return {canonical_name: 1}

    return None


def get_image_paths(folder_path: Path, recursive: bool = True) -> List[Path]:
    """
    Get all image paths from a folder.
    
    Args:
        folder_path: Path to the folder
        recursive: Whether to search recursively
    
    Returns:
        List of image paths
    """
    image_paths = []
    
    if recursive:
        for ext in IMAGE_EXTENSIONS:
            image_paths.extend(folder_path.rglob(f'*{ext}'))
    else:
        for ext in IMAGE_EXTENSIONS:
            image_paths.extend(folder_path.glob(f'*{ext}'))
    
    return sorted(image_paths)


def extract_pathology_from_filename(filename: str) -> Optional[str]:
    """
    Extract pathology name from filename.
    
    Supports patterns like:
    - Pneumonia_img1.png
    - Effusion_img23.png
    - Atelectasis_img456.png
    
    Args:
        filename: Filename to parse
    
    Returns:
        Pathology name (capitalized) or None
    """
    label = extract_label_from_filename(filename)
    if label:
        return next(iter(label.keys()))

    return None


def extract_folder_label(img_path: Path) -> Optional[Dict[str, int]]:
    """
    Extract ground truth label from BOTH folder name AND filename.
    
    Supports multiple labeling methods:
    
    FILENAME-BASED (Priority 1):
    - Pneumonia_img1.png → Pneumonia=1 (positive)
    - Effusion_img23.png → Effusion=1 (positive)
    
    FOLDER-BASED (Priority 2):
    - pneumonia_positive/img1.png → Pneumonia=1
    - pneumonia_negative/img2.png → Pneumonia=0
    - pneumonia_1/img3.png → Pneumonia=1
    - pneumonia_0/img4.png → Pneumonia=0
    - pneumonia/positive/img5.png → Pneumonia=1
    - pneumonia/negative/img6.png → Pneumonia=0
    - normal/img7.png → Normal=1
    
    Args:
        img_path: Path to the image file
    
    Returns:
        Dictionary mapping pathology names to binary labels (0 or 1)
        Returns None if no label pattern is found
    """
    # PRIORITY 1: Check filename first
    filename = img_path.stem  # Without extension
    label_from_filename = extract_label_from_filename(filename)
    if label_from_filename:
        return label_from_filename
    
    # PRIORITY 2: Check immediate folder name
    folder_name = img_path.parent.name.lower()
    
    # Check for "normal" folders
    if 'normal' in folder_name and 'abnormal' not in folder_name:
        return {'Normal': 1}
    
    # Common pathology keys (lowercase)
    pathologies = [k for k in PATHOLOGY_NAME_MAP.keys() if k != 'normal']
    
    labels = {}
    
    # Pattern 1: pathology_positive / pathology_negative
    for pathology in pathologies:
        if f'{pathology}_positive' in folder_name or f'positive_{pathology}' in folder_name:
            labels[PATHOLOGY_NAME_MAP[pathology]] = 1
            return labels
        elif f'{pathology}_negative' in folder_name or f'negative_{pathology}' in folder_name:
            labels[PATHOLOGY_NAME_MAP[pathology]] = 0
            return labels
    
    # Pattern 2: pathology_1 / pathology_0
    for pathology in pathologies:
        if f'{pathology}_1' in folder_name or f'{pathology}1' in folder_name:
            labels[PATHOLOGY_NAME_MAP[pathology]] = 1
            return labels
        elif f'{pathology}_0' in folder_name or f'{pathology}0' in folder_name:
            labels[PATHOLOGY_NAME_MAP[pathology]] = 0
            return labels
    
    # Pattern 3: Just "positive" or "negative"
    if 'positive' in folder_name and 'negative' not in folder_name:
        # Try to infer pathology from parent folder
        parent_folder = img_path.parent.parent.name.lower()
        for pathology in pathologies:
            if pathology in parent_folder:
                labels[PATHOLOGY_NAME_MAP[pathology]] = 1
                return labels
        # Also check if parent is just the pathology name
        if parent_folder.replace('_', '').replace('-', '') in [p.replace('_', '') for p in pathologies]:
            for pathology in pathologies:
                if pathology.replace('_', '') == parent_folder.replace('_', '').replace('-', ''):
                    labels[PATHOLOGY_NAME_MAP[pathology]] = 1
                    return labels
    elif 'negative' in folder_name:
        parent_folder = img_path.parent.parent.name.lower()
        for pathology in pathologies:
            if pathology in parent_folder:
                labels[PATHOLOGY_NAME_MAP[pathology]] = 0
                return labels
        # Also check if parent is just the pathology name
        if parent_folder.replace('_', '').replace('-', '') in [p.replace('_', '') for p in pathologies]:
            for pathology in pathologies:
                if pathology.replace('_', '') == parent_folder.replace('_', '').replace('-', ''):
                    labels[PATHOLOGY_NAME_MAP[pathology]] = 0
                    return labels
    
    # Pattern 4: Just pathology name in folder (assume positive)
    for pathology in pathologies:
        if folder_name == pathology or folder_name.replace('_', '').replace('-', '') == pathology.replace('_', ''):
            # Folder is just the pathology name, check subfolder for pos/neg
            # If no subfolder pattern, assume it's positive
            labels[PATHOLOGY_NAME_MAP[pathology]] = 1
            return labels
    
    # Pattern 5: Check for pathology name with numeric suffix
    match = re.search(r'(\w+)[_-]?([01])', folder_name)
    if match:
        potential_pathology = match.group(1)
        label_value = int(match.group(2))
        
        for pathology in pathologies:
            if pathology in potential_pathology:
                labels[PATHOLOGY_NAME_MAP[pathology]] = label_value
                return labels
    
    # No pattern found
    return None


def validate_labels_in_folder(folder_path: Path, recursive: bool = True) -> Dict:
    """
    Validate and show statistics about label detection in a folder.
    
    Args:
        folder_path: Path to the folder
        recursive: Whether to search recursively
    
    Returns:
        Dictionary with label statistics
    """
    if not folder_path.exists() or not folder_path.is_dir():
        return {'valid': False, 'error': 'Invalid folder path'}
    
    images = get_image_paths(folder_path, recursive=recursive)
    
    if len(images) == 0:
        return {'valid': False, 'error': 'No images found'}
    
    # Analyze labels
    label_stats = {
        'total_images': len(images),
        'labeled_count': 0,
        'unlabeled_count': 0,
        'filename_labels': 0,
        'folder_labels': 0,
        'pathology_breakdown': {},
        'labeled_files': [],
        'unlabeled_files': []
    }
    
    for img_path in images:
        # Check filename pattern
        filename_pathology = extract_pathology_from_filename(img_path.stem)
        
        # Check folder pattern
        folder_label = extract_folder_label(img_path)
        
        if folder_label is not None:
            label_stats['labeled_count'] += 1
            
            # Determine source of label
            if filename_pathology:
                label_stats['filename_labels'] += 1
                source = 'filename'
            else:
                label_stats['folder_labels'] += 1
                source = 'folder'
            
            # Track pathology breakdown
            for pathology, value in folder_label.items():
                if pathology not in label_stats['pathology_breakdown']:
                    label_stats['pathology_breakdown'][pathology] = {
                        'positive': 0, 'negative': 0
                    }
                
                if value == 1:
                    label_stats['pathology_breakdown'][pathology]['positive'] += 1
                else:
                    label_stats['pathology_breakdown'][pathology]['negative'] += 1
            
            label_stats['labeled_files'].append({
                'filename': img_path.name,
                'folder': img_path.parent.name,
                'pathology': list(folder_label.keys())[0],
                'label': list(folder_label.values())[0],
                'source': source
            })
        else:
            label_stats['unlabeled_count'] += 1
            label_stats['unlabeled_files'].append({
                'filename': img_path.name,
                'folder': img_path.parent.name
            })
    
    label_stats['valid'] = True
    label_stats['label_percentage'] = (label_stats['labeled_count'] / len(images)) * 100
    
    return label_stats


def save_results_to_csv(results: List[Dict], output_path: str):
    """
    Save prediction results to CSV file.
    
    Args:
        results: List of prediction dictionaries
        output_path: Path to save the CSV file
    """
    df = pd.DataFrame(results)
    
    # Reorder columns for better readability
    column_order = [
        'filename', 'filepath', 'model', 'pathology', 
        'probability', 'timestamp'
    ]
    
    # Add ground_truth column if it exists
    if 'ground_truth' in df.columns:
        column_order.insert(5, 'ground_truth')
    
    # Reorder columns
    existing_columns = [col for col in column_order if col in df.columns]
    df = df[existing_columns]
    
    # Sort by probability descending
    df = df.sort_values('probability', ascending=False)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    return output_path


def format_pathology_name(pathology: str) -> str:
    """
    Format pathology name for display.
    
    Args:
        pathology: Raw pathology name
    
    Returns:
        Formatted pathology name
    """
    # Replace underscores with spaces and capitalize
    formatted = pathology.replace('_', ' ').title()
    return formatted


def aggregate_predictions_by_image(results: List[Dict]) -> pd.DataFrame:
    """
    Aggregate predictions by image and pathology across models.
    
    Args:
        results: List of prediction dictionaries
    
    Returns:
        DataFrame with aggregated predictions
    """
    df = pd.DataFrame(results)
    
    # Group by filename and pathology, aggregate probabilities
    aggregated = df.groupby(['filename', 'pathology']).agg({
        'probability': ['mean', 'std', 'min', 'max'],
        'model': lambda x: ', '.join(sorted(set(x)))
    }).reset_index()
    
    # Flatten column names
    aggregated.columns = [
        'filename', 'pathology', 'prob_mean', 'prob_std', 
        'prob_min', 'prob_max', 'models'
    ]
    
    return aggregated


def get_high_risk_predictions(results: List[Dict], threshold: float = 0.5) -> pd.DataFrame:
    """
    Filter high-risk predictions above a threshold.
    
    Args:
        results: List of prediction dictionaries
        threshold: Probability threshold
    
    Returns:
        DataFrame with high-risk predictions
    """
    df = pd.DataFrame(results)
    high_risk = df[df['probability'] >= threshold].copy()
    high_risk = high_risk.sort_values('probability', ascending=False)
    
    return high_risk


def validate_folder_structure(folder_path: Path) -> Dict:
    """
    Validate folder structure and provide statistics.
    
    Args:
        folder_path: Path to the folder
    
    Returns:
        Dictionary with validation results
    """
    if not folder_path.exists():
        return {'valid': False, 'error': 'Folder does not exist'}
    
    if not folder_path.is_dir():
        return {'valid': False, 'error': 'Path is not a directory'}
    
    # Get image paths
    image_paths = get_image_paths(folder_path, recursive=True)
    
    if len(image_paths) == 0:
        return {
            'valid': False,
            'error': 'No images found in folder'
        }
    
    # Check for labeled subfolders
    has_labels = False
    labeled_images = 0
    
    for img_path in image_paths:
        label = extract_folder_label(img_path)
        if label is not None:
            has_labels = True
            labeled_images += 1
    
    return {
        'valid': True,
        'total_images': len(image_paths),
        'has_labels': has_labels,
        'labeled_images': labeled_images,
        'label_percentage': (labeled_images / len(image_paths)) * 100 if len(image_paths) > 0 else 0
    }


def create_summary_report(results: List[Dict]) -> str:
    """
    Create a text summary report of predictions.
    
    Args:
        results: List of prediction dictionaries
    
    Returns:
        Summary report as string
    """
    df = pd.DataFrame(results)
    
    report = []
    report.append("=" * 60)
    report.append("CHEST X-RAY INFERENCE SUMMARY REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Overall statistics
    report.append("OVERALL STATISTICS")
    report.append("-" * 60)
    report.append(f"Total predictions: {len(df)}")
    report.append(f"Unique images: {df['filename'].nunique()}")
    report.append(f"Models used: {', '.join(df['model'].unique())}")
    report.append(f"Pathologies analyzed: {df['pathology'].nunique()}")
    report.append("")
    
    # Top predictions by model
    report.append("TOP PREDICTIONS BY MODEL")
    report.append("-" * 60)
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        top_5 = model_data.nlargest(5, 'probability')
        
        report.append(f"\n{model.upper()}:")
        for idx, row in top_5.iterrows():
            report.append(f"  {row['filename']}: {row['pathology']} = {row['probability']:.4f}")
    
    report.append("")
    report.append("=" * 60)
    
    return "\n".join(report)