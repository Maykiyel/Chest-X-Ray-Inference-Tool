import pandas as pd
from pathlib import Path
import re
from typing import List, Dict, Optional

# Supported image extensions
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.dcm', '.dicom'}


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


def extract_folder_label(img_path: Path) -> Optional[Dict[str, int]]:
    """
    Extract ground truth label from folder name.
    
    Supports patterns like:
    - pneumonia_positive / pneumonia_negative
    - Pneumonia_1 / Pneumonia_0
    - positive_pneumonia / negative_pneumonia
    - pneumonia/positive, pneumonia/negative
    - normal / Normal (for normal X-rays)
    
    Args:
        img_path: Path to the image file
    
    Returns:
        Dictionary mapping pathology names to binary labels (0 or 1)
        Returns None if no label pattern is found
    """
    # Get folder name (parent directory)
    folder_name = img_path.parent.name.lower()
    
    # Check for "normal" folders
    if 'normal' in folder_name and 'abnormal' not in folder_name:
        return {'Normal': 1}
    
    # Common pathology names (adjust as needed)
    pathologies = [
        'atelectasis', 'cardiomegaly', 'consolidation', 'edema',
        'effusion', 'emphysema', 'fibrosis', 'hernia', 'infiltration',
        'mass', 'nodule', 'pleural_thickening', 'pneumonia', 
        'pneumothorax', 'covid', 'fracture'
    ]
    
    labels = {}
    
    # Pattern 1: pathology_positive / pathology_negative
    for pathology in pathologies:
        if f'{pathology}_positive' in folder_name or f'positive_{pathology}' in folder_name:
            labels[pathology.capitalize()] = 1
            return labels
        elif f'{pathology}_negative' in folder_name or f'negative_{pathology}' in folder_name:
            labels[pathology.capitalize()] = 0
            return labels
    
    # Pattern 2: pathology_1 / pathology_0
    for pathology in pathologies:
        if f'{pathology}_1' in folder_name or f'{pathology}1' in folder_name:
            labels[pathology.capitalize()] = 1
            return labels
        elif f'{pathology}_0' in folder_name or f'{pathology}0' in folder_name:
            labels[pathology.capitalize()] = 0
            return labels
    
    # Pattern 3: Just "positive" or "negative"
    if 'positive' in folder_name and 'negative' not in folder_name:
        # Try to infer pathology from parent folder
        parent_folder = img_path.parent.parent.name.lower()
        for pathology in pathologies:
            if pathology in parent_folder:
                labels[pathology.capitalize()] = 1
                return labels
    elif 'negative' in folder_name:
        parent_folder = img_path.parent.parent.name.lower()
        for pathology in pathologies:
            if pathology in parent_folder:
                labels[pathology.capitalize()] = 0
                return labels
    
    # Pattern 4: Check for pathology name with numeric suffix
    match = re.search(r'(\w+)[_-]?([01])', folder_name)
    if match:
        potential_pathology = match.group(1)
        label_value = int(match.group(2))
        
        for pathology in pathologies:
            if pathology in potential_pathology:
                labels[pathology.capitalize()] = label_value
                return labels
    
    # No pattern found
    return None


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