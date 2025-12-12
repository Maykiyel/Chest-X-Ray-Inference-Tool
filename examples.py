"""
Example usage scripts for the Chest X-Ray Inference Tool

These examples demonstrate how to use the core modules programmatically
without the Streamlit interface.
"""

import torch
from pathlib import Path
import pandas as pd
from datetime import datetime

from inference import load_models, predict_single_image, predict_batch
from utils import get_image_paths, save_results_to_csv, create_summary_report
from metrics import compute_roc_auc, plot_roc_curve, compute_classification_metrics


# ==============================================================================
# EXAMPLE 1: Single Image Inference
# ==============================================================================

def example_single_image_inference():
    """Example: Analyze a single X-ray image."""
    
    print("=" * 60)
    print("EXAMPLE 1: Single Image Inference")
    print("=" * 60)
    
    # Configuration
    image_path = "path/to/your/xray.png"  # Replace with actual path
    model_names = ['nih', 'mimic']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nDevice: {device}")
    print(f"Models: {model_names}")
    print(f"Image: {image_path}\n")
    
    # Load models
    print("Loading models...")
    models = load_models(model_names, device)
    print(f"✓ Loaded {len(models)} model(s)\n")
    
    # Run inference
    print("Running inference...")
    all_results = []
    
    for model_name, model in models.items():
        print(f"  Processing with {model_name}...")
        results = predict_single_image(image_path, model, model_name, device)
        all_results.extend(results)
    
    # Display top predictions
    df = pd.DataFrame(all_results)
    print("\n" + "=" * 60)
    print("TOP 10 PREDICTIONS")
    print("=" * 60)
    top_10 = df.nlargest(10, 'probability')
    
    for idx, row in top_10.iterrows():
        print(f"{row['model']:10s} | {row['pathology']:20s} | {row['probability']:.4f}")
    
    print("\n✓ Complete!\n")
    
    return all_results


# ==============================================================================
# EXAMPLE 2: Batch Processing
# ==============================================================================

def example_batch_processing():
    """Example: Process a folder of X-ray images."""
    
    print("=" * 60)
    print("EXAMPLE 2: Batch Processing")
    print("=" * 60)
    
    # Configuration
    folder_path = Path("path/to/xray/folder")  # Replace with actual path
    model_names = ['nih', 'mimic', 'chexpert']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 8
    
    print(f"\nFolder: {folder_path}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}\n")
    
    # Get image paths
    image_paths = get_image_paths(folder_path, recursive=True)
    print(f"Found {len(image_paths)} images\n")
    
    if len(image_paths) == 0:
        print("⚠ No images found!")
        return
    
    # Load models
    print("Loading models...")
    models = load_models(model_names, device)
    print(f"✓ Loaded {len(models)} model(s)\n")
    
    # Process batch
    print("Processing images...")
    all_results = []
    
    for model_name, model in models.items():
        print(f"\n  Processing with {model_name}...")
        
        def progress_callback(p):
            bar_length = 40
            filled = int(bar_length * p)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"\r  [{bar}] {p*100:.1f}%", end='', flush=True)
        
        results = predict_batch(
            image_paths,
            model,
            model_name,
            device,
            batch_size=batch_size,
            auto_label=True,
            progress_callback=progress_callback
        )
        
        all_results.extend(results)
        print()  # New line after progress bar
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f'batch_results_{timestamp}.csv'
    save_results_to_csv(all_results, csv_path)
    print(f"\n✓ Results saved to: {csv_path}")
    
    # Create summary report
    summary = create_summary_report(all_results)
    print("\n" + summary)
    
    return all_results


# ==============================================================================
# EXAMPLE 3: ROC/AUC Analysis
# ==============================================================================

def example_roc_analysis():
    """Example: Compute and plot ROC curves."""
    
    print("=" * 60)
    print("EXAMPLE 3: ROC/AUC Analysis")
    print("=" * 60)
    
    # Load results from a previous batch run
    # This assumes you have a CSV with predictions and ground truth labels
    csv_path = "batch_results.csv"  # Replace with actual path
    
    print(f"\nLoading results from: {csv_path}\n")
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"⚠ File not found: {csv_path}")
        print("Run example_batch_processing() first to generate results.\n")
        return
    
    # Check if ground truth labels exist
    if 'ground_truth' not in df.columns:
        print("⚠ No ground truth labels found in the data.")
        print("Make sure your images are organized in labeled folders.\n")
        return
    
    # Filter for a specific model and pathology
    model_name = 'nih'
    pathology_name = 'Pneumonia'
    
    roc_data = df[
        (df['model'] == model_name) & 
        (df['pathology'] == pathology_name) &
        (df['ground_truth'].notna())
    ]
    
    if len(roc_data) < 10:
        print(f"⚠ Not enough data for {pathology_name} (need at least 10 samples)")
        print(f"Found only {len(roc_data)} samples.\n")
        return
    
    print(f"Model: {model_name}")
    print(f"Pathology: {pathology_name}")
    print(f"Samples: {len(roc_data)}\n")
    
    # Compute ROC/AUC
    y_true = roc_data['ground_truth'].values
    y_pred = roc_data['probability'].values
    
    fpr, tpr, thresholds, auc_score = compute_roc_auc(y_true, y_pred)
    
    print("ROC/AUC Results:")
    print(f"  AUC Score: {auc_score:.4f}")
    print(f"  Number of thresholds: {len(thresholds)}")
    
    # Compute metrics at default threshold (0.5)
    metrics = compute_classification_metrics(y_true, y_pred, threshold=0.5)
    
    print("\nClassification Metrics (threshold=0.5):")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    
    print("\nConfusion Matrix:")
    print(f"  True Positives:  {metrics['true_positives']}")
    print(f"  True Negatives:  {metrics['true_negatives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    
    # Create ROC plot
    fig = plot_roc_curve(fpr, tpr, auc_score, f"{model_name.upper()} - {pathology_name}")
    
    # Save plot
    plot_path = f"roc_curve_{model_name}_{pathology_name}.html"
    fig.write_html(plot_path)
    print(f"\n✓ ROC curve saved to: {plot_path}\n")
    
    return {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'auc': auc_score}


# ==============================================================================
# EXAMPLE 4: Compare Multiple Models
# ==============================================================================

def example_model_comparison():
    """Example: Compare predictions from multiple models."""
    
    print("=" * 60)
    print("EXAMPLE 4: Model Comparison")
    print("=" * 60)
    
    # Load results
    csv_path = "batch_results.csv"  # Replace with actual path
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"⚠ File not found: {csv_path}\n")
        return
    
    # Compare average probabilities across models
    print("\nAverage Probabilities by Model:")
    print("-" * 60)
    
    model_avg = df.groupby('model')['probability'].agg(['mean', 'std', 'min', 'max'])
    print(model_avg)
    
    # Compare top pathologies for each model
    print("\n\nTop 5 Pathologies by Model:")
    print("-" * 60)
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        top_pathologies = model_data.groupby('pathology')['probability'].mean().nlargest(5)
        
        print(f"\n{model.upper()}:")
        for pathology, prob in top_pathologies.items():
            print(f"  {pathology:25s}: {prob:.4f}")
    
    # Agreement between models
    print("\n\nModel Agreement Analysis:")
    print("-" * 60)
    
    # Get predictions for the same images
    pivot = df.pivot_table(
        index=['filename', 'pathology'],
        columns='model',
        values='probability'
    )
    
    # Compute correlation between models
    if len(pivot.columns) > 1:
        correlations = pivot.corr()
        print("\nCorrelation Matrix:")
        print(correlations)
        print("\nInterpretation: Higher correlation = models agree more")
    
    print("\n✓ Complete!\n")


# ==============================================================================
# EXAMPLE 5: Custom Threshold Analysis
# ==============================================================================

def example_threshold_analysis():
    """Example: Find optimal classification threshold."""
    
    print("=" * 60)
    print("EXAMPLE 5: Threshold Analysis")
    print("=" * 60)
    
    # Load results with ground truth
    csv_path = "batch_results.csv"  # Replace with actual path
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"⚠ File not found: {csv_path}\n")
        return
    
    if 'ground_truth' not in df.columns:
        print("⚠ No ground truth labels found.\n")
        return
    
    # Select data
    model_name = 'nih'
    pathology_name = 'Pneumonia'
    
    data = df[
        (df['model'] == model_name) & 
        (df['pathology'] == pathology_name) &
        (df['ground_truth'].notna())
    ]
    
    if len(data) < 10:
        print(f"⚠ Not enough data (need at least 10 samples)\n")
        return
    
    print(f"Model: {model_name}")
    print(f"Pathology: {pathology_name}")
    print(f"Samples: {len(data)}\n")
    
    y_true = data['ground_truth'].values
    y_pred = data['probability'].values
    
    # Test different thresholds
    thresholds_to_test = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    print("Threshold Analysis:")
    print("-" * 80)
    print(f"{'Threshold':>10} {'Sensitivity':>12} {'Specificity':>12} {'Precision':>12} {'F1 Score':>12}")
    print("-" * 80)
    
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds_to_test:
        metrics = compute_classification_metrics(y_true, y_pred, threshold)
        
        print(f"{threshold:10.2f} {metrics['sensitivity']:12.4f} {metrics['specificity']:12.4f} "
              f"{metrics['precision']:12.4f} {metrics['f1_score']:12.4f}")
        
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_threshold = threshold
    
    print("-" * 80)
    print(f"\n✓ Best threshold: {best_threshold:.2f} (F1={best_f1:.4f})\n")
    
    return best_threshold


# ==============================================================================
# MAIN: Run All Examples
# ==============================================================================

def run_all_examples():
    """Run all example scripts."""
    
    print("\n" + "=" * 60)
    print(" CHEST X-RAY INFERENCE TOOL - EXAMPLE SCRIPTS")
    print("=" * 60 + "\n")
    
    examples = [
        ("Single Image Inference", example_single_image_inference),
        ("Batch Processing", example_batch_processing),
        ("ROC/AUC Analysis", example_roc_analysis),
        ("Model Comparison", example_model_comparison),
        ("Threshold Analysis", example_threshold_analysis)
    ]
    
    for i, (name, func) in enumerate(examples, 1):
        print(f"\n[{i}/{len(examples)}] {name}")
        print("=" * 60)
        
        try:
            func()
        except Exception as e:
            print(f"\n⚠ Error: {str(e)}\n")
        
        if i < len(examples):
            input("\nPress Enter to continue to next example...")
    
    print("\n" + "=" * 60)
    print(" ALL EXAMPLES COMPLETE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Uncomment the example you want to run:
    
    # example_single_image_inference()
    # example_batch_processing()
    # example_roc_analysis()
    # example_model_comparison()
    # example_threshold_analysis()
    
    # Or run all examples:
    # run_all_examples()
    
    print("\nTo run an example, uncomment the corresponding line in the __main__ block.")
    print("Example: example_single_image_inference()\n")