import numpy as np
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
from typing import Dict, Tuple
import pandas as pd


def compute_confusion_matrix_metrics(y_true, y_pred, threshold=0.5) -> Dict:
    """
    Compute confusion matrix and all related metrics.
    
    Args:
        y_true: Ground truth binary labels (0 or 1)
        y_pred: Predicted probabilities (0.0 to 1.0)
        threshold: Classification threshold (default 0.5)
    
    Returns:
        Dictionary with confusion matrix and metrics
    """
    # Convert to numpy arrays
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred)
    
    # Apply threshold to get binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    
    # Handle edge cases (e.g., all predictions are one class)
    if cm.shape == (1, 1):
        # Only one class present
        if y_true[0] == 0:
            tn, fp, fn, tp = cm[0, 0], 0, 0, 0
        else:
            tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
    else:
        tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    total = tn + fp + fn + tp
    
    # Sensitivity (Recall, True Positive Rate)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Precision (Positive Predictive Value)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Negative Predictive Value
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    # Accuracy
    accuracy = (tp + tn) / total if total > 0 else 0.0
    
    # F1 Score
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    
    # False Positive Rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # False Negative Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    return {
        'confusion_matrix': cm,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'total': int(total),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'precision': float(precision),
        'npv': float(npv),
        'accuracy': float(accuracy),
        'f1_score': float(f1_score),
        'fpr': float(fpr),
        'fnr': float(fnr),
        'threshold': float(threshold)
    }


def plot_confusion_matrix_heatmap(cm, title="Confusion Matrix", labels=['Negative', 'Positive']):
    """
    Create an interactive confusion matrix heatmap using Plotly.
    
    Args:
        cm: Confusion matrix (2x2 numpy array)
        title: Plot title
        labels: Class labels
    
    Returns:
        Plotly figure object
    """
    # Normalize for percentages
    cm_sum = cm.sum()
    cm_normalized = cm.astype('float') / cm_sum if cm_sum > 0 else cm
    
    # Create text annotations
    annotations = []
    
    # Handle different matrix shapes
    if cm.shape == (2, 2):
        for i in range(2):
            for j in range(2):
                count = cm[i, j]
                percentage = cm_normalized[i, j] * 100
                annotations.append(
                    dict(
                        x=j,
                        y=i,
                        text=f'<b>{count}</b><br>({percentage:.1f}%)',
                        font=dict(
                            size=16, 
                            color='white' if cm_normalized[i, j] > 0.5 else '#333'
                        ),
                        showarrow=False
                    )
                )
    else:
        # Single class case
        annotations.append(
            dict(
                x=0,
                y=0,
                text=f'<b>{cm[0, 0]}</b><br>(100%)',
                font=dict(size=16, color='#333'),
                showarrow=False
            )
        )
    
    # Create heatmap with FIXED colorbar configuration
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        showscale=True,
        hovertemplate='Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>',
        colorbar=dict(
            title=dict(
                text="Count",
                side="right"
            ),
            tickmode="linear",
            tick0=0,
            dtick=max(1, cm.max() // 5)
        )
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#333'}
        },
        xaxis=dict(
            title='Predicted Label',
            title_font=dict(size=14, color='#333'),
            tickfont=dict(size=12, color='#333'),
            side='bottom'
        ),
        yaxis=dict(
            title='True Label',
            title_font=dict(size=14, color='#333'),
            tickfont=dict(size=12, color='#333'),
            autorange='reversed'
        ),
        width=600,
        height=500,
        annotations=annotations,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=80, r=80, t=80, b=80)
    )
    
    return fig


def compare_thresholds(y_true, y_pred, thresholds=None) -> pd.DataFrame:
    """
    Compare metrics across different classification thresholds.
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted probabilities
        thresholds: List of thresholds to test (default: 0.1 to 0.9 in 0.1 steps)
    
    Returns:
        DataFrame with metrics for each threshold
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)
    
    results = []
    
    for threshold in thresholds:
        metrics = compute_confusion_matrix_metrics(y_true, y_pred, threshold)
        results.append({
            'threshold': threshold,
            'accuracy': metrics['accuracy'],
            'sensitivity': metrics['sensitivity'],
            'specificity': metrics['specificity'],
            'precision': metrics['precision'],
            'f1_score': metrics['f1_score']
        })
    
    return pd.DataFrame(results)


def plot_threshold_comparison(y_true, y_pred):
    """
    Plot how metrics change with different thresholds.
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted probabilities
    
    Returns:
        Plotly figure object
    """
    thresholds = np.arange(0.05, 0.96, 0.05)
    df = compare_thresholds(y_true, y_pred, thresholds)
    
    fig = go.Figure()
    
    # Add traces for each metric
    metrics_to_plot = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for metric, color in zip(metrics_to_plot, colors):
        fig.add_trace(go.Scatter(
            x=df['threshold'],
            y=df[metric],
            mode='lines+markers',
            name=metric.replace('_', ' ').title(),
            line=dict(color=color, width=2),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title={
            'text': 'Metrics vs Classification Threshold',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#333'}
        },
        xaxis=dict(
            title='Threshold',
            range=[0, 1],
            gridcolor='#e0e0e0',
            title_font=dict(size=14, color='#333'),
            tickfont=dict(size=12, color='#333')
        ),
        yaxis=dict(
            title='Metric Value',
            range=[0, 1],
            gridcolor='#e0e0e0',
            title_font=dict(size=14, color='#333'),
            tickfont=dict(size=12, color='#333')
        ),
        width=None,
        height=500,
        showlegend=True,
        legend=dict(
            x=0.98,
            y=0.98,
            xanchor='right',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='#ccc',
            borderwidth=1,
            font=dict(size=11, color='#333')
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified',
        margin=dict(l=60, r=20, t=60, b=60)
    )
    
    return fig


def find_optimal_threshold(y_true, y_pred, method='youden') -> Tuple[float, Dict]:
    """
    Find optimal classification threshold.
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted probabilities
        method: Method to use ('youden', 'f1', 'accuracy')
    
    Returns:
        Tuple of (optimal_threshold, metrics_at_threshold)
    """
    thresholds = np.arange(0.01, 1.0, 0.01)
    best_score = -1
    optimal_threshold = 0.5
    
    for threshold in thresholds:
        metrics = compute_confusion_matrix_metrics(y_true, y_pred, threshold)
        
        if method == 'youden':
            # Youden's J statistic: sensitivity + specificity - 1
            score = metrics['sensitivity'] + metrics['specificity'] - 1
        elif method == 'f1':
            score = metrics['f1_score']
        elif method == 'accuracy':
            score = metrics['accuracy']
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if score > best_score:
            best_score = score
            optimal_threshold = threshold
    
    # Get final metrics at optimal threshold
    final_metrics = compute_confusion_matrix_metrics(y_true, y_pred, optimal_threshold)
    
    return optimal_threshold, final_metrics


def compute_classification_report(y_true, y_pred, threshold=0.5) -> str:
    """
    Generate a text classification report.
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted probabilities
        threshold: Classification threshold
    
    Returns:
        Formatted string report
    """
    metrics = compute_confusion_matrix_metrics(y_true, y_pred, threshold)
    
    report = []
    report.append("=" * 60)
    report.append("CLASSIFICATION REPORT")
    report.append("=" * 60)
    report.append(f"\nThreshold: {threshold:.3f}")
    report.append(f"Total Samples: {metrics['total']}")
    report.append("\n" + "-" * 60)
    report.append("CONFUSION MATRIX")
    report.append("-" * 60)
    report.append(f"True Positives:  {metrics['true_positives']:>5}")
    report.append(f"True Negatives:  {metrics['true_negatives']:>5}")
    report.append(f"False Positives: {metrics['false_positives']:>5}")
    report.append(f"False Negatives: {metrics['false_negatives']:>5}")
    report.append("\n" + "-" * 60)
    report.append("PERFORMANCE METRICS")
    report.append("-" * 60)
    report.append(f"Accuracy:    {metrics['accuracy']:.4f}")
    report.append(f"Sensitivity: {metrics['sensitivity']:.4f}  (True Positive Rate)")
    report.append(f"Specificity: {metrics['specificity']:.4f}  (True Negative Rate)")
    report.append(f"Precision:   {metrics['precision']:.4f}  (Positive Predictive Value)")
    report.append(f"NPV:         {metrics['npv']:.4f}  (Negative Predictive Value)")
    report.append(f"F1 Score:    {metrics['f1_score']:.4f}")
    report.append(f"FPR:         {metrics['fpr']:.4f}  (False Positive Rate)")
    report.append(f"FNR:         {metrics['fnr']:.4f}  (False Negative Rate)")
    report.append("=" * 60)
    
    return "\n".join(report)