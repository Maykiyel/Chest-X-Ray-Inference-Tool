import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score
import plotly.graph_objects as go
from typing import Tuple, Optional
import pandas as pd


def compute_roc_auc(y_true, y_pred) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute ROC curve and AUC score.
    
    Args:
        y_true: Ground truth binary labels (0 or 1)
        y_pred: Predicted probabilities
    
    Returns:
        Tuple of (fpr, tpr, thresholds, auc_score)
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    
    # Compute AUC score
    auc_score = roc_auc_score(y_true, y_pred)
    
    return fpr, tpr, thresholds, auc_score


def interpolate_roc_curve(fpr, tpr, num_points=100):
    """
    Interpolate ROC curve for smoother visualization.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        num_points: Number of interpolation points
    
    Returns:
        Tuple of (interpolated_fpr, interpolated_tpr)
    """
    # Create evenly spaced points
    mean_fpr = np.linspace(0, 1, num_points)
    
    # Interpolate TPR at these points
    mean_tpr = np.interp(mean_fpr, fpr, tpr)
    
    # Ensure start and end points
    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0
    
    return mean_fpr, mean_tpr


def compute_precision_recall(y_true, y_pred) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute Precision-Recall curve and Average Precision.
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted probabilities
    
    Returns:
        Tuple of (precision, recall, thresholds, avg_precision)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    avg_precision = average_precision_score(y_true, y_pred)
    
    return precision, recall, thresholds, avg_precision


def compute_optimal_threshold(fpr, tpr, thresholds) -> Tuple[float, float, float]:
    """
    Compute optimal threshold using Youden's J statistic.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        thresholds: Thresholds
    
    Returns:
        Tuple of (optimal_threshold, optimal_tpr, optimal_fpr)
    """
    # Youden's J statistic
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    
    optimal_threshold = thresholds[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    
    return optimal_threshold, optimal_tpr, optimal_fpr


def compute_confusion_matrix_at_threshold(y_true, y_pred, threshold=0.5):
    """
    Compute confusion matrix at a given threshold.
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted probabilities
        threshold: Classification threshold
    
    Returns:
        Confusion matrix as 2x2 numpy array [[TN, FP], [FN, TP]]
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    y_pred_binary = (y_pred >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_binary)
    
    return cm


def compute_classification_metrics(y_true, y_pred, threshold=0.5):
    """
    Compute various classification metrics.
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted probabilities
        threshold: Classification threshold
    
    Returns:
        Dictionary with metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    
    # Compute metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative predictive value
    
    metrics = {
        'threshold': threshold,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'npv': npv,
        'accuracy': accuracy,
        'f1_score': f1_score,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }
    
    return metrics


def plot_roc_curve(fpr, tpr, auc_score, title="ROC Curve"):
    """
    Create an interactive ROC curve plot using Plotly with smooth interpolation.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc_score: AUC score
        title: Plot title
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Determine if we should interpolate (for smoother curves with few points)
    if len(fpr) < 50:
        # Interpolate for smoother visualization
        fpr_smooth, tpr_smooth = interpolate_roc_curve(fpr, tpr, num_points=200)
        
        # Plot original points as markers
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='markers',
            name='Actual Thresholds',
            marker=dict(color='#1f77b4', size=8, symbol='circle'),
            hovertemplate='FPR: %{x:.4f}<br>TPR: %{y:.4f}<extra></extra>',
            showlegend=True
        ))
        
        # Plot smooth interpolated line
        fig.add_trace(go.Scatter(
            x=fpr_smooth,
            y=tpr_smooth,
            mode='lines',
            name=f'ROC Curve (AUC = {auc_score:.4f})',
            line=dict(color='#1f77b4', width=3),
            hoverinfo='skip',
            showlegend=True
        ))
    else:
        # For larger datasets, just plot the curve directly
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {auc_score:.4f})',
            line=dict(color='#1f77b4', width=3),
            hovertemplate='FPR: %{x:.4f}<br>TPR: %{y:.4f}<extra></extra>',
            showlegend=True
        ))
    
    # Diagonal reference line (random classifier)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', width=2, dash='dash'),
        hoverinfo='skip',
        showlegend=True
    ))
    
    # Layout
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#333'}
        },
        xaxis_title='False Positive Rate (1 - Specificity)',
        yaxis_title='True Positive Rate (Sensitivity)',
        xaxis=dict(
            range=[0, 1], 
            gridcolor='#e0e0e0',
            showgrid=True,
            zeroline=True,
            zerolinecolor='#999',
            zerolinewidth=1,
            title_font=dict(size=14, color='#333'), 
            tickfont=dict(size=12, color='#333')    
        ),
        yaxis=dict(
            range=[0, 1], 
            gridcolor='#e0e0e0',
            showgrid=True,
            zeroline=True,
            zerolinecolor='#999',
            zerolinewidth=1,
            title_font=dict(size=14, color='#333'), 
            tickfont=dict(size=12, color='#333')    
        ),
        width=None,  
        height=500,
        showlegend=True,
        legend=dict(
            x=0.98,
            y=0.02,
            xanchor='right',
            yanchor='bottom',
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='#ccc',
            borderwidth=1,
            font=dict(size=12, color='#333') 
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        margin=dict(l=60, r=20, t=60, b=60)
    )
    
    return fig


def plot_precision_recall_curve(precision, recall, avg_precision, title="Precision-Recall Curve"):
    """
    Create an interactive Precision-Recall curve plot.
    
    Args:
        precision: Precision values
        recall: Recall values
        avg_precision: Average precision score
        title: Plot title
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # PR curve
    fig.add_trace(go.Scatter(
        x=recall,
        y=precision,
        mode='lines',
        name=f'PR Curve (AP = {avg_precision:.4f})',
        line=dict(color='#2ca02c', width=3),
        hovertemplate='Recall: %{x:.4f}<br>Precision: %{y:.4f}<extra></extra>',
        showlegend=True
    ))
    
    # Layout
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#333'}
        },
        xaxis_title='Recall (Sensitivity)',
        yaxis_title='Precision',
        xaxis=dict(
            range=[0, 1], 
            gridcolor='#e0e0e0',
            showgrid=True,
            title_font=dict(size=14, color='#333'),
            tickfont=dict(size=12, color='#333')
        ),
        yaxis=dict(
            range=[0, 1], 
            gridcolor='#e0e0e0',
            showgrid=True,
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
            font=dict(size=12, color='#333') 
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        margin=dict(l=60, r=20, t=60, b=60)
    )
    
    return fig


def plot_confusion_matrix(cm, labels=['Negative', 'Positive'], title="Confusion Matrix"):
    """
    Create an interactive confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix (2x2 numpy array)
        labels: Class labels
        title: Plot title
    
    Returns:
        Plotly figure object
    """
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create annotations
    annotations = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            annotations.append(
                dict(
                    x=j,
                    y=i,
                    text=f'{cm[i, j]}<br>({cm_normalized[i, j]:.2%})',
                    font=dict(size=14, color='white' if cm_normalized[i, j] > 0.5 else 'black'),
                    showarrow=False
                )
            )
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        showscale=True,
        hovertemplate='Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#333'}
        },
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        xaxis=dict(
            title_font=dict(size=14, color='#333'),
            tickfont=dict(size=12, color='#333')
        ),
        yaxis=dict(
            title_font=dict(size=14, color='#333'),
            tickfont=dict(size=12, color='#333')
        ),
        width=None,
        height=500,
        annotations=annotations,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig


def compare_models_roc(results_dict):
    """
    Compare multiple models on the same ROC plot with smooth curves.
    
    Args:
        results_dict: Dictionary of {model_name: (fpr, tpr, auc_score)}
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, (model_name, (fpr, tpr, auc_score)) in enumerate(results_dict.items()):
        color = colors[idx % len(colors)]
        
        # Check if we should interpolate
        if len(fpr) < 50:
            # Interpolate for smoother curves
            fpr_smooth, tpr_smooth = interpolate_roc_curve(fpr, tpr, num_points=200)
            
            # Plot smooth line
            fig.add_trace(go.Scatter(
                x=fpr_smooth,
                y=tpr_smooth,
                mode='lines',
                name=f'{model_name} (AUC = {auc_score:.4f})',
                line=dict(color=color, width=2),
                hoverinfo='skip',
                showlegend=True
            ))
            
            # Add markers for actual points
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode='markers',
                name=f'{model_name} Points',
                marker=dict(color=color, size=6, symbol='circle'),
                hovertemplate=f'{model_name}<br>FPR: %{{x:.4f}}<br>TPR: %{{y:.4f}}<extra></extra>',
                showlegend=False
            ))
        else:
            # Plot directly for larger datasets
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name=f'{model_name} (AUC = {auc_score:.4f})',
                line=dict(color=color, width=2),
                hovertemplate=f'{model_name}<br>FPR: %{{x:.4f}}<br>TPR: %{{y:.4f}}<extra></extra>',
                showlegend=True
            ))
    
    # Diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='gray', width=2, dash='dash'),
        hoverinfo='skip',
        showlegend=True
    ))
    
    fig.update_layout(
        title={
            'text': 'Model Comparison - ROC Curves',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#333'}
        },
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        xaxis=dict(
            range=[0, 1],
            gridcolor='#e0e0e0',
            showgrid=True,
            title_font=dict(size=14, color='#333'),
            tickfont=dict(size=12, color='#333')
        ),
        yaxis=dict(
            range=[0, 1],
            gridcolor='#e0e0e0',
            showgrid=True,
            title_font=dict(size=14, color='#333'),
            tickfont=dict(size=12, color='#333')
        ),
        width=None,
        height=600,
        showlegend=True,
        legend=dict(
            x=0.98,
            y=0.02,
            xanchor='right',
            yanchor='bottom',
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='#ccc',
            borderwidth=1,
            font=dict(size=11, color='#333')
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        margin=dict(l=60, r=20, t=60, b=60)
    )
    
    return fig