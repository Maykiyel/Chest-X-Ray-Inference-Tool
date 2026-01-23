"""
Configuration file for Chest X-Ray Inference Tool

Modify these settings to customize the application behavior.
"""

# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================

# Available models and their configurations
MODELS_CONFIG = {
    'nih': {
        'name': 'NIH Model',
        'description': 'DenseNet121 trained on NIH ChestX-ray8',
        'weights': 'densenet121-res224-nih',
        'input_size': 224,
        'enabled_by_default': True
    },
    'mimic': {
        'name': 'MIMIC Model',
        'description': 'DenseNet121 trained on MIMIC-CXR (MIT)',
        'weights': 'densenet121-res224-mimic_nb',
        'input_size': 224,
        'enabled_by_default': True
    },
    'chexpert': {
        'name': 'CheXpert Model',
        'description': 'DenseNet121 trained on CheXpert (Stanford)',
        'weights': 'densenet121-res224-chex',  # UPDATED
        'input_size': 224,
        'enabled_by_default': True
    }
}

# ==============================================================================
# INFERENCE CONFIGURATION
# ==============================================================================

# Default batch size for inference
DEFAULT_BATCH_SIZE = 16

# Maximum batch size (to prevent OOM errors)
MAX_BATCH_SIZE = 64

# Use mixed precision for faster inference (if supported)
USE_MIXED_PRECISION = False

# Number of workers for DataLoader (set to 0 for Streamlit compatibility)
NUM_WORKERS = 0

# ==============================================================================
# IMAGE PROCESSING CONFIGURATION
# ==============================================================================

# Supported image file extensions
SUPPORTED_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.dcm', '.dicom']

# Image preprocessing settings
IMAGE_CONFIG = {
    'target_size': 224,
    'normalize_mean': 0.5,
    'normalize_std': 0.5,
    'interpolation': 'bilinear'  # Options: 'bilinear', 'bicubic', 'nearest'
}

# DICOM-specific settings
DICOM_CONFIG = {
    'apply_voi_lut': True,
    'apply_modality_lut': True
}

# ==============================================================================
# UI CONFIGURATION
# ==============================================================================

# Page configuration
PAGE_CONFIG = {
    'page_title': 'Chest X-Ray Inference Tool',
    'page_icon': '🫁',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Theme colors (for custom styling)
THEME_COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff9800',
    'info': '#17a2b8'
}

# ==============================================================================
# RESULTS CONFIGURATION
# ==============================================================================

# Default probability threshold for high-risk predictions
DEFAULT_THRESHOLD = 0.5

# Number of top predictions to show by default
TOP_N_PREDICTIONS = 5

# CSV export settings
CSV_CONFIG = {
    'include_timestamp': True,
    'include_filepath': True,
    'decimal_places': 4,
    'date_format': '%Y-%m-%d %H:%M:%S'
}

# ==============================================================================
# ROC/AUC CONFIGURATION
# ==============================================================================

# ROC curve plotting settings
ROC_CONFIG = {
    'figure_width': 800,
    'figure_height': 600,
    'line_width': 3,
    'show_optimal_threshold': True,
    'show_confidence_interval': False
}

# Minimum number of samples required for ROC analysis
MIN_SAMPLES_FOR_ROC = 10

# ==============================================================================
# PERFORMANCE CONFIGURATION
# ==============================================================================

# Cache settings
CACHE_CONFIG = {
    'ttl': None,  # Time to live for cached models (None = never expire)
    'max_entries': 10  # Maximum number of cached items
}

# Memory management
MEMORY_CONFIG = {
    'clear_cache_after_batch': False,
    'empty_cuda_cache': True  # Clear CUDA cache after each batch
}

# ==============================================================================
# LOGGING CONFIGURATION
# ==============================================================================

# Enable/disable logging
ENABLE_LOGGING = True

# Log level: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
LOG_LEVEL = 'INFO'

# Log file path (None to disable file logging)
LOG_FILE = None  # e.g., 'xray_inference.log'

# ==============================================================================
# LABEL DETECTION CONFIGURATION
# ==============================================================================

# Pathology names for automatic label detection
PATHOLOGY_NAMES = [
    'atelectasis',
    'cardiomegaly',
    'consolidation',
    'edema',
    'effusion',
    'emphysema',
    'fibrosis',
    'hernia',
    'infiltration',
    'mass',
    'nodule',
    'pleural_thickening',
    'pneumonia',
    'pneumothorax',
    'covid',
    'fracture',
    'lung_opacity',
    'lung_lesion'
]

# Label detection patterns
LABEL_PATTERNS = {
    'positive': ['_positive', '_1', 'positive_', '_pos', 'pos_'],
    'negative': ['_negative', '_0', 'negative_', '_neg', 'neg_']
}

# ==============================================================================
# ADVANCED SETTINGS
# ==============================================================================

# GPU settings
GPU_CONFIG = {
    'cuda_deterministic': False,
    'cuda_benchmark': True,  # Enable cuDNN auto-tuner
    'multi_gpu': False  # Enable multi-GPU inference (experimental)
}

# Preprocessing optimization
PREPROCESSING_CONFIG = {
    'cache_preprocessed': False,  # Cache preprocessed images
    'parallel_preprocessing': False  # Use multiprocessing for preprocessing
}

# Ensemble settings (for combining multiple models)
ENSEMBLE_CONFIG = {
    'method': 'mean',  # Options: 'mean', 'max', 'weighted'
    'weights': None  # Dictionary of model weights for weighted ensemble
}

# ==============================================================================
# VALIDATION
# ==============================================================================

def validate_config():
    """Validate configuration settings."""
    assert DEFAULT_BATCH_SIZE > 0, "Batch size must be positive"
    assert DEFAULT_BATCH_SIZE <= MAX_BATCH_SIZE, "Default batch size exceeds maximum"
    assert IMAGE_CONFIG['target_size'] > 0, "Target size must be positive"
    assert 0 <= DEFAULT_THRESHOLD <= 1, "Threshold must be between 0 and 1"
    assert TOP_N_PREDICTIONS > 0, "Top N predictions must be positive"
    assert MIN_SAMPLES_FOR_ROC > 0, "Minimum samples for ROC must be positive"
    print("✅ Configuration validated successfully")

# Validate on import
validate_config()