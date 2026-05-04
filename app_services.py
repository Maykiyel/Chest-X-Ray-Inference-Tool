"""Service layer for model loading, batch inference, and result summaries."""

from pathlib import Path
import hashlib
import json
import tempfile
from typing import Callable, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd
from PIL import Image, ImageStat
import streamlit as st

from inference import load_models, predict_batch
from utils import extract_folder_label, get_image_paths


ProgressCallback = Optional[Callable[[int, float], None]]
CONFIG_FILE = Path('.app_config.json')


@st.cache_resource(show_spinner=False)
def get_cached_models(model_names: Tuple[str, ...], device: str):
    """Cache model loading to avoid reloading weights between reruns."""
    return load_models(list(model_names), device)


def run_upload_inference(
    uploaded_files: Iterable,
    models: Dict,
    device: str,
    batch_size: int,
    auto_label: bool = True,
    progress_callback: ProgressCallback = None,
    predict_batch_fn: Callable = predict_batch,
) -> List[dict]:
    """Run batched inference for uploaded files across all selected models."""
    uploaded_files = list(uploaded_files)
    if not uploaded_files or not models:
        return []

    effective_batch_size = max(1, min(batch_size, len(uploaded_files)))

    with tempfile.TemporaryDirectory(prefix="xray_uploads_") as temp_dir:
        temp_paths = []
        for file in uploaded_files:
            temp_path = Path(temp_dir) / file.name
            with open(temp_path, "wb") as handle:
                handle.write(file.getbuffer())
            temp_paths.append(temp_path)

        results = []
        for model_idx, (model_name, model) in enumerate(models.items()):
            nested_progress = None
            if progress_callback is not None:
                def nested_progress(progress: float, idx: int = model_idx):
                    progress_callback(idx, progress)

            predictions = predict_batch_fn(
                temp_paths,
                model,
                model_name,
                device,
                batch_size=effective_batch_size,
                auto_label=auto_label,
                progress_callback=nested_progress,
            )
            results.extend(predictions)

        return results


def build_top_findings_summary(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """Return highest-logit pathology per image/model pair."""
    if filtered_df.empty:
        return pd.DataFrame(columns=['filename', 'model', 'top_pathology', 'top_logit'])

    score_col = 'logit' if 'logit' in filtered_df.columns else 'probability'
    summary = (
        filtered_df
        .sort_values(score_col, ascending=False)
        .groupby(['filename', 'model'], as_index=False)
        .first()[['filename', 'model', 'pathology', score_col]]
        .rename(columns={'pathology': 'top_pathology', score_col: 'top_logit'})
    )
    return summary


def apply_results_preset(df: pd.DataFrame, preset: str) -> pd.DataFrame:
    """Apply a UI-friendly preset filter to a results dataframe."""
    if df.empty:
        return df

    score_col = 'logit' if 'logit' in df.columns else 'probability'

    if preset == 'High logit (≥0.847)':
        return df[df[score_col] >= 0.847]
    if preset == 'Top finding per image/model':
        top_df = build_top_findings_summary(df)
        return top_df.rename(columns={'top_pathology': 'pathology', 'top_logit': score_col})

    return df


def create_run_snapshot(results_df: pd.DataFrame, run_stats: Optional[dict], label: str) -> dict:
    """Create a serializable run snapshot for quick reload/compare UX."""
    stats = run_stats or {}
    return {
        'id': str(uuid4())[:8],
        'label': label,
        'timestamp': pd.Timestamp.utcnow().isoformat(),
        'rows': len(results_df),
        'results': results_df.copy(),
        'stats': dict(stats),
    }


def save_snapshot_to_history(snapshot: dict, base_dir: str = '.run_history') -> Path:
    """Persist snapshot metadata + results to local run-history storage."""
    history_dir = Path(base_dir)
    history_dir.mkdir(parents=True, exist_ok=True)

    run_id = snapshot['id']
    metadata_path = history_dir / f'{run_id}.json'
    csv_path = history_dir / f'{run_id}.csv'

    metadata = {k: v for k, v in snapshot.items() if k != 'results'}
    metadata['csv_file'] = csv_path.name

    with open(metadata_path, 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2)

    snapshot['results'].to_csv(csv_path, index=False)
    return metadata_path


def list_run_history(base_dir: str = '.run_history') -> List[dict]:
    """List persisted run-history metadata records."""
    history_dir = Path(base_dir)
    if not history_dir.exists():
        return []

    records = []
    for metadata_path in sorted(history_dir.glob('*.json')):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as handle:
                records.append(json.load(handle))
        except Exception:
            continue

    return sorted(records, key=lambda item: item.get('timestamp', ''), reverse=True)


def load_snapshot_from_history(run_id: str, base_dir: str = '.run_history') -> Optional[dict]:
    """Load a persisted run snapshot by id."""
    history_dir = Path(base_dir)
    metadata_path = history_dir / f'{run_id}.json'
    if not metadata_path.exists():
        return None

    with open(metadata_path, 'r', encoding='utf-8') as handle:
        metadata = json.load(handle)

    csv_name = metadata.get('csv_file', f'{run_id}.csv')
    csv_path = history_dir / csv_name
    if not csv_path.exists():
        return None

    results_df = pd.read_csv(csv_path)
    return {
        'id': metadata.get('id', run_id),
        'label': metadata.get('label', run_id),
        'timestamp': metadata.get('timestamp'),
        'rows': metadata.get('rows', len(results_df)),
        'results': results_df,
        'stats': metadata.get('stats', {}),
    }


def audit_folder_quality(folder_path: Path, recursive: bool = True) -> dict:
    """Basic data-quality audit for imaging folder."""
    images = get_image_paths(folder_path, recursive=recursive)
    stats = {
        'total_images': len(images),
        'unreadable': [],
        'low_contrast': [],
        'small_resolution': [],
        'duplicates': {},
        'labeled_count': 0,
    }
    hashes = {}

    for image_path in images:
        if extract_folder_label(image_path) is not None:
            stats['labeled_count'] += 1

        try:
            with Image.open(image_path) as img:
                gray = img.convert('L')
                width, height = gray.size
                contrast = float(ImageStat.Stat(gray).stddev[0])
                if width < 256 or height < 256:
                    stats['small_resolution'].append({'file': image_path.name, 'size': f'{width}x{height}'})
                if contrast < 20.0:
                    stats['low_contrast'].append({'file': image_path.name, 'contrast_std': round(contrast, 2)})
        except Exception as exc:
            stats['unreadable'].append({'file': image_path.name, 'error': str(exc)})
            continue

        digest = hashlib.md5(image_path.read_bytes()).hexdigest()
        hashes.setdefault(digest, []).append(image_path.name)

    stats['duplicates'] = {k: v for k, v in hashes.items() if len(v) > 1}
    stats['label_percentage'] = round((stats['labeled_count'] / stats['total_images'] * 100), 1) if stats['total_images'] else 0.0
    return stats


def get_image_explainability(results_df: pd.DataFrame, filename: str) -> dict:
    """Build lightweight explainability summary for one image."""
    subset = results_df[results_df['filename'] == filename].copy()
    if subset.empty:
        return {'available': False}

    top3 = (
        subset.sort_values('probability', ascending=False)
        .groupby('model', as_index=False)
        .head(3)[['model', 'pathology', 'probability']]
    )
    top1 = subset.sort_values('probability', ascending=False).groupby('model', as_index=False).first()
    agreement_pathology = top1['pathology'].mode().iloc[0] if not top1.empty else None
    agreement_score = round((top1['pathology'] == agreement_pathology).mean(), 2) if agreement_pathology else 0.0

    normal_rows = subset[subset['pathology'] == 'Normal']
    normal_confidence = float(normal_rows['probability'].mean()) if not normal_rows.empty else 0.0

    return {
        'available': True,
        'top3': top3,
        'agreement_pathology': agreement_pathology,
        'agreement_score': agreement_score,
        'normal_confidence': round(normal_confidence, 3),
    }


def build_image_consensus_triage(
    results_df: pd.DataFrame,
    positive_threshold: float = 0.5,
    high_risk_threshold: float = 0.75,
) -> pd.DataFrame:
    """Aggregate model outputs into per-image triage rows for clinical prioritization."""
    required_cols = {'filename', 'pathology', 'model', 'probability'}
    if results_df.empty or not required_cols.issubset(results_df.columns):
        return pd.DataFrame(columns=[
            'filename',
            'pathology',
            'models_reporting',
            'mean_probability',
            'max_probability',
            'std_probability',
            'positive_votes',
            'vote_fraction',
            'risk_band',
        ])

    grouped = (
        results_df
        .groupby(['filename', 'pathology'], as_index=False)
        .agg(
            models_reporting=('model', 'nunique'),
            mean_probability=('probability', 'mean'),
            max_probability=('probability', 'max'),
            std_probability=('probability', 'std'),
            positive_votes=('probability', lambda s: int((s >= positive_threshold).sum())),
        )
    )

    grouped['std_probability'] = grouped['std_probability'].fillna(0.0)
    grouped['vote_fraction'] = np.where(
        grouped['models_reporting'] > 0,
        grouped['positive_votes'] / grouped['models_reporting'],
        0.0,
    )

    def classify_risk(row: pd.Series) -> str:
        if row['mean_probability'] >= high_risk_threshold and row['vote_fraction'] >= 0.67:
            return 'High'
        if row['mean_probability'] >= positive_threshold and row['vote_fraction'] >= 0.5:
            return 'Moderate'
        return 'Low'

    grouped['risk_band'] = grouped.apply(classify_risk, axis=1)
    grouped['mean_probability'] = grouped['mean_probability'].round(4)
    grouped['max_probability'] = grouped['max_probability'].round(4)
    grouped['std_probability'] = grouped['std_probability'].round(4)
    grouped['vote_fraction'] = grouped['vote_fraction'].round(4)

    return grouped.sort_values(
        ['risk_band', 'mean_probability', 'vote_fraction', 'max_probability'],
        ascending=[True, False, False, False],
        key=lambda col: col.map({'High': 0, 'Moderate': 1, 'Low': 2}) if col.name == 'risk_band' else col,
    )


def load_app_config() -> dict:
    """Load user config from json file if available."""
    if not CONFIG_FILE.exists():
        return {}
    try:
        return json.loads(CONFIG_FILE.read_text(encoding='utf-8'))
    except Exception:
        return {}


def save_app_config(config: dict) -> None:
    """Save user config to json file."""
    CONFIG_FILE.write_text(json.dumps(config, indent=2), encoding='utf-8')
