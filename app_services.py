"""Service layer for model loading, batch inference, and result summaries."""

from pathlib import Path
import json
import tempfile
from typing import Callable, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

import pandas as pd
import streamlit as st

from inference import load_models, predict_batch


ProgressCallback = Optional[Callable[[int, float], None]]


@st.cache_resource(show_spinner=False)
def get_cached_models(model_names: Tuple[str, ...], device: str):
    """Cache model loading to avoid reloading weights between reruns."""
    return load_models(list(model_names), device)


def run_upload_inference(
    uploaded_files: Iterable,
    models: Dict,
    device: str,
    batch_size: int,
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
                auto_label=False,
                progress_callback=nested_progress,
            )
            results.extend(predictions)

        return results


def build_top_findings_summary(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """Return highest-probability pathology per image/model pair."""
    if filtered_df.empty:
        return pd.DataFrame(columns=['filename', 'model', 'top_pathology', 'top_probability'])

    return (
        filtered_df
        .sort_values('probability', ascending=False)
        .groupby(['filename', 'model'], as_index=False)
        .first()[['filename', 'model', 'pathology', 'probability']]
        .rename(columns={'pathology': 'top_pathology', 'probability': 'top_probability'})
    )


def apply_results_preset(df: pd.DataFrame, preset: str) -> pd.DataFrame:
    """Apply a UI-friendly preset filter to a results dataframe."""
    if df.empty:
        return df

    if preset == 'High confidence (≥0.70)':
        return df[df['probability'] >= 0.70]
    if preset == 'Top finding per image/model':
        top_df = build_top_findings_summary(df)
        return top_df.rename(columns={'top_pathology': 'pathology', 'top_probability': 'probability'})

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
