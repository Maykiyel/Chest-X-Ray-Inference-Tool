"""Service layer for model loading, batch inference, and result summaries."""

from pathlib import Path
import tempfile
from typing import Callable, Dict, Iterable, List, Optional, Tuple

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
