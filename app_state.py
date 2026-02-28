"""Session-state helpers for the Streamlit UI."""

from copy import deepcopy

import streamlit as st

SESSION_DEFAULTS = {
    'results_df': None,
    'models_loaded': {},
    'selected_pathologies': [],
    'prediction_cache_key': 0,
    'debug_mode': False,
    'last_run_stats': None,
}


def initialize_session_state() -> None:
    """Populate any missing session keys with independent default values."""
    for key, default in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = deepcopy(default)
