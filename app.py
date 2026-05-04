import streamlit as st
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import time

from inference import predict_batch
from utils import get_image_paths, validate_labels_in_folder
from metrics import compute_confusion_matrix_metrics, plot_confusion_matrix_heatmap, recommend_threshold
from app_constants import ALL_PATHOLOGIES, DEFAULT_COMMON_PATHOLOGIES
from app_services import (
    get_cached_models,
    run_upload_inference,
    build_top_findings_summary,
    build_image_consensus_triage,
    apply_results_preset,
    create_run_snapshot,
    save_snapshot_to_history,
    list_run_history,
    load_snapshot_from_history,
    audit_folder_quality,
    get_image_explainability,
    load_app_config,
    save_app_config,
)
from app_state import initialize_session_state

# Page configuration
st.set_page_config(
    page_title="Chest X-Ray Inference Tool",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 0rem 1rem; }
    .stProgress > div > div > div > div { background-color: #1f77b4; }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
initialize_session_state()

THRESHOLD_PRESETS = {
    'Screening (high sensitivity)': 0.35,
    'Balanced': 0.50,
    'Confirmatory (high specificity)': 0.70,
}

TRUE_RESULT_LOGIT_THRESHOLD = 0.68

st.title("🫁 Chest X-Ray Inference Tool")
st.markdown("Advanced multi-model inference using TorchXRayVision")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Select Models")
    use_nih = st.checkbox("NIH Model (DenseNet121)", value=True)
    use_mimic = st.checkbox("MIMIC Model", value=True)
    use_chexpert = st.checkbox("CheXpert Model", value=True)
    
    selected_models = []
    if use_nih: selected_models.append('nih')
    if use_mimic: selected_models.append('mimic')
    if use_chexpert: selected_models.append('chexpert')
    
    st.divider()
    
    st.subheader("🎯 Select Pathologies")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Common", use_container_width=True):
            st.session_state.selected_pathologies = DEFAULT_COMMON_PATHOLOGIES.copy()
    with col2:
        if st.button("All", use_container_width=True):
            st.session_state.selected_pathologies = ALL_PATHOLOGIES.copy()
    
    selected_pathologies = st.multiselect(
        "Choose pathologies:",
        options=ALL_PATHOLOGIES,
        default=st.session_state.selected_pathologies if st.session_state.selected_pathologies else DEFAULT_COMMON_PATHOLOGIES
    )
    st.session_state.selected_pathologies = selected_pathologies
    
    if selected_pathologies:
        st.success(f"✓ {len(selected_pathologies)} selected")
    
    st.divider()
    
    st.subheader("Device Selection")
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        device_options = ['GPU (CUDA)', 'CPU']
        default_device = 0
    else:
        device_options = ['CPU']
        default_device = 0
    
    selected_device = st.radio("Processing Device:", device_options, index=default_device)
    device = 'cuda' if selected_device == 'GPU (CUDA)' else 'cpu'
    
    if device == 'cuda':
        st.success(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        st.info("🖥️ Using CPU")
    
    st.divider()
    
    st.subheader("Batch Processing")
    batch_size = st.slider("Batch Size", 1, 64 if device == 'cuda' else 16, 
                           16 if device == 'cuda' else 4)

    st.subheader("🐞 Developer Tools")
    st.session_state.debug_mode = st.checkbox("Enable debug panel", value=st.session_state.debug_mode)

    with st.expander("⚙️ JSON Configuration", expanded=False):
        if st.button("Load config", use_container_width=True):
            cfg = load_app_config()
            if cfg:
                st.session_state.selected_pathologies = cfg.get('default_pathologies', st.session_state.selected_pathologies)
                st.success("Loaded .app_config.json")
            else:
                st.info("No config found or invalid config.")
        default_threshold_mode = st.selectbox("Default threshold mode", list(THRESHOLD_PRESETS.keys()), index=1)
        if st.button("Save current setup as config", use_container_width=True):
            save_app_config({
                'default_models': selected_models,
                'default_pathologies': selected_pathologies,
                'default_threshold_mode': default_threshold_mode,
                'default_batch_size': batch_size,
            })
            st.success("Saved .app_config.json")
    
    st.divider()
    
    st.subheader("🔄 Cache Control")
    if st.button("Clear Cache", use_container_width=True):
        st.cache_resource.clear()
        st.session_state.models_loaded = {}
        st.session_state.prediction_cache_key += 1
        if cuda_available:
            torch.cuda.empty_cache()
        st.success("✓ Cache cleared!")
        st.rerun()

# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📁 Inference",
    "📊 Results",
    "🔢 Confusion Matrix",
    "📝 Rename Files",
    "🏷️ Label Preview",
    "🧪 Audit & Explainability",
    "🩺 Clinical Triage",
])

with tab1:
    st.header("Upload Images")

    st.caption("Guided workflow")
    setup_ready = bool(selected_models) and bool(selected_pathologies)
    steps = [
        f"1) Models: {'✅' if selected_models else '⚪'}",
        f"2) Pathologies: {'✅' if selected_pathologies else '⚪'}",
        "3) Input source: Upload files or set folder path",
        "4) Run inference and review results",
    ]
    st.markdown(" | ".join(steps))

    with st.expander("Current run configuration", expanded=False):
        st.write({
            'models': selected_models,
            'pathologies_count': len(selected_pathologies),
            'device': device,
            'batch_size': batch_size,
        })

    if selected_pathologies:
        st.info(f"🎯 **Selected:** {', '.join(selected_pathologies)}")
    else:
        st.warning("Select at least one pathology to enable inference.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Multiple Images")
        auto_label_upload = st.checkbox(
            "Auto-detect labels from uploaded filenames",
            value=True,
            help="For uploads, labels can be inferred from filenames such as pathology_1, pathology_0, positive_pathology.",
        )
        uploaded_files = st.file_uploader(
            "Choose X-ray images",
            type=['png', 'jpg', 'jpeg', 'dcm'],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.prediction_cache_key}"
        )
        
        if uploaded_files:
            if not setup_ready:
                st.info("Select at least one model and pathology to run analysis.")
            if setup_ready and st.button("🔍 Analyze", use_container_width=True):
                st.session_state.prediction_cache_key += 1
                
                run_started = time.perf_counter()
                with st.spinner("Loading models..."):
                    models = get_cached_models(tuple(sorted(selected_models)), device)
                
                if models:
                    progress_bar = st.progress(0)

                    def update_progress(model_idx, progress):
                        progress_bar.progress((model_idx + progress) / len(models))

                    try:
                        results = run_upload_inference(
                            uploaded_files,
                            models,
                            device,
                            batch_size,
                            auto_label=auto_label_upload,
                            progress_callback=update_progress,
                        )
                    except Exception as e:
                        st.error(f"Upload processing failed: {str(e)}")
                        results = []

                    if results:
                        df = pd.DataFrame(results)
                        df = df[df['pathology'].isin(selected_pathologies)]
                        st.session_state.results_df = df
                        st.session_state.last_run_stats = {
                            'mode': 'upload',
                            'image_count': len(uploaded_files),
                            'model_count': len(models),
                            'rows': len(df),
                            'device': device,
                            'duration_sec': round(time.perf_counter() - run_started, 2),
                        }
                        st.success(f"✅ Processed {len(uploaded_files)} images!")
    
    with col2:
        st.subheader("Batch Folder")
        
        # Path input with better help text
        folder_path = st.text_input(
            "Folder path", 
            placeholder="test  or  ./test  or  C:\\full\\path\\to\\folder",
            help="Use relative path (e.g., 'test') or full absolute path"
        )
        
        # Show current working directory for reference
        if st.checkbox("Show current directory", value=False):
            import os
            st.code(f"Current directory: {os.getcwd()}", language="bash")
            st.info("💡 Tip: Use relative paths from this directory (e.g., 'test' for a folder named 'test')")
        
        recursive_search = st.checkbox("Search subfolders", value=True)
        auto_label = st.checkbox("Auto-detect labels", value=True)
        
        if folder_path:
            if not setup_ready:
                st.info("Select at least one model and pathology to run batch processing.")

            enqueue_col, start_col = st.columns(2)
            with enqueue_col:
                if setup_ready and st.button("➕ Enqueue Batch Job", use_container_width=True):
                    folder = Path(folder_path)
                    if folder.exists() and folder.is_dir():
                        st.session_state.batch_jobs.append({
                            'job_id': st.session_state.next_job_id,
                            'folder_path': str(folder),
                            'recursive': recursive_search,
                            'auto_label': auto_label,
                            'status': 'queued',
                            'progress': 0.0,
                            'created_at': datetime.now().isoformat(),
                            'message': '',
                            'cancel_requested': False,
                        })
                        st.session_state.next_job_id += 1
                        st.success("Batch job queued.")
                    else:
                        st.error("Invalid folder path. Please verify and try again.")
                        st.info("Try sample dataset: point to your local test folder and enqueue again.")
            with start_col:
                if setup_ready and st.button("▶️ Run Next Job", use_container_width=True):
                    queued_job = next((j for j in st.session_state.batch_jobs if j['status'] == 'queued'), None)
                    if queued_job is None:
                        st.info("No queued jobs available.")
                    else:
                        queued_job['status'] = 'running'
                        run_started = time.perf_counter()
                        try:
                            with st.spinner("Loading models..."):
                                models = get_cached_models(tuple(sorted(selected_models)), device)

                            if not models:
                                queued_job['status'] = 'failed'
                                queued_job['message'] = 'No models loaded.'
                            else:
                                image_paths = get_image_paths(Path(queued_job['folder_path']), recursive=queued_job['recursive'])
                                if not image_paths:
                                    queued_job['status'] = 'failed'
                                    queued_job['message'] = 'No supported images found.'
                                else:
                                    progress_bar = st.progress(0)
                                    all_results = []
                                    for model_idx, (model_name, model) in enumerate(models.items()):
                                        def update_progress(p):
                                            queued_job['progress'] = (model_idx + p) / len(models)
                                            progress_bar.progress(queued_job['progress'])

                                        if queued_job.get('cancel_requested'):
                                            queued_job['status'] = 'cancelled'
                                            queued_job['message'] = 'Cancelled by user.'
                                            break

                                        results = predict_batch(
                                            image_paths,
                                            model,
                                            model_name,
                                            device,
                                            batch_size=batch_size,
                                            auto_label=queued_job['auto_label'],
                                            progress_callback=update_progress,
                                        )
                                        all_results.extend(results)

                                    if queued_job['status'] != 'cancelled':
                                        df = pd.DataFrame(all_results)
                                        df = df[df['pathology'].isin(selected_pathologies)]
                                        st.session_state.results_df = df
                                        st.session_state.last_run_stats = {
                                            'mode': 'folder',
                                            'image_count': len(image_paths),
                                            'model_count': len(models),
                                            'rows': len(df),
                                            'device': device,
                                            'batch_size': batch_size,
                                            'selected_models': selected_models,
                                            'selected_pathologies': selected_pathologies,
                                            'duration_sec': round(time.perf_counter() - run_started, 2),
                                        }
                                        queued_job['status'] = 'completed'
                                        queued_job['progress'] = 1.0
                                        queued_job['message'] = f"Processed {len(image_paths)} images."
                                        st.success(queued_job['message'])
                        except Exception as exc:
                            queued_job['status'] = 'failed'
                            queued_job['message'] = f'Inference exception: {exc}'
                            st.error(queued_job['message'])

            if st.session_state.batch_jobs:
                st.subheader("Batch Queue Lifecycle")
                jobs_df = pd.DataFrame(st.session_state.batch_jobs)
                st.dataframe(jobs_df[['job_id', 'folder_path', 'status', 'progress', 'created_at', 'message']], use_container_width=True, hide_index=True)
                job_ids = [j['job_id'] for j in st.session_state.batch_jobs if j['status'] in ['queued', 'running']]
                if job_ids:
                    cancel_job_id = st.selectbox('Select job to cancel', job_ids)
                    if st.button('🛑 Cancel Selected Job', use_container_width=True):
                        for job in st.session_state.batch_jobs:
                            if job['job_id'] == cancel_job_id:
                                if job['status'] == 'queued':
                                    job['status'] = 'cancelled'
                                    job['message'] = 'Cancelled before execution.'
                                else:
                                    job['cancel_requested'] = True
                                break
                        st.warning('Cancellation requested.')

with tab2:
    st.header("Results & Analysis")
    
    if st.session_state.results_df is not None:
        df = st.session_state.results_df.copy()
        if 'logit' not in df.columns and 'probability' in df.columns:
            probs = df['probability'].clip(1e-6, 1 - 1e-6)
            df['logit'] = probs.apply(lambda p: float(np.log(p / (1 - p))))
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Predictions", len(df))
        with col2:
            st.metric("Images", df['filename'].nunique())
        with col3:
            st.metric("Models", df['model'].nunique())
        with col4:
            st.metric("Pathologies", df['pathology'].nunique())
        
        st.divider()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            model_filter = st.multiselect("Filter Model", df['model'].unique(), df['model'].unique())
        with col2:
            pathology_filter = st.multiselect("Filter Pathology", sorted(df['pathology'].unique()), [])
        with col3:
            min_logit = st.number_input("Min Logit", value=float(-10.0), step=0.1, format='%.3f')
        with col4:
            preset_filter = st.selectbox("View Preset", ['All results', 'High logit (≥0.847)', 'Top finding per image/model'])

        top_only = st.toggle('Top findings only', value=False)
        sort_preset = st.selectbox('Sort preset', ['Highest risk first', 'By model', 'By image'])
        
        filtered_df = df[df['model'].isin(model_filter)]
        if pathology_filter:
            filtered_df = filtered_df[filtered_df['pathology'].isin(pathology_filter)]
        filtered_df = filtered_df[filtered_df['logit'] >= min_logit]
        filtered_df = apply_results_preset(filtered_df, preset_filter)
        if top_only:
            filtered_df = apply_results_preset(filtered_df, 'Top finding per image/model')

        if 'logit' in filtered_df.columns:
            if sort_preset == 'By model':
                filtered_df = filtered_df.sort_values(['model', 'logit'], ascending=[True, False])
            elif sort_preset == 'By image':
                filtered_df = filtered_df.sort_values(['filename', 'logit'], ascending=[True, False])
            else:
                filtered_df = filtered_df.sort_values('logit', ascending=False)
        
        if st.session_state.last_run_stats:
            stats = st.session_state.last_run_stats
            st.caption(f"Last run: mode={stats.get('mode')} | images={stats.get('image_count')} | duration={stats.get('duration_sec', 'n/a')}s")

        save_col1, save_col2, save_col3 = st.columns([2,1,1])
        with save_col1:
            run_label = st.text_input('Run label', value=f"Run {datetime.now().strftime('%H:%M:%S')}")
        with save_col2:
            if st.button('💾 Save Run', use_container_width=True):
                snapshot = create_run_snapshot(df, st.session_state.last_run_stats, run_label)
                st.session_state.saved_runs.append(snapshot)
                history_path = save_snapshot_to_history(snapshot)
                st.success(f"Saved {snapshot['label']} (history: {history_path.name})")
        with save_col3:
            if st.button('🗑️ Clear Saved', use_container_width=True):
                st.session_state.saved_runs = []
                st.session_state.active_run_id = None
                st.info('Saved runs cleared.')

        if not filtered_df.empty and 'logit' in filtered_df.columns:
            display_df = filtered_df.copy()
            display_df['true_result'] = pd.NA
            if 'ground_truth' in display_df.columns:
                has_gt = display_df['ground_truth'].notna()
                is_positive_pred = display_df['logit'] >= TRUE_RESULT_LOGIT_THRESHOLD
                display_df.loc[has_gt, 'true_result'] = ((is_positive_pred & (display_df['ground_truth'] == 1)) | (~is_positive_pred & (display_df['ground_truth'] == 0))).loc[has_gt].astype(int)
            st.dataframe(display_df, use_container_width=True, height=400)
        else:
            st.dataframe(filtered_df, use_container_width=True, height=400)

        st.subheader("Top Findings per Image (Top Logit)")
        if not filtered_df.empty:
            summary_df = build_top_findings_summary(filtered_df)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        else:
            st.info("No rows match the current filters.")

        if st.session_state.saved_runs:
            st.subheader('Saved Runs')
            run_options = {f"{item['label']} ({item['rows']} rows)": item['id'] for item in st.session_state.saved_runs}
            selected_run_label = st.selectbox('Load a saved run', list(run_options.keys()))
            selected_run_id = run_options[selected_run_label]
            load_col1, load_col2 = st.columns(2)
            with load_col1:
                if st.button('📂 Load Selected Run', use_container_width=True):
                    selected = next((r for r in st.session_state.saved_runs if r['id'] == selected_run_id), None)
                    if selected is not None:
                        st.session_state.results_df = selected['results'].copy()
                        st.session_state.last_run_stats = selected.get('stats')
                        st.success(f"Loaded {selected['label']}")
                        st.rerun()
            with load_col2:
                if st.button('❌ Delete Selected Run', use_container_width=True):
                    st.session_state.saved_runs = [r for r in st.session_state.saved_runs if r['id'] != selected_run_id]
                    st.success('Deleted selected run.')
                    st.rerun()



        if len(st.session_state.saved_runs) >= 2:
            st.subheader('Run Comparison')
            compare_options = {f"{r['label']} ({r['rows']} rows)": r for r in st.session_state.saved_runs}
            c1, c2 = st.columns(2)
            with c1:
                run_a_label = st.selectbox('Run A', list(compare_options.keys()), key='run_a')
            with c2:
                run_b_label = st.selectbox('Run B', list(compare_options.keys()), index=1, key='run_b')
            run_a = compare_options[run_a_label]
            run_b = compare_options[run_b_label]
            comp_df = pd.DataFrame([
                {'metric': 'rows', 'run_a': run_a['rows'], 'run_b': run_b['rows']},
                {'metric': 'images', 'run_a': run_a['results']['filename'].nunique(), 'run_b': run_b['results']['filename'].nunique()},
                {'metric': 'models', 'run_a': run_a['results']['model'].nunique(), 'run_b': run_b['results']['model'].nunique()},
            ])
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

        st.subheader('Run History (persisted on disk)')
        history_rows = list_run_history()
        if history_rows:
            history_labels = {f"{item.get('label', item.get('id'))} | {item.get('timestamp', 'n/a')}": item.get('id') for item in history_rows}
            history_pick = st.selectbox('History records', list(history_labels.keys()))
            history_id = history_labels[history_pick]
            hist_col1, hist_col2 = st.columns(2)
            with hist_col1:
                if st.button('📥 Load History Record', use_container_width=True):
                    history_snapshot = load_snapshot_from_history(history_id)
                    if history_snapshot is not None:
                        st.session_state.results_df = history_snapshot['results'].copy()
                        st.session_state.last_run_stats = history_snapshot.get('stats')
                        st.success(f"Loaded history run: {history_snapshot['label']}")
                        st.rerun()
                    else:
                        st.error('Could not load history record.')
            with hist_col2:
                st.caption('History files are stored in `.run_history/` in your project directory.')
        else:
            st.info('No persisted run history yet. Save a run to create your first record.')

        csv = filtered_df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv,
            f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            use_container_width=True
        )
    else:
        st.info("👈 Upload images to see results")

with tab3:
    st.header("Automated Confusion Matrix")
    
    if st.session_state.results_df is not None:
        df = st.session_state.results_df.copy()
        if 'logit' not in df.columns and 'probability' in df.columns:
            probs = df['probability'].clip(1e-6, 1 - 1e-6)
            df['logit'] = probs.apply(lambda p: float(np.log(p / (1 - p))))
        
        # Check if ground truth is available
        has_ground_truth = 'ground_truth' in df.columns and df['ground_truth'].notna().any()
        
        if not has_ground_truth:
            st.warning("⚠️ No ground truth labels detected in filenames or folders!")
            st.markdown("""
            **To enable automatic confusion matrix:**
            
            1. Rename files using the **Rename Files** tab with pattern: `Pathology_img1.png`
            2. Or include filename labels like: `pathology_1.png`, `pathology_0.png`, `positive_pathology.png`
            3. Or organize files in folders: `pathology_positive/`, `pathology_negative/`
            4. Ensure "Auto-detect labels" is enabled when processing
            
            **Example filenames:**
            - `Pneumonia_img1.png` → Pneumonia positive
            - `Effusion_img23.png` → Effusion positive
            - `Atelectasis_img456.png` → Atelectasis positive
            """)
            st.info(
                "If you used file upload mode, make sure 'Auto-detect labels from uploaded filenames' was enabled before running Analyze."
            )
        else:
            labeled_predictions = int(df['ground_truth'].notna().sum())
            unlabeled_predictions = int(len(df) - labeled_predictions)
            st.success(f"✓ Ground truth detected for {labeled_predictions} predictions")
            if unlabeled_predictions > 0:
                st.warning(
                    f"⚠️ {unlabeled_predictions} predictions have no ground-truth label and will be excluded from confusion matrix metrics."
                )
            
            # Filters
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                model_cm = st.selectbox("Model", df['model'].unique())
            with col2:
                # Get pathologies that have ground truth
                pathologies_with_gt = df[df['ground_truth'].notna()]['pathology'].unique()
                pathology_cm = st.selectbox("Pathology", sorted(pathologies_with_gt))
            with col3:
                threshold_mode = st.selectbox('Threshold mode', list(THRESHOLD_PRESETS.keys()), index=1)
            with col4:
                threshold = st.number_input('Logit Threshold', value=0.0, step=0.1, format='%.3f')

            selection_df = df[(df['model'] == model_cm) & (df['pathology'] == pathology_cm)]
            selection_labeled = int(selection_df['ground_truth'].notna().sum())
            selection_unlabeled = int(len(selection_df) - selection_labeled)
            if selection_unlabeled > 0:
                st.info(
                    f"For this selection, {selection_labeled} labeled rows will be used and {selection_unlabeled} unlabeled rows will be skipped."
                )

            with st.expander("🎯 Auto-recommend threshold", expanded=False):
                threshold_strategy = st.selectbox(
                    "Optimization strategy",
                    options=['youden', 'f1', 'accuracy'],
                    help="Finds the threshold that maximizes the selected criterion on labeled rows.",
                )
                if st.button("Suggest threshold", use_container_width=True):
                    threshold_df = df[
                        (df['model'] == model_cm) &
                        (df['pathology'] == pathology_cm) &
                        (df['ground_truth'].notna())
                    ].copy()

                    if len(threshold_df) < 10:
                        st.warning("At least 10 labeled rows are recommended for stable threshold suggestion.")

                    if threshold_df.empty:
                        st.error("No labeled rows available for threshold recommendation.")
                    else:
                        rec = recommend_threshold(
                            threshold_df['ground_truth'].values,
                            threshold_df['logit'].values,
                            strategy=threshold_strategy,
                        )
                        st.success(
                            f"Suggested threshold: {rec['threshold']:.2f} using {threshold_strategy.upper()} optimization "
                            f"(score={rec['score']:.3f})."
                        )
                        rec_metrics = rec.get('metrics') or {}
                        if rec_metrics:
                            st.caption(
                                f"Expected sensitivity={rec_metrics.get('sensitivity', 0.0):.3f}, "
                                f"specificity={rec_metrics.get('specificity', 0.0):.3f}, "
                                f"F1={rec_metrics.get('f1_score', 0.0):.3f}"
                            )
            
            if st.button("📊 Generate Matrix", use_container_width=True, type="primary"):
                # Filter data
                cm_data = df[(df['model'] == model_cm) & 
                            (df['pathology'] == pathology_cm) & 
                            (df['ground_truth'].notna())].copy()
                
                if len(cm_data) > 0:
                    # Compute metrics
                    metrics = compute_confusion_matrix_metrics(
                        cm_data['ground_truth'].values,
                        cm_data['logit'].values,
                        threshold
                    )
                    
                    # Display confusion matrix
                    st.subheader(f"{model_cm.upper()} - {pathology_cm}")
                    
                    fig = plot_confusion_matrix_heatmap(
                        metrics['confusion_matrix'],
                        title=f"{model_cm.upper()} - {pathology_cm} (Logit threshold: {threshold})"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display metrics
                    st.subheader("Performance Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                    with col2:
                        st.metric("Sensitivity", f"{metrics['sensitivity']:.3f}")
                    with col3:
                        st.metric("Specificity", f"{metrics['specificity']:.3f}")
                    with col4:
                        st.metric("F1 Score", f"{metrics['f1_score']:.3f}")
                    
                    # Details
                    with st.expander("📋 Detailed Metrics"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**True Positives:** {metrics['true_positives']}")
                            st.write(f"**True Negatives:** {metrics['true_negatives']}")
                            st.write(f"**Precision:** {metrics['precision']:.3f}")
                            st.write(f"**NPV:** {metrics['npv']:.3f}")
                        with col2:
                            st.write(f"**False Positives:** {metrics['false_positives']}")
                            st.write(f"**False Negatives:** {metrics['false_negatives']}")
                            st.write(f"**FPR:** {metrics['fpr']:.3f}")
                            st.write(f"**FNR:** {metrics['fnr']:.3f}")
                        
                        st.write(f"**Total Samples:** {metrics['total']}")
                        st.write(f"**Logit Threshold:** {metrics['threshold']:.3f}")
                else:
                    st.error("No data with ground truth for this model/pathology combination")
    else:
        st.info("👈 Process images first")

with tab4:
    st.header("Batch File Renaming - Pathology-Based")
    
    st.markdown("""
    Rename X-ray images by **pathology type** with pattern: `Pathology_img1.png`
    
    **This naming enables automatic confusion matrix generation!**
    
    **Features:**
    - Group files by pathology
    - Sequential numbering per pathology
    - Preview before applying
    - Backup of original names
    """)
    
    rename_folder = st.text_input(
        "Folder Path", 
        placeholder="test  or  ./test  or  C:\\full\\path\\to\\folder",
        help="Use relative path (e.g., 'test') or full absolute path"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        pathology_for_rename = st.selectbox(
            "Pathology Type",
            ALL_PATHOLOGIES,
            help="All files will be renamed as this pathology"
        )
    with col2:
        start_num = st.number_input("Start Number", min_value=1, value=1)
    
    include_subfolders_rename = st.checkbox("Include subfolders", value=False)
    dry_run = st.checkbox('Dry run (preview only)', value=True)
    collision_policy = st.selectbox('Collision policy', ['skip', 'overwrite', 'append_suffix'])
    
    if rename_folder:
        folder = Path(rename_folder)
        
        # Better path validation
        if not folder.exists():
            st.error(f"❌ Folder not found: `{folder.absolute()}`")
            
            import os
            cwd = Path(os.getcwd())
            st.code(f"Current directory: {cwd}", language="bash")
            st.info("💡 Use relative path (e.g., 'test') or full absolute path")
            
        elif not folder.is_dir():
            st.error(f"❌ Path exists but is not a folder: `{folder.absolute()}`")
        else:
            images = get_image_paths(folder, recursive=include_subfolders_rename)
            
            if images:
                st.success(f"✓ Found {len(images)} images")
                
                # Generate preview with pathology-based naming
                rename_map = {}
                counter = start_num
                
                for img_path in sorted(images):
                    ext = img_path.suffix
                    # Format: Pathology_imgN.ext
                    new_name = f"{pathology_for_rename}_img{counter}{ext}"
                    rename_map[img_path] = new_name
                    counter += 1
                
                # Preview
                st.subheader(f"Preview: {pathology_for_rename} Images")
                preview_df = pd.DataFrame([
                    {'Original': p.name, 'New Name': new_name}
                    for p, new_name in list(rename_map.items())[:10]
                ])
                st.dataframe(preview_df, use_container_width=True, hide_index=True)
                
                if len(images) > 10:
                    st.info(f"Showing first 10 of {len(images)} files...")
                
                # Rename buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("✅ Apply Renaming", use_container_width=True, type="primary"):
                        # Create backup file
                        backup_path = folder / f"rename_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                        
                        with open(backup_path, 'w') as f:
                            f.write(f"Pathology: {pathology_for_rename}\n")
                            f.write(f"Date: {datetime.now().isoformat()}\n")
                            f.write(f"Total files: {len(rename_map)}\n\n")
                            for old_path, new_name in rename_map.items():
                                f.write(f"{old_path.name} -> {new_name}\n")
                        
                        # Rename files
                        success_count = 0
                        errors = []
                        undo_records = []
                        
                        for old_path, new_name in rename_map.items():
                            try:
                                new_path = old_path.parent / new_name

                                if new_path.exists():
                                    if collision_policy == 'skip':
                                        errors.append(f"{new_name}: File already exists")
                                        continue
                                    elif collision_policy == 'append_suffix':
                                        stem = new_path.stem
                                        suffix = new_path.suffix
                                        new_path = new_path.parent / f"{stem}_dup{suffix}"
                                    # overwrite falls through
                                
                                if not dry_run:
                                    old_path.rename(new_path)
                                    undo_records.append(f"{new_path.name} -> {old_path.name}")
                                success_count += 1
                            except Exception as e:
                                errors.append(f"{old_path.name}: {str(e)}")
                        
                        if success_count == len(rename_map):
                            if dry_run:
                                st.success(f"✅ Dry run complete for {success_count} files. No files were changed.")
                            else:
                                st.success(f"✅ Renamed {success_count} files as **{pathology_for_rename}** images!")
                                st.info(f"📁 Backup saved: {backup_path.name}")
                                undo_file = folder / 'rename_undo_last.txt'
                                undo_file.write_text('\n'.join(undo_records), encoding='utf-8')
                                st.info(f"↩️ Undo map saved: {undo_file.name}")
                                st.balloons()
                        else:
                            st.warning(f"⚠️ Renamed {success_count}/{len(rename_map)} files")
                            if errors:
                                with st.expander("⚠️ Show Errors"):
                                    for err in errors[:20]:  # Show first 20 errors
                                        st.error(err)
                                    if len(errors) > 20:
                                        st.info(f"... and {len(errors) - 20} more errors")
                
                if st.button('↩️ Undo Last Rename', use_container_width=True):
                    undo_file = folder / 'rename_undo_last.txt'
                    if undo_file.exists():
                        lines = [line.strip() for line in undo_file.read_text(encoding='utf-8').splitlines() if '->' in line]
                        undone = 0
                        for line in lines:
                            new_name, old_name = [x.strip() for x in line.split('->')]
                            new_path = folder / new_name
                            old_path = folder / old_name
                            if new_path.exists() and not old_path.exists():
                                new_path.rename(old_path)
                                undone += 1
                        st.success(f'Undo completed for {undone} files.')
                    else:
                        st.info('No undo file found yet.')

                with col2:
                    # Download mapping CSV
                    mapping_df = pd.DataFrame([
                        {'Original': p.name, 'New': new_name, 'Pathology': pathology_for_rename}
                        for p, new_name in rename_map.items()
                    ])
                    csv = mapping_df.to_csv(index=False)
                    
                    st.download_button(
                        "📥 Download Mapping",
                        csv,
                        f"rename_map_{pathology_for_rename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        use_container_width=True
                    )
            else:
                st.warning("No images found in folder")

with tab5:
    st.header("🏷️ Label Detection Preview")
    
    st.markdown("""
    Preview what labels will be detected from your folder structure or filenames.
    
    **Supports TWO labeling methods:**
    
    ### 1. Filename-Based Labeling (Priority 1)
    - `Pneumonia_img1.png` → Pneumonia = Positive
    - `Effusion_img23.png` → Effusion = Positive
    - `Normal_img5.png` → Normal = Positive
    
    ### 2. Folder-Based Labeling (Priority 2)
    - `pneumonia_positive/img1.png` → Pneumonia = Positive
    - `pneumonia_negative/img2.png` → Pneumonia = Negative
    - `pneumonia_1/img3.png` → Pneumonia = Positive
    - `pneumonia_0/img4.png` → Pneumonia = Negative
    - `pneumonia/positive/img5.png` → Pneumonia = Positive
    - `pneumonia/negative/img6.png` → Pneumonia = Negative
    - `normal/img7.png` → Normal = Positive
    
    **Note:** Filename labels take priority over folder labels!
    """)
    
    st.divider()
    
    preview_folder = st.text_input(
        "Folder to Preview", 
        placeholder="test  or  ./test  or  C:\\full\\path\\to\\folder",
        help="Use relative path (e.g., 'test') or full absolute path"
    )
    preview_recursive = st.checkbox("Search subfolders recursively", value=True, key="preview_recursive")
    
    if preview_folder:
        if st.button("🔍 Analyze Labels", use_container_width=True, type="primary"):
            folder = Path(preview_folder)
            
            # Better error handling
            if not folder.exists():
                st.error(f"❌ Folder not found: `{folder.absolute()}`")
                
                import os
                cwd = Path(os.getcwd())
                st.code(f"Current directory: {cwd}", language="bash")
                
                # Try to find the folder
                alternatives = [
                    cwd / preview_folder,
                    Path(preview_folder.lstrip('/')),
                ]
                
                for alt in alternatives:
                    if alt.exists() and alt.is_dir():
                        st.success(f"✓ Found folder at: `{alt.absolute()}`")
                        st.info(f"💡 Try entering: `{alt.name}` or `{alt.absolute()}`")
                        break
                else:
                    st.warning("Use relative path (e.g., 'test') or full absolute path")
                    
            elif not folder.is_dir():
                st.error(f"❌ Path exists but is not a folder: `{folder.absolute()}`")
            else:
                with st.spinner("Analyzing labels..."):
                    stats = validate_labels_in_folder(folder, recursive=preview_recursive)
                
                if stats['valid']:
                    # Summary metrics
                    st.subheader("📊 Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Images", stats['total_images'])
                    with col2:
                        st.metric("Labeled", stats['labeled_count'], 
                                 delta=f"{stats['label_percentage']:.1f}%")
                    with col3:
                        st.metric("From Filename", stats['filename_labels'])
                    with col4:
                        st.metric("From Folder", stats['folder_labels'])
                    
                    # Pathology breakdown
                    if stats['pathology_breakdown']:
                        st.subheader("📋 Pathology Breakdown")
                        
                        breakdown_data = []
                        for pathology, counts in stats['pathology_breakdown'].items():
                            breakdown_data.append({
                                'Pathology': pathology,
                                'Positive': counts['positive'],
                                'Negative': counts['negative'],
                                'Total': counts['positive'] + counts['negative']
                            })
                        
                        breakdown_df = pd.DataFrame(breakdown_data)
                        st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
                    
                    # Labeled files preview
                    if stats['labeled_files']:
                        st.subheader("✅ Labeled Files (First 20)")
                        labeled_df = pd.DataFrame(stats['labeled_files'][:20])
                        labeled_df['Label'] = labeled_df['label'].map({1: 'Positive', 0: 'Negative'})
                        labeled_df['Source'] = labeled_df['source'].map({
                            'filename': '📄 Filename',
                            'folder': '📁 Folder'
                        })
                        
                        display_df = labeled_df[['filename', 'pathology', 'Label', 'Source', 'folder']]
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                        
                        if len(stats['labeled_files']) > 20:
                            st.info(f"Showing 20 of {len(stats['labeled_files'])} labeled files")
                    
                    # Unlabeled files warning
                    if stats['unlabeled_files']:
                        with st.expander(f"⚠️ Unlabeled Files ({stats['unlabeled_count']})", expanded=False):
                            st.warning("These files won't have ground truth for confusion matrix:")
                            unlabeled_df = pd.DataFrame(stats['unlabeled_files'][:20])
                            st.dataframe(unlabeled_df, use_container_width=True, hide_index=True)
                            
                            if len(stats['unlabeled_files']) > 20:
                                st.info(f"Showing 20 of {len(stats['unlabeled_files'])} unlabeled files")
                            
                            st.markdown("""
                            **To fix:** Use the **Rename Files** tab to rename these files with pathology labels,
                            or organize them into labeled folders.
                            """)
                    
                    # Download full report
                    st.divider()
                    
                    if stats['labeled_files']:
                        labeled_report_df = pd.DataFrame(stats['labeled_files'])
                        labeled_report_df['Label'] = labeled_report_df['label'].map({1: 'Positive', 0: 'Negative'})
                        csv_report = labeled_report_df.to_csv(index=False)
                        
                        st.download_button(
                            "📥 Download Full Label Report",
                            csv_report,
                            f"label_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv",
                            use_container_width=True
                        )
                else:
                    st.error(stats.get('error', 'Unknown error'))


with tab6:
    st.header("🧪 Data Quality Audit + Explainability")

    st.subheader("Data Quality Audit")
    audit_folder = st.text_input("Audit folder path", placeholder="test")
    audit_recursive = st.checkbox("Audit recursively", value=True)
    if audit_folder and st.button("Run Audit", use_container_width=True):
        folder = Path(audit_folder)
        if not folder.exists() or not folder.is_dir():
            st.error("Invalid audit folder path.")
            st.info("Try sample dataset: use your test folder path and click Run Audit.")
        else:
            audit = audit_folder_quality(folder, recursive=audit_recursive)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Images", audit['total_images'])
            c2.metric("Label completeness", f"{audit['label_percentage']}%")
            c3.metric("Unreadable", len(audit['unreadable']))
            c4.metric("Duplicate groups", len(audit['duplicates']))

            if audit['small_resolution']:
                st.warning("Unusual small-resolution images detected")
                st.dataframe(pd.DataFrame(audit['small_resolution'][:20]), use_container_width=True, hide_index=True)
            if audit['low_contrast']:
                st.warning("Low-contrast images detected")
                st.dataframe(pd.DataFrame(audit['low_contrast'][:20]), use_container_width=True, hide_index=True)
            if audit['duplicates']:
                dup_df = pd.DataFrame([{'hash': h, 'files': ', '.join(v)} for h, v in audit['duplicates'].items()])
                st.info("Potential duplicate files")
                st.dataframe(dup_df, use_container_width=True, hide_index=True)

    st.subheader("Per-image Explainability")
    if st.session_state.results_df is None or st.session_state.results_df.empty:
        st.info("Run inference first to unlock explainability cards.")
    else:
        filenames = sorted(st.session_state.results_df['filename'].unique())
        selected_image = st.selectbox("Select image", filenames)
        explain = get_image_explainability(st.session_state.results_df, selected_image)
        if explain.get('available'):
            col1, col2, col3 = st.columns(3)
            col1.metric("Model agreement", f"{explain['agreement_score']*100:.0f}%")
            col2.metric("Consensus pathology", explain['agreement_pathology'] or 'n/a')
            col3.metric("Normality confidence", f"{explain['normal_confidence']:.3f}")
            st.dataframe(explain['top3'], use_container_width=True, hide_index=True)


with tab7:
    st.header("🩺 Clinical Triage & Prioritization")
    st.caption("Use model-consensus signals to prioritize likely positive studies for fastest review.")

    if st.session_state.results_df is None or st.session_state.results_df.empty:
        st.info("Run inference first to generate triage-ready predictions.")
    else:
        triage_source_df = st.session_state.results_df.copy()

        triage_col1, triage_col2, triage_col3 = st.columns(3)
        with triage_col1:
            triage_positive_threshold = st.slider(
                "Positive vote threshold",
                min_value=0.05,
                max_value=0.95,
                value=0.50,
                step=0.05,
                help="A model contributes a positive vote if its probability is >= this threshold.",
            )
        with triage_col2:
            triage_high_risk_threshold = st.slider(
                "High-risk mean threshold",
                min_value=0.10,
                max_value=0.99,
                value=0.75,
                step=0.05,
                help="Rows above this mean probability and with strong vote agreement are marked High risk.",
            )
        with triage_col3:
            min_agreement = st.slider(
                "Minimum vote fraction",
                min_value=0.0,
                max_value=1.0,
                value=0.50,
                step=0.05,
                help="Filter out rows where model agreement is too weak.",
            )

        triage_df = build_image_consensus_triage(
            triage_source_df,
            positive_threshold=triage_positive_threshold,
            high_risk_threshold=triage_high_risk_threshold,
        )

        if triage_df.empty:
            st.warning("No triage rows could be generated from current results.")
        else:
            triage_df = triage_df[triage_df['vote_fraction'] >= min_agreement].copy()
            if triage_df.empty:
                st.warning("No rows satisfy the selected minimum vote fraction.")
            else:
                filter_col1, filter_col2 = st.columns(2)
                with filter_col1:
                    risk_filter = st.multiselect(
                        "Risk bands",
                        options=['High', 'Moderate', 'Low'],
                        default=['High', 'Moderate'],
                    )
                with filter_col2:
                    pathology_options = sorted(triage_df['pathology'].unique())
                    pathology_focus = st.multiselect(
                        "Pathology focus",
                        options=pathology_options,
                        default=[],
                    )

                filtered_triage_df = triage_df.copy()
                if risk_filter:
                    filtered_triage_df = filtered_triage_df[filtered_triage_df['risk_band'].isin(risk_filter)]
                if pathology_focus:
                    filtered_triage_df = filtered_triage_df[filtered_triage_df['pathology'].isin(pathology_focus)]

                high_count = int((triage_df['risk_band'] == 'High').sum())
                moderate_count = int((triage_df['risk_band'] == 'Moderate').sum())
                low_count = int((triage_df['risk_band'] == 'Low').sum())
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Triage Rows", int(len(triage_df)))
                m2.metric("High Risk", high_count)
                m3.metric("Moderate Risk", moderate_count)
                m4.metric("Low Risk", low_count)

                st.subheader("Priority Worklist")
                st.dataframe(filtered_triage_df, use_container_width=True, hide_index=True)

                if not filtered_triage_df.empty:
                    top_case = filtered_triage_df.iloc[0]
                    st.info(
                        (
                            f"Top priority now: {top_case['filename']} | {top_case['pathology']} | "
                            f"risk={top_case['risk_band']} | mean={top_case['mean_probability']:.3f} | "
                            f"agreement={top_case['vote_fraction']*100:.0f}%"
                        )
                    )

                    st.subheader("Case Brief Generator")
                    case_list = sorted(filtered_triage_df['filename'].unique())
                    selected_case = st.selectbox("Select case", case_list)
                    case_rows = filtered_triage_df[filtered_triage_df['filename'] == selected_case].copy()
                    case_rows = case_rows.sort_values('mean_probability', ascending=False)
                    st.dataframe(case_rows.head(10), use_container_width=True, hide_index=True)

                    summary_lines = [
                        f"Case: {selected_case}",
                        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
                        "Top consensus findings:",
                    ]
                    for _, row in case_rows.head(5).iterrows():
                        summary_lines.append(
                            (
                                f"- {row['pathology']}: risk={row['risk_band']}, mean={row['mean_probability']:.3f}, "
                                f"max={row['max_probability']:.3f}, agreement={row['vote_fraction']*100:.0f}% "
                                f"({int(row['positive_votes'])}/{int(row['models_reporting'])} models)"
                            )
                        )

                    case_summary = "\n".join(summary_lines)
                    st.text_area("Case brief", value=case_summary, height=180)

                triage_csv = filtered_triage_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Triage Worklist CSV",
                    triage_csv,
                    f"triage_worklist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True,
                )



if st.session_state.debug_mode:
    st.divider()
    st.subheader("🐞 Debug Panel")
    st.json({
        'device': device,
        'selected_models': selected_models,
        'selected_pathologies_count': len(selected_pathologies),
        'cache_key': st.session_state.prediction_cache_key,
        'last_run_stats': st.session_state.last_run_stats,
    })
    if st.session_state.results_df is not None:
        st.caption("Results dataframe preview (first 5 rows)")
        st.dataframe(st.session_state.results_df.head(5), use_container_width=True, hide_index=True)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Powered by TorchXRayVision | Built with Streamlit</p>
    <p style='font-size: 0.9em;'>💡 Supports both filename-based AND folder-based labeling!</p>
</div>
""", unsafe_allow_html=True)
