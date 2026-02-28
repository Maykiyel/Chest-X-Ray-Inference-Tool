import streamlit as st
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
import time

from inference import predict_batch
from utils import get_image_paths, validate_labels_in_folder
from metrics import compute_confusion_matrix_metrics, plot_confusion_matrix_heatmap
from app_constants import ALL_PATHOLOGIES, DEFAULT_COMMON_PATHOLOGIES
from app_services import (
    get_cached_models,
    run_upload_inference,
    build_top_findings_summary,
    apply_results_preset,
    create_run_snapshot,
    save_snapshot_to_history,
    list_run_history,
    load_snapshot_from_history,
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
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📁 Inference", "📊 Results", "🔢 Confusion Matrix", "📝 Rename Files", "🏷️ Label Preview"])

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
            if setup_ready and st.button("🚀 Process", use_container_width=True):
                st.session_state.prediction_cache_key += 1
                folder = Path(folder_path)
                
                # Check if path exists and provide helpful feedback
                if not folder.exists():
                    st.error(f"❌ Folder not found: `{folder.absolute()}`")
                    
                    # Suggest alternatives
                    st.markdown("**Try these:**")
                    
                    # Check if it's a relative path issue
                    import os
                    cwd = Path(os.getcwd())
                    
                    # Try common variations
                    alternatives = [
                        Path(folder_path),  # As-is
                        cwd / folder_path,  # Relative to current dir
                        Path(folder_path.lstrip('/')),  # Remove leading slash
                        Path(folder_path.lstrip('./')),  # Remove ./
                    ]
                    
                    st.code(f"Current directory: {cwd}", language="bash")
                    
                    for alt in alternatives:
                        if alt.exists() and alt.is_dir():
                            st.success(f"✓ Found folder at: `{alt.absolute()}`")
                            st.info(f"💡 Try entering: `{alt.name}` or `{alt.absolute()}`")
                            break
                    else:
                        st.warning("Folder not found in common locations. Please check the path.")
                        st.markdown("""
                        **Tips:**
                        - Use `test` for a folder named 'test' in the current directory
                        - Use `./test` for the same
                        - Use full path: `C:\\Users\\Name\\folder\\test` (Windows)
                        - Use full path: `/home/user/folder/test` (Linux/Mac)
                        """)
                    
                elif not folder.is_dir():
                    st.error(f"❌ Path exists but is not a folder: `{folder.absolute()}`")
                else:
                    # Path is valid, proceed
                    run_started = time.perf_counter()
                    with st.spinner("Loading models..."):
                        models = get_cached_models(tuple(sorted(selected_models)), device)
                    
                    if models:
                        image_paths = get_image_paths(folder, recursive=recursive_search)
                        
                        if image_paths:
                            st.success(f"✓ Found {len(image_paths)} images in: `{folder.absolute()}`")
                            
                            progress_bar = st.progress(0)
                            all_results = []
                            
                            for model_idx, (model_name, model) in enumerate(models.items()):
                                def update_progress(p):
                                    progress_bar.progress((model_idx + p) / len(models))
                                
                                results = predict_batch(
                                    image_paths, model, model_name, device,
                                    batch_size=batch_size, auto_label=auto_label,
                                    progress_callback=update_progress
                                )
                                all_results.extend(results)
                            
                            if all_results:
                                df = pd.DataFrame(all_results)
                                df = df[df['pathology'].isin(selected_pathologies)]
                                st.session_state.results_df = df
                                st.session_state.last_run_stats = {
                                    'mode': 'folder',
                                    'image_count': len(image_paths),
                                    'model_count': len(models),
                                    'rows': len(df),
                                    'device': device,
                                    'duration_sec': round(time.perf_counter() - run_started, 2),
                                }
                                st.success(f"✅ Processed {len(image_paths)} images!")
                                st.balloons()
                        else:
                            st.warning(f"No images found in: `{folder.absolute()}`")
                            st.info("Supported formats: .png, .jpg, .jpeg, .dcm, .dicom")

with tab2:
    st.header("Results & Analysis")
    
    if st.session_state.results_df is not None:
        df = st.session_state.results_df
        
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
            min_prob = st.slider("Min Probability", 0.0, 1.0, 0.0, 0.05)
        with col4:
            preset_filter = st.selectbox("View Preset", ['All results', 'High confidence (≥0.70)', 'Top finding per image/model'])
        
        filtered_df = df[df['model'].isin(model_filter)]
        if pathology_filter:
            filtered_df = filtered_df[filtered_df['pathology'].isin(pathology_filter)]
        filtered_df = filtered_df[filtered_df['probability'] >= min_prob]
        filtered_df = apply_results_preset(filtered_df, preset_filter)
        
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

        st.dataframe(filtered_df.sort_values('probability', ascending=False), use_container_width=True, height=400)

        st.subheader("Top Findings per Image")
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
        df = st.session_state.results_df
        
        # Check if ground truth is available
        has_ground_truth = 'ground_truth' in df.columns and df['ground_truth'].notna().any()
        
        if not has_ground_truth:
            st.warning("⚠️ No ground truth labels detected in filenames or folders!")
            st.markdown("""
            **To enable automatic confusion matrix:**
            
            1. Rename files using the **Rename Files** tab with pattern: `Pathology_img1.png`
            2. Or organize files in folders: `pathology_positive/`, `pathology_negative/`
            3. Ensure "Auto-detect labels" is enabled when processing
            
            **Example filenames:**
            - `Pneumonia_img1.png` → Pneumonia positive
            - `Effusion_img23.png` → Effusion positive
            - `Atelectasis_img456.png` → Atelectasis positive
            """)
        else:
            st.success(f"✓ Ground truth detected for {df['ground_truth'].notna().sum()} predictions")
            
            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                model_cm = st.selectbox("Model", df['model'].unique())
            with col2:
                # Get pathologies that have ground truth
                pathologies_with_gt = df[df['ground_truth'].notna()]['pathology'].unique()
                pathology_cm = st.selectbox("Pathology", sorted(pathologies_with_gt))
            with col3:
                threshold = st.slider("Threshold", 0.0, 1.0, 0.5, 0.05)
            
            if st.button("📊 Generate Matrix", use_container_width=True, type="primary"):
                # Filter data
                cm_data = df[(df['model'] == model_cm) & 
                            (df['pathology'] == pathology_cm) & 
                            (df['ground_truth'].notna())].copy()
                
                if len(cm_data) > 0:
                    # Compute metrics
                    metrics = compute_confusion_matrix_metrics(
                        cm_data['ground_truth'].values,
                        cm_data['probability'].values,
                        threshold
                    )
                    
                    # Display confusion matrix
                    st.subheader(f"{model_cm.upper()} - {pathology_cm}")
                    
                    fig = plot_confusion_matrix_heatmap(
                        metrics['confusion_matrix'],
                        title=f"{model_cm.upper()} - {pathology_cm} (Threshold: {threshold})"
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
                        st.write(f"**Threshold:** {metrics['threshold']:.3f}")
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
                        
                        for old_path, new_name in rename_map.items():
                            try:
                                new_path = old_path.parent / new_name
                                
                                # Check if target exists
                                if new_path.exists():
                                    errors.append(f"{new_name}: File already exists")
                                    continue
                                
                                old_path.rename(new_path)
                                success_count += 1
                            except Exception as e:
                                errors.append(f"{old_path.name}: {str(e)}")
                        
                        if success_count == len(rename_map):
                            st.success(f"✅ Renamed {success_count} files as **{pathology_for_rename}** images!")
                            st.info(f"📁 Backup saved: {backup_path.name}")
                            st.balloons()
                        else:
                            st.warning(f"⚠️ Renamed {success_count}/{len(rename_map)} files")
                            if errors:
                                with st.expander("⚠️ Show Errors"):
                                    for err in errors[:20]:  # Show first 20 errors
                                        st.error(err)
                                    if len(errors) > 20:
                                        st.info(f"... and {len(errors) - 20} more errors")
                
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
