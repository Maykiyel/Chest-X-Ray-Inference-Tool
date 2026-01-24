import streamlit as st
import pandas as pd
import torch
from pathlib import Path
import time
from datetime import datetime
import plotly.graph_objects as go
import shutil

from inference import load_models, predict_single_image, predict_batch
from utils import get_image_paths, extract_folder_label, save_results_to_csv
from metrics import compute_confusion_matrix_metrics, plot_confusion_matrix_heatmap

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
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = {}
if 'selected_pathologies' not in st.session_state:
    st.session_state.selected_pathologies = []
if 'prediction_cache_key' not in st.session_state:
    st.session_state.prediction_cache_key = 0

ALL_PATHOLOGIES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule',
    'Pleural_Thickening', 'Pneumonia', 'Pneumothorax', 'Normal'
]

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
            st.session_state.selected_pathologies = [
                'Atelectasis', 'Cardiomegaly', 'Effusion', 
                'Edema', 'Pneumonia', 'Pneumothorax', 'Normal'
            ]
    with col2:
        if st.button("All", use_container_width=True):
            st.session_state.selected_pathologies = ALL_PATHOLOGIES.copy()
    
    selected_pathologies = st.multiselect(
        "Choose pathologies:",
        options=ALL_PATHOLOGIES,
        default=st.session_state.selected_pathologies if st.session_state.selected_pathologies else [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 
            'Edema', 'Pneumonia', 'Pneumothorax', 'Normal'
        ]
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
tab1, tab2, tab3, tab4 = st.tabs(["📁 Inference", "📊 Results", "🔢 Confusion Matrix", "📝 Rename Files"])

with tab1:
    st.header("Upload Images")
    
    if selected_pathologies:
        st.info(f"🎯 **Selected:** {', '.join(selected_pathologies)}")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Multiple Images")
        uploaded_files = st.file_uploader(
            "Choose X-ray images",
            type=['png', 'jpg', 'jpeg', 'dcm'],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.prediction_cache_key}"
        )
        
        if uploaded_files and selected_models and selected_pathologies:
            if st.button("🔍 Analyze", use_container_width=True):
                st.session_state.prediction_cache_key += 1
                
                with st.spinner("Loading models..."):
                    models = load_models(selected_models, device)
                
                if models:
                    progress_bar = st.progress(0)
                    results = []
                    
                    for idx, file in enumerate(uploaded_files):
                        temp_path = Path(f"temp_{st.session_state.prediction_cache_key}_{file.name}")
                        with open(temp_path, "wb") as f:
                            f.write(file.getbuffer())
                        
                        for model_name, model in models.items():
                            try:
                                predictions = predict_single_image(str(temp_path), model, model_name, device)
                                results.extend(predictions)
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                        
                        temp_path.unlink()
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                    
                    if results:
                        df = pd.DataFrame(results)
                        df = df[df['pathology'].isin(selected_pathologies)]
                        st.session_state.results_df = df
                        st.success(f"✅ Processed {len(uploaded_files)} images!")
    
    with col2:
        st.subheader("Batch Folder")
        folder_path = st.text_input("Folder path", placeholder="/path/to/images")
        recursive_search = st.checkbox("Search subfolders", value=True)
        auto_label = st.checkbox("Auto-detect labels", value=True)
        
        if folder_path and selected_models and selected_pathologies:
            if st.button("🚀 Process", use_container_width=True):
                st.session_state.prediction_cache_key += 1
                folder = Path(folder_path)
                
                if folder.exists():
                    with st.spinner("Loading models..."):
                        models = load_models(selected_models, device)
                    
                    if models:
                        image_paths = get_image_paths(folder, recursive=recursive_search)
                        
                        if image_paths:
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
                                st.success(f"✅ Processed {len(image_paths)} images!")
                                st.balloons()

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
        
        col1, col2, col3 = st.columns(3)
        with col1:
            model_filter = st.multiselect("Filter Model", df['model'].unique(), df['model'].unique())
        with col2:
            pathology_filter = st.multiselect("Filter Pathology", sorted(df['pathology'].unique()), [])
        with col3:
            min_prob = st.slider("Min Probability", 0.0, 1.0, 0.0, 0.05)
        
        filtered_df = df[df['model'].isin(model_filter)]
        if pathology_filter:
            filtered_df = filtered_df[filtered_df['pathology'].isin(pathology_filter)]
        filtered_df = filtered_df[filtered_df['probability'] >= min_prob]
        
        st.dataframe(filtered_df.sort_values('probability', ascending=False), use_container_width=True, height=400)
        
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
    st.header("Confusion Matrix Analysis")
    
    if st.session_state.results_df is not None:
        df = st.session_state.results_df
        
        st.info("Set threshold and enter expected labels to generate confusion matrix")
        
        threshold = st.slider("Threshold", 0.0, 1.0, 0.5, 0.05)
        
        col1, col2 = st.columns(2)
        with col1:
            model_cm = st.selectbox("Model", df['model'].unique())
        with col2:
            pathology_cm = st.selectbox("Pathology", sorted([p for p in df['pathology'].unique() if p != 'Normal']))
        
        st.subheader("Expected Labels")
        st.markdown("""
        Enter expected labels (one per line): `filename: label`
        - Use `1`, `positive`, `pos` for positive
        - Use `0`, `negative`, `neg` for negative
        
        **Example:**
        ```
        img_001.png: 1
        img_002.png: 0
        ```
        """)
        
        labels_text = st.text_area("Expected labels:", height=200, 
                                   placeholder="img_001.png: 1\nimg_002.png: 0")
        
        if st.button("📊 Generate Matrix", use_container_width=True):
            if labels_text.strip():
                expected = {}
                for line in labels_text.strip().split('\n'):
                    if ':' in line:
                        filename, label = line.split(':', 1)
                        filename = filename.strip()
                        label = label.strip().lower()
                        
                        if label in ['1', 'positive', 'pos', 'true', 'yes']:
                            expected[filename] = 1
                        elif label in ['0', 'negative', 'neg', 'false', 'no']:
                            expected[filename] = 0
                
                if expected:
                    cm_data = df[(df['model'] == model_cm) & (df['pathology'] == pathology_cm)].copy()
                    cm_data['expected'] = cm_data['filename'].map(expected)
                    cm_data = cm_data[cm_data['expected'].notna()]
                    
                    if len(cm_data) > 0:
                        metrics = compute_confusion_matrix_metrics(
                            cm_data['expected'].values,
                            cm_data['probability'].values,
                            threshold
                        )
                        
                        st.subheader(f"{model_cm.upper()} - {pathology_cm}")
                        
                        fig = plot_confusion_matrix_heatmap(
                            metrics['confusion_matrix'],
                            title=f"{model_cm.upper()} - {pathology_cm} (Threshold: {threshold})"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.subheader("Metrics")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                        with col2:
                            st.metric("Sensitivity", f"{metrics['sensitivity']:.3f}")
                        with col3:
                            st.metric("Specificity", f"{metrics['specificity']:.3f}")
                        with col4:
                            st.metric("F1 Score", f"{metrics['f1_score']:.3f}")
                        
                        with st.expander("📋 Details"):
                            st.write(f"TP: {metrics['true_positives']}, TN: {metrics['true_negatives']}")
                            st.write(f"FP: {metrics['false_positives']}, FN: {metrics['false_negatives']}")
                    else:
                        st.error("No matching images found")
                else:
                    st.error("No valid labels found")
    else:
        st.info("👈 Process images first")

with tab4:
    st.header("Batch File Renaming")
    
    st.markdown("""
    Rename X-ray images with sequential numbering: `img_1.png`, `img_2.png`, etc.
    
    **Features:**
    - Custom prefix and numbering
    - Preview before applying
    - Backup of original names
    """)
    
    rename_folder = st.text_input("Folder Path", placeholder="/path/to/images")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        prefix = st.text_input("Prefix", value="img")
    with col2:
        start_num = st.number_input("Start #", min_value=1, value=1)
    with col3:
        padding = st.number_input("Padding", min_value=1, max_value=6, value=3)
    
    include_subfolders_rename = st.checkbox("Include subfolders", value=False)
    
    if rename_folder:
        folder = Path(rename_folder)
        
        if folder.exists() and folder.is_dir():
            images = get_image_paths(folder, recursive=include_subfolders_rename)
            
            if images:
                st.success(f"✓ Found {len(images)} images")
                
                # Generate preview
                rename_map = {}
                counter = start_num
                
                for img_path in sorted(images):
                    ext = img_path.suffix
                    new_name = f"{prefix}_{str(counter).zfill(padding)}{ext}"
                    rename_map[img_path] = new_name
                    counter += 1
                
                # Preview
                st.subheader("Preview (first 10)")
                preview_df = pd.DataFrame([
                    {'Original': p.name, 'New Name': new_name}
                    for p, new_name in list(rename_map.items())[:10]
                ])
                st.dataframe(preview_df, use_container_width=True, hide_index=True)
                
                # Rename buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("✅ Apply Renaming", use_container_width=True, type="primary"):
                        # Create backup file
                        backup_path = folder / f"rename_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                        
                        with open(backup_path, 'w') as f:
                            for old_path, new_name in rename_map.items():
                                f.write(f"{old_path.name} -> {new_name}\n")
                        
                        # Rename files
                        success_count = 0
                        errors = []
                        
                        for old_path, new_name in rename_map.items():
                            try:
                                new_path = old_path.parent / new_name
                                old_path.rename(new_path)
                                success_count += 1
                            except Exception as e:
                                errors.append(f"{old_path.name}: {str(e)}")
                        
                        if success_count == len(rename_map):
                            st.success(f"✅ Renamed {success_count} files!")
                            st.info(f"Backup saved: {backup_path.name}")
                        else:
                            st.warning(f"⚠️ Renamed {success_count}/{len(rename_map)} files")
                            if errors:
                                with st.expander("Show Errors"):
                                    for err in errors:
                                        st.error(err)
                
                with col2:
                    # Download mapping CSV
                    mapping_df = pd.DataFrame([
                        {'Original': p.name, 'New': new_name}
                        for p, new_name in rename_map.items()
                    ])
                    csv = mapping_df.to_csv(index=False)
                    
                    st.download_button(
                        "📥 Download Mapping",
                        csv,
                        f"rename_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        use_container_width=True
                    )
            else:
                st.warning("No images found")
        else:
            st.error("Invalid folder path")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Powered by TorchXRayVision | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)