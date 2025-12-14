import streamlit as st
import pandas as pd
import torch
from pathlib import Path
import time
from datetime import datetime
import plotly.graph_objects as go

from inference import load_models, predict_single_image, predict_batch
from utils import get_image_paths, extract_folder_label, save_results_to_csv
from metrics import compute_roc_auc, plot_roc_curve

# Page configuration
st.set_page_config(
    page_title="Chest X-Ray Inference Tool",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = {}

# Title and description
st.title("🫁 Chest X-Ray Inference Tool")
st.markdown("Advanced multi-model inference using TorchXRayVision")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    st.subheader("Select Models")
    use_nih = st.checkbox("NIH Model (DenseNet121)", value=True)
    use_mimic = st.checkbox("MIMIC Model", value=True)
    use_chexpert = st.checkbox("CheXpert Model", value=True)
    
    selected_models = []
    if use_nih:
        selected_models.append('nih')
    if use_mimic:
        selected_models.append('mimic')
    if use_chexpert:
        selected_models.append('chexpert')
    
    st.divider()
    
    # Device selection
    st.subheader("Device Selection")
    
    # Check available devices
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        device_options = ['GPU (CUDA)', 'CPU']
        device_help = f"GPU: {torch.cuda.get_device_name(0)} | Switch to CPU to save GPU for other tasks"
        default_device = 0  # GPU by default
    else:
        device_options = ['CPU']
        device_help = "GPU not available. Install CUDA-enabled PyTorch for GPU support."
        default_device = 0
    
    selected_device = st.radio(
        "Select Processing Device:",
        device_options,
        index=default_device,
        help=device_help
    )
    
    # Convert selection to device string
    device = 'cuda' if selected_device == 'GPU (CUDA)' else 'cpu'
    
    # Display device info
    if device == 'cuda':
        st.success(f"✅ Using: **GPU (CUDA)**")
        st.info(f"🎮 {torch.cuda.get_device_name(0)}")
        
        # Show VRAM info if available
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            st.caption(f"💾 VRAM: {total_memory:.1f} GB")
        except:
            pass
    else:
        st.info(f"🖥️ Using: **CPU**")
        if cuda_available:
            st.caption("💡 Switch to GPU for 10-30x faster processing")
    
    st.divider()
    
    # Batch settings
    st.subheader("Batch Processing")
    
    # Recommend batch size based on device
    if device == 'cuda':
        recommended_batch = 16
        batch_help = "GPU Mode: Larger batches recommended (8-32). Faster processing with minimal memory overhead."
        max_batch = 64
    else:
        recommended_batch = 4
        batch_help = "CPU Mode: Smaller batches recommended (1-4). Reduces memory usage."
        max_batch = 16
    
    batch_size = st.slider(
        "Batch Size", 
        1, 
        max_batch, 
        recommended_batch,
        help=batch_help
    )
    
    # Show performance estimate
    if device == 'cuda':
        est_time = "~0.5-2s per image"
        st.caption(f"⚡ Estimated speed: {est_time}")
    else:
        est_time = "~10-30s per image"
        st.caption(f"🐢 Estimated speed: {est_time}")
    
    st.divider()
    
    # About
    st.subheader("About")
    st.markdown("""
    This tool performs inference on chest X-ray images using state-of-the-art 
    deep learning models from TorchXRayVision.
    
    **Features:**
    - Multi-model inference
    - Multiple image upload
    - Batch processing (150+ images)
    - CPU/GPU selection
    - Normal X-ray detection
    - ROC/AUC analysis
    - CSV export
    
    **Tips:**
    - Use GPU for 10-30x speed boost
    - Upload multiple images at once
    - Organize folders by pathology
    """)
    
    # Show session info
    with st.expander("📊 Session Info"):
        st.write(f"**Device:** {device.upper()}")
        st.write(f"**Batch Size:** {batch_size}")
        st.write(f"**Models Selected:** {len(selected_models)}")
        if st.session_state.results_df is not None:
            st.write(f"**Results Loaded:** {len(st.session_state.results_df)} predictions")

# Main content
tab1, tab2, tab3 = st.tabs(["📁 Single/Batch Inference", "📊 Results & Analysis", "📈 ROC/AUC Curves"])

with tab1:
    st.header("Upload Images")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Multiple Image Upload")
        uploaded_files = st.file_uploader(
            "Choose X-ray images",
            type=['png', 'jpg', 'jpeg', 'dcm'],
            accept_multiple_files=True,
            help="Select one or more images to analyze"
        )
        
        if uploaded_files and selected_models:
            if st.button("🔍 Analyze Images", width='stretch'):
                with st.spinner("Loading models..."):
                    models = load_models(selected_models, device)
                
                if not models:
                    st.error("❌ No models were successfully loaded. Please check the error messages above.")
                else:
                    st.info(f"Processing {len(uploaded_files)} image(s)...")
                    
                    progress_bar = st.progress(0)
                    results = []
                    
                    for idx, uploaded_file in enumerate(uploaded_files):
                        with st.spinner(f"Processing {uploaded_file.name}..."):
                            # Save uploaded file temporarily
                            temp_path = Path(f"temp_{uploaded_file.name}")
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            
                            for model_name, model in models.items():
                                try:
                                    predictions = predict_single_image(
                                        str(temp_path),
                                        model,
                                        model_name,
                                        device
                                    )
                                    results.extend(predictions)
                                except Exception as e:
                                    st.error(f"Error processing {uploaded_file.name} with {model_name}: {str(e)}")
                            
                            temp_path.unlink()  # Clean up
                        
                        # Update progress
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                    
                    if not results:
                        st.error("❌ No predictions were generated. Please check the error messages.")
                    else:
                        # Convert to DataFrame
                        df = pd.DataFrame(results)
                        st.session_state.results_df = df
                        
                        st.success(f"✅ Analysis complete! Processed {len(uploaded_files)} image(s) with {len(models)} model(s).")
                        
                        # Display results in a grid
                        if len(uploaded_files) == 1:
                            # Single image - show side by side
                            col_img, col_pred = st.columns([1, 1])
                            with col_img:
                                st.image(uploaded_files[0], caption="Uploaded X-ray", width='stretch')
                            
                            with col_pred:
                                st.subheader("Top Predictions")
                                for model_name in models.keys():
                                    model_results = df[df['model'] == model_name].nlargest(5, 'probability')
                                    if len(model_results) > 0:
                                        st.markdown(f"**{model_name.upper()}**")
                                        for _, row in model_results.iterrows():
                                            st.progress(row['probability'], text=f"{row['pathology']}: {row['probability']:.3f}")
                        else:
                            # Multiple images - show summary
                            st.subheader("Processing Summary")
                            st.write(f"**Total Images:** {len(uploaded_files)}")
                            st.write(f"**Total Predictions:** {len(df)}")
                            st.write(f"**Models Used:** {', '.join([m.upper() for m in models.keys()])}")
                            
                            # Show preview of results
                            st.subheader("Top 10 High-Risk Findings")
                            top_findings = df.nlargest(10, 'probability')[['filename', 'model', 'pathology', 'probability']]
                            st.dataframe(top_findings, width='stretch')
                            
                            st.info("👉 Go to the 'Results & Analysis' tab to see detailed results and visualizations.")
    
    with col2:
        st.subheader("Batch Folder Processing")
        folder_path = st.text_input(
            "Enter folder path",
            placeholder="/path/to/xray/images"
        )
        
        recursive_search = st.checkbox("Search subfolders recursively", value=True)
        auto_label = st.checkbox("Auto-detect labels from folder names", value=True)
        
        if folder_path and selected_models:
            if st.button("🚀 Process Batch", width='stretch'):
                folder = Path(folder_path)
                
                if not folder.exists():
                    st.error("❌ Folder does not exist!")
                else:
                    with st.spinner("Loading models..."):
                        models = load_models(selected_models, device)
                    
                    if not models:
                        st.error("❌ No models were successfully loaded. Please check the error messages above.")
                    else:
                        # Get all image paths
                        image_paths = get_image_paths(folder, recursive=recursive_search)
                        
                        if not image_paths:
                            st.warning("⚠️ No images found in the specified folder!")
                        else:
                            st.info(f"📁 Found **{len(image_paths)}** images")
                            
                            # Show folder statistics if auto-label is enabled
                            if auto_label:
                                # Count images by folder
                                folder_counts = {}
                                for img_path in image_paths:
                                    folder_name = img_path.parent.name
                                    folder_counts[folder_name] = folder_counts.get(folder_name, 0) + 1
                                
                                if folder_counts:
                                    st.write("**Images per folder:**")
                                    for folder_name, count in sorted(folder_counts.items()):
                                        st.write(f"  • {folder_name}: {count} images")
                            
                            # Estimate processing time
                            est_time_per_image = 3 if device == 'cuda' else 15  # seconds
                            total_est_time = (len(image_paths) * len(models) * est_time_per_image) / batch_size
                            st.info(f"⏱️ Estimated time: ~{total_est_time/60:.1f} minutes")
                            
                            # Process batch with progress bar
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            all_results = []
                            start_time = time.time()
                            
                            for model_idx, (model_name, model) in enumerate(models.items()):
                                status_text.text(f"Processing with {model_name.upper()} model ({model_idx+1}/{len(models)})...")
                                
                                try:
                                    def update_progress(p):
                                        overall_progress = (model_idx + p) / len(models)
                                        progress_bar.progress(overall_progress)
                                    
                                    results = predict_batch(
                                        image_paths,
                                        model,
                                        model_name,
                                        device,
                                        batch_size=batch_size,
                                        auto_label=auto_label,
                                        progress_callback=update_progress
                                    )
                                    all_results.extend(results)
                                except Exception as e:
                                    st.error(f"Error processing batch with {model_name}: {str(e)}")
                            
                            if not all_results:
                                st.error("❌ No predictions were generated. Please check the error messages.")
                            else:
                                elapsed_time = time.time() - start_time
                                progress_bar.progress(1.0)
                                status_text.text("✅ Processing complete!")
                                
                                # Convert to DataFrame
                                df = pd.DataFrame(all_results)
                                st.session_state.results_df = df
                                
                                # Show completion statistics
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Images Processed", len(image_paths))
                                with col2:
                                    st.metric("Total Predictions", len(df))
                                with col3:
                                    st.metric("Processing Time", f"{elapsed_time/60:.1f} min")
                                with col4:
                                    avg_time = elapsed_time / len(image_paths)
                                    st.metric("Avg Time/Image", f"{avg_time:.1f}s")
                                
                                st.success(f"✅ Successfully processed {len(image_paths)} images with {len(models)} model(s)!")
                                st.balloons()

with tab2:
    st.header("Results & Analysis")
    
    if st.session_state.results_df is not None:
        df = st.session_state.results_df
        
        # Define priority pathologies
        priority_pathologies = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Edema', 'Pneumonia', 'Pneumothorax']
        
        # Filter to priority pathologies
        df = df[df['pathology'].isin(priority_pathologies + ['Normal'])]
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Predictions", len(df))
        with col2:
            st.metric("Unique Images", df['filename'].nunique())
        with col3:
            st.metric("Models Used", df['model'].nunique())
        with col4:
            st.metric("Pathologies", df['pathology'].nunique())
        
        st.divider()
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_model_filter = st.multiselect(
                "Filter by Model",
                options=df['model'].unique(),
                default=df['model'].unique()
            )
        with col2:
            # Only show priority pathologies in filter
            available_pathologies = sorted([p for p in df['pathology'].unique() if p in priority_pathologies + ['Normal']])
            selected_pathology_filter = st.multiselect(
                "Filter by Pathology",
                options=available_pathologies,
                default=[],
                help="Showing only: Atelectasis, Cardiomegaly, Effusion, Edema, Pneumonia, Pneumothorax"
            )
        with col3:
            min_prob = st.slider("Minimum Probability", 0.0, 1.0, 0.0, 0.05)
        
        # Apply filters
        filtered_df = df[df['model'].isin(selected_model_filter)]
        if selected_pathology_filter:
            filtered_df = filtered_df[filtered_df['pathology'].isin(selected_pathology_filter)]
        filtered_df = filtered_df[filtered_df['probability'] >= min_prob]
        
        # Display table
        st.subheader("Prediction Results")
        st.dataframe(
            filtered_df.sort_values('probability', ascending=False),
            width='stretch',
            height=400
        )
        
        # Download CSV
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results CSV",
            data=csv,
            file_name=f"xray_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Visualizations
        st.divider()
        st.subheader("Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top pathologies by average probability
            st.markdown("**Top Pathologies (Avg Probability)**")
            top_pathologies = filtered_df.groupby('pathology')['probability'].mean().nlargest(10)
            
            fig = go.Figure(go.Bar(
                x=top_pathologies.values,
                y=top_pathologies.index,
                orientation='h',
                marker=dict(color='#1f77b4')
            ))
            fig.update_layout(
                xaxis_title="Average Probability",
                yaxis_title="Pathology",
                height=400,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            # Model comparison
            st.markdown("**Model Comparison (Avg Probability)**")
            model_avg = filtered_df.groupby('model')['probability'].mean()
            
            fig = go.Figure(go.Bar(
                x=model_avg.index,
                y=model_avg.values,
                marker=dict(color='#ff7f0e')
            ))
            fig.update_layout(
                xaxis_title="Model",
                yaxis_title="Average Probability",
                height=400,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig, width='stretch')
    else:
        st.info("👈 Upload images and run inference to see results here!")

with tab3:
    st.header("ROC/AUC Analysis")
    
    if st.session_state.results_df is not None:
        df = st.session_state.results_df
        
        # Define priority pathologies to show first
        priority_pathologies = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Edema', 'Pneumonia', 'Pneumothorax']
        
        # Filter dataframe to only show priority pathologies
        df_filtered = df[df['pathology'].isin(priority_pathologies)]
        
        # Check if we have ground truth labels
        if 'ground_truth' in df_filtered.columns and df_filtered['ground_truth'].notna().any():
            st.info("Ground truth labels detected!")
            
            # Get available pathologies (only those with ground truth data)
            available_pathologies = sorted([
                p for p in priority_pathologies 
                if p in df_filtered['pathology'].unique() and 
                df_filtered[df_filtered['pathology'] == p]['ground_truth'].notna().any()
            ])
            
            if not available_pathologies:
                st.warning("⚠️ No priority pathologies found with ground truth labels. Showing all available pathologies.")
                available_pathologies = sorted(df['pathology'].unique())
                df_filtered = df
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Add "All Models" option
                model_options = ['All Models'] + list(df_filtered['model'].unique())
                selected_model_roc = st.selectbox(
                    "Select Model",
                    options=model_options,
                    help="Select a specific model or 'All Models' to compare all models"
                )
            
            with col2:
                selected_pathology_roc = st.selectbox(
                    "Select Pathology",
                    options=available_pathologies,
                    help="Only showing: Atelectasis, Cardiomegaly, Effusion, Edema, Pneumonia, Pneumothorax"
                )
            
            with col3:
                st.write("")  # Spacer
            
            if st.button("📈 Generate ROC Curve", width='stretch'):
                with st.spinner("Computing ROC/AUC..."):
                    
                    if selected_model_roc == 'All Models':
                        # Compare all models for the selected pathology
                        st.subheader(f"Model Comparison - {selected_pathology_roc}")
                        
                        results_dict = {}
                        models_with_data = []
                        
                        for model_name in df_filtered['model'].unique():
                            roc_data = df_filtered[
                                (df_filtered['model'] == model_name) & 
                                (df_filtered['pathology'] == selected_pathology_roc) &
                                (df_filtered['ground_truth'].notna())
                            ]
                            
                            if len(roc_data) >= 10:
                                try:
                                    fpr, tpr, thresholds, auc_score = compute_roc_auc(
                                        roc_data['ground_truth'].values,
                                        roc_data['probability'].values
                                    )
                                    results_dict[model_name.upper()] = (fpr, tpr, auc_score)
                                    models_with_data.append(model_name)
                                except Exception as e:
                                    st.warning(f"Could not compute ROC for {model_name}: {str(e)}")
                        
                        if len(results_dict) == 0:
                            st.error("❌ Not enough data for any model. Need at least 10 samples per model.")
                        elif len(results_dict) == 1:
                            st.info("ℹ️ Only one model has sufficient data. Showing single ROC curve.")
                            model_name = list(results_dict.keys())[0]
                            fpr, tpr, auc_score = results_dict[model_name]
                            fig = plot_roc_curve(fpr, tpr, auc_score, f"{model_name} - {selected_pathology_roc}")
                            st.plotly_chart(fig, width='stretch')
                        else:
                            # Import the comparison function
                            from metrics import compare_models_roc
                            fig = compare_models_roc(results_dict)
                            st.plotly_chart(fig, width='stretch')
                            
                            # Show summary metrics
                            st.subheader("Model Performance Summary")
                            summary_data = []
                            for model_name in models_with_data:
                                roc_data = df_filtered[
                                    (df_filtered['model'] == model_name) & 
                                    (df_filtered['pathology'] == selected_pathology_roc) &
                                    (df_filtered['ground_truth'].notna())
                                ]
                                auc = results_dict[model_name.upper()][2]
                                summary_data.append({
                                    'Model': model_name.upper(),
                                    'AUC Score': f"{auc:.4f}",
                                    'Data Points': len(roc_data),
                                    'Positive Rate': f"{roc_data['ground_truth'].mean():.2%}"
                                })
                            
                            summary_df = pd.DataFrame(summary_data)
                            st.dataframe(summary_df, width='stretch', hide_index=True)
                    
                    else:
                        # Single model ROC curve
                        roc_data = df_filtered[
                            (df_filtered['model'] == selected_model_roc) & 
                            (df_filtered['pathology'] == selected_pathology_roc) &
                            (df_filtered['ground_truth'].notna())
                        ]
                        
                        if len(roc_data) < 10:
                            st.warning("⚠️ Not enough data points for ROC analysis (need at least 10)")
                        else:
                            try:
                                # Compute ROC
                                fpr, tpr, thresholds, auc_score = compute_roc_auc(
                                    roc_data['ground_truth'].values,
                                    roc_data['probability'].values
                                )
                                
                                # Plot
                                fig = plot_roc_curve(fpr, tpr, auc_score, 
                                                   f"{selected_model_roc.upper()} - {selected_pathology_roc}")
                                
                                st.plotly_chart(fig, width='stretch')
                                
                                # Metrics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("AUC Score", f"{auc_score:.4f}")
                                with col2:
                                    st.metric("Data Points", len(roc_data))
                                with col3:
                                    positive_rate = roc_data['ground_truth'].mean()
                                    st.metric("Positive Rate", f"{positive_rate:.2%}")
                                
                            except Exception as e:
                                st.error(f"Error computing ROC: {str(e)}")
        else:
            st.warning("""
            ⚠️ No ground truth labels found in the data.
            
            To enable ROC/AUC analysis:
            1. Organize images in folders by label (e.g., `pneumonia_positive`, `pneumonia_negative`)
            2. Enable "Auto-detect labels from folder names" option
            3. Process the batch
            
            Supported pathologies: Atelectasis, Cardiomegaly, Effusion, Edema, Pneumonia, Pneumothorax
            
            Or provide a CSV with ground truth labels.
            """)
    else:
        st.info("👈 Process images first to enable ROC/AUC analysis!")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Powered by TorchXRayVision | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)