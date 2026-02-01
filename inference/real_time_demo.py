"""
Real-time Image Quality Assessment Demo using Streamlit.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.predictor import IQAPredictor
from models.deep_learning import get_model
from models.traditional.traditional_iqa import TraditionalIQA


st.set_page_config(
    page_title="Endoscopic IQA Demo",
    page_icon="🔬",
    layout="wide"
)


@st.cache_resource
def load_predictor(model_path=None):
    """Load IQA predictor (cached)."""
    try:
        if model_path and Path(model_path).exists():
            predictor = IQAPredictor(model_path=model_path)
        else:
            # Create dummy model for demo
            model = get_model(model_type='lightweight')
            predictor = IQAPredictor(model=model)
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_resource
def load_traditional_iqa():
    """Load traditional IQA metrics (cached)."""
    return TraditionalIQA()


def main():
    st.title("🔬 Endoscopic Image Quality Assessment")
    st.markdown("""
    This demo assesses the quality of endoscopic images using both traditional
    computer vision metrics and deep learning models.
    """)
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Model selection
    use_deep_learning = st.sidebar.checkbox("Use Deep Learning Model", value=True)
    use_traditional = st.sidebar.checkbox("Use Traditional Metrics", value=True)
    
    model_path = st.sidebar.text_input(
        "Model Path (optional)",
        help="Path to trained model checkpoint"
    )
    
    # Image source
    st.sidebar.title("Image Source")
    source = st.sidebar.radio(
        "Select source:",
        ["Upload Image", "Use Webcam", "Sample Images"]
    )
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    image = None
    
    with col1:
        st.subheader("Input Image")
        
        if source == "Upload Image":
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=['jpg', 'jpeg', 'png', 'bmp']
            )
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                image = np.array(image)
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        elif source == "Use Webcam":
            img_file_buffer = st.camera_input("Take a picture")
            if img_file_buffer is not None:
                image = Image.open(img_file_buffer)
                image = np.array(image)
        
        elif source == "Sample Images":
            st.info("Sample images would be loaded here in a full deployment")
            # In a real deployment, you would have sample endoscopic images
        
        if image is not None:
            st.image(image, caption="Input Image", use_column_width=True)
    
    with col2:
        st.subheader("Quality Assessment Results")
        
        if image is not None:
            with st.spinner("Analyzing image quality..."):
                results = {}
                
                # Deep learning prediction
                if use_deep_learning:
                    predictor = load_predictor(model_path if model_path else None)
                    if predictor:
                        try:
                            score, inference_time = predictor.predict(
                                image, return_time=True
                            )
                            results['Deep Learning Score'] = score
                            results['Inference Time (ms)'] = inference_time
                            
                            # Quality category
                            category = predictor.get_quality_category(score)
                            results['Quality Category'] = category
                        except Exception as e:
                            st.error(f"Deep learning prediction error: {e}")
                
                # Traditional metrics
                if use_traditional:
                    traditional_iqa = load_traditional_iqa()
                    try:
                        metrics = traditional_iqa.compute_all_metrics(image)
                        
                        # Also compute overall quality score
                        trad_score = traditional_iqa.compute_quality_score(image)
                        results['Traditional Score'] = trad_score
                        
                        # Add individual metrics
                        for key, value in metrics.items():
                            results[f"Traditional - {key}"] = value
                    except Exception as e:
                        st.error(f"Traditional metrics error: {e}")
                
                # Display results
                if results:
                    # Main scores
                    if 'Deep Learning Score' in results:
                        score = results['Deep Learning Score']
                        color = "green" if score > 0.7 else "orange" if score > 0.4 else "red"
                        st.markdown(f"### Deep Learning Quality Score")
                        st.markdown(f"# :{color}[{score:.3f}]")
                        st.markdown(f"**Category:** {results.get('Quality Category', 'N/A')}")
                        st.markdown(f"**Inference Time:** {results.get('Inference Time (ms)', 0):.2f} ms")
                    
                    if 'Traditional Score' in results:
                        score = results['Traditional Score']
                        st.markdown(f"### Traditional Quality Score")
                        st.markdown(f"# {score:.3f}")
                    
                    # Detailed metrics
                    st.markdown("---")
                    st.markdown("### Detailed Metrics")
                    
                    # Create metrics display
                    metric_cols = st.columns(2)
                    col_idx = 0
                    
                    for key, value in results.items():
                        if key not in ['Deep Learning Score', 'Traditional Score',
                                      'Quality Category', 'Inference Time (ms)']:
                            with metric_cols[col_idx % 2]:
                                if isinstance(value, float):
                                    st.metric(key, f"{value:.4f}")
                                else:
                                    st.metric(key, value)
                            col_idx += 1
                    
                    # Recommendations
                    st.markdown("---")
                    st.markdown("### Recommendations")
                    
                    if 'Deep Learning Score' in results:
                        score = results['Deep Learning Score']
                        if score < 0.4:
                            st.warning("⚠️ Poor quality detected. Consider:")
                            st.markdown("- Check focus and lighting")
                            st.markdown("- Reduce motion blur")
                            st.markdown("- Clean the lens")
                        elif score < 0.7:
                            st.info("ℹ️ Acceptable quality. Could be improved.")
                        else:
                            st.success("✅ Excellent image quality!")
        else:
            st.info("👆 Please select or upload an image to begin assessment")
    
    # Additional information
    st.markdown("---")
    with st.expander("ℹ️ About this demo"):
        st.markdown("""
        ### Endoscopic Image Quality Assessment
        
        This system evaluates endoscopic image quality using:
        
        **Deep Learning Approach:**
        - Lightweight CNN architecture optimized for real-time processing
        - Multi-scale feature extraction
        - Clinical-aware attention mechanisms
        - Quality score: 0 (poor) to 1 (excellent)
        
        **Traditional Metrics:**
        - Laplacian variance (blur detection)
        - Gradient energy (sharpness)
        - RMS contrast
        - Entropy (information content)
        - Noise estimation
        
        **Quality Categories:**
        - Excellent: 0.8 - 1.0
        - Good: 0.6 - 0.8
        - Fair: 0.4 - 0.6
        - Poor: 0.2 - 0.4
        - Bad: 0.0 - 0.2
        """)


if __name__ == "__main__":
    main()
