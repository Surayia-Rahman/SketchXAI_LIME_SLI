import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from streamlit_drawable_canvas import st_canvas
from skimage.segmentation import quickshift
from sklearn.linear_model import Ridge
import os

# 1. MAIN INTERFACE CONFIGURATIONS
st.set_page_config(page_title="SketchXAI Inference", layout="wide")
st.title("SketchXAI: Spatiotemporal Explainability Inference")
st.markdown("---")


@st.cache_resource
def initialize_cached_network():
    # Dynamic Label Engine: Read directly from labels.txt file target
    if os.path.exists("labels.txt"):
        with open("labels.txt", "r") as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
    else:
        st.error(" **Critical Configuration Error:** `labels.txt` was not found in the root directory. Please generate it to enable multi-class network mapping.")
        st.stop()
        
    if len(classes) == 0:
        st.error(" **Critical Configuration Error:** `labels.txt` is empty. Please populate your target classes roster.")
        st.stop()
        
    # Build the Neural Network dynamically sized to your exact number of labels
    net = Sequential([
        Conv2D(30, (3, 3), activation='relu', input_shape=(28, 28, 1), name="Conv_Block_1"),
        MaxPooling2D((2, 2), name="MaxPool_1"),
        Conv2D(15, (3, 3), activation='relu', name="Conv_Block_2"),
        MaxPooling2D((2, 2), name="MaxPool_2"),
        Dropout(0.2, name="Overfitting_Guard_Dropout"),
        Flatten(name="Vector_Flatten"),
        Dense(128, activation='relu', name="Dense_Hidden_1"),
        Dense(50, activation='relu', name="Dense_Hidden_2"),
        Dense(len(classes), activation='softmax', name="Softmax_Output")
    ])
    
    # Load compiled production model weights file
    if os.path.exists("sketchxai_matching_model.h5"):
        try:
            net.load_weights("sketchxai_matching_model.h5")
        except Exception as e:
            st.error(f" **Weights Dimension Collision:** Could not fit weights file to network configuration. Ensure your local weights model matches your `labels.txt` count ({len(classes)} classes).")
            st.stop()
    else:
        st.warning(" `sketchxai_matching_model.h5` weights file not found. Running inference engine on uncalibrated randomized vectors.")
        
    return net, classes

# Execute dynamic startup pass
model, target_classes = initialize_cached_network()

# Render status confirmation pill in Streamlit sidebar room
st.sidebar.success(f"📊 Engine Active: Loaded `{len(target_classes)}` classes from `labels.txt` successfully.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Draw Your Sketch Below:")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=14, 
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=350,
        width=350,
        drawing_mode="freedraw",
        key="xai_canvas",
        update_streamlit=True
    )

with col2:
    st.subheader("XAI Prediction & Analytics Engine:")
    
    # Check if drawing strokes physically exist on the board
    if canvas_result.image_data is not None and np.any(canvas_result.image_data[:, :, 3] > 0):
        
        # Extract the standard RGB channels (ignoring the volatile alpha channel entirely)
        rgb_channels = canvas_result.image_data[:, :, :3]
        
        # Convert the high-res frame to grayscale
        gray_canvas = cv2.cvtColor(rgb_channels, cv2.COLOR_RGB2GRAY)
        
        # Apply binary thresholding: turn the black sketch ink into clean, bright white pixels (255)
        # and turn the white canvas background into crisp, clean neural-network black (0)
        _, binary_ink_mask = cv2.threshold(gray_canvas, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Downscale the inverted binary image down to QuickDraw's native 28x28 size
        resized_img = cv2.resize(binary_ink_mask, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Strict activity guard check
        if np.max(resized_img) == 0:
            st.info("Awaiting active canvas draw commands to initiate spatiotemporal XAI algorithms.")
        else:
            normalized_img = resized_img.astype('float32') / 255.0
            input_tensor = np.expand_dims(np.expand_dims(normalized_img, axis=0), axis=-1)
            
            # --- BASELINE INFERENCE ---
            predictions = model.predict(input_tensor)[0]
            best_class_idx = np.argmax(predictions)
            confidence_base = predictions[best_class_idx]
            
            st.metric(
                label="Predicted Target Class:", 
                value=target_classes[best_class_idx].upper(), 
                delta=f"{confidence_base * 100:.2f}% Baseline System Confidence"
            )
        
            # Fetch the physical timeline objects from the drawing board
            stroke_objects = canvas_result.json_data.get("objects", []) if canvas_result.json_data else []
            num_strokes = len(stroke_objects)
            
            # Create clear layout partitions for our two separate XAI tabs
            tab1, tab2 = st.tabs(["📊 Spatial Explainability (LIME)", "⏱️ Temporal Explainability (SLI Engine)"])
            

            # TAB 1: FROM-SCRATCH SPATIAL LIME ENGINE
            with tab1:
                with st.spinner("Executing spatial superpixel perturbations..."):
                    img_3ch = cv2.merge([resized_img, resized_img, resized_img])
                    segments = quickshift(img_3ch, kernel_size=1, max_dist=4, ratio=0.5)
                    num_segments = len(np.unique(segments))
                    
                    num_samples = 150
                    perturbations = np.random.randint(0, 2, size=(num_samples, num_segments))
                    batch_perturbed_images = np.zeros((num_samples, 28, 28, 1), dtype='float32')
                    
                    for idx, sample in enumerate(perturbations):
                        mask = np.zeros(segments.shape, dtype=bool)
                        for i, active in enumerate(sample):
                            if active == 1:
                                mask[segments == i] = True
                        
                        perturbed_img = resized_img.copy()
                        perturbed_img[~mask] = 0 
                        batch_perturbed_images[idx] = np.expand_dims(perturbed_img.astype('float32') / 255.0, axis=-1)
                    
                    all_perturbed_predictions = model.predict(batch_perturbed_images, verbose=0)
                    perturbed_predictions = all_perturbed_predictions[:, best_class_idx]
                    
                    lr = Ridge(alpha=1.0)
                    lr.fit(perturbations, perturbed_predictions)
                    importances = lr.coef_
                    
                    explanation_mask = np.zeros(resized_img.shape, dtype=float)
                    for i, coef in enumerate(importances):
                        explanation_mask[segments == i] = coef
                    
                    explanation_mask = (explanation_mask - np.min(explanation_mask)) / (np.max(explanation_mask) - np.min(explanation_mask) + 1e-5)
                    explanation_mask_rescaled = (explanation_mask * 255).astype(np.uint8)
                    explanation_mask_visual = cv2.resize(explanation_mask_rescaled, (150, 150), interpolation=cv2.INTER_NEAREST)
                
                col_preview, col_mask = st.columns(2)
                with col_preview:
                    st.image(resized_img, caption="Network Input Matrix (28x28)", width=150, channels="L")
                with col_mask:
                    st.image(explanation_mask_visual, caption="LIME Explanation Map", width=150, channels="L")
                st.caption("💡 Bright regions point to the specific pixel locations that anchored this classification.")

         
            # TAB 2: FROM-SCRATCH TEMPORAL SLI ENGINE
            with tab2:
                st.write(f"**Total Sequential Strokes Captured in Timeline:** `{num_strokes}`")
                
                if num_strokes < 2:
                    st.info("Draw at least 2 distinct line strokes on the canvas board to enable temporal elimination analytics.")
                else:
                    with st.spinner("Analyzing stroke sequence drop impacts..."):
                        stroke_drops = []
                        
                        # Temporarily drop one stroke index at a time to check inverse impact
                        for drop_idx in range(num_strokes):
                            # Setup a blank canvas mirroring our real display layout dimensions
                            temp_canvas = np.zeros((350, 350), dtype=np.uint8)
                            
                            # Re-render every historical line path EXCEPT the current dropped stroke
                            for idx, stroke in enumerate(stroke_objects):
                                if idx == drop_idx:
                                    continue
                                
                                # Parse the JSON vector points to rebuild the shape paths
                                path_points = stroke.get("path", [])
                                for i in range(len(path_points) - 1):
                                    if len(path_points[i]) >= 3 and len(path_points[i+1]) >= 3:
                                        pt1 = (int(path_points[i][-2]), int(path_points[i][-1]))
                                        pt2 = (int(path_points[i+1][-2]), int(path_points[i+1][-1]))
                                        cv2.line(temp_canvas, pt1, pt2, 255, thickness=14)
                            
                            # Process our newly dropped variation tensor shape
                            t_resized = cv2.resize(temp_canvas, (28, 28), interpolation=cv2.INTER_AREA)
                            t_norm = t_resized.astype('float32') / 255.0
                            t_tensor = np.expand_dims(np.expand_dims(t_norm, axis=0), axis=-1)
                            
                            # Check confidence drop magnitude
                            drop_pred = model.predict(t_tensor, verbose=0)[0][best_class_idx]
                            confidence_drop = max(0.0, confidence_base - drop_pred)
                            stroke_drops.append(confidence_drop)
                        
                        # Convert raw drops into a clean, normalized importance percentage layout
                        stroke_drops = np.array(stroke_drops)
                        total_drop_sum = np.sum(stroke_drops) + 1e-5
                        normalized_sli_scores = (stroke_drops / total_drop_sum) * 100
                    
                    # Render an interactive metrics chart showing the importance data of each stroke
                    st.write("### Spatiotemporal Stroke Rankings")
                    for idx, score in enumerate(normalized_sli_scores):
                        st.progress(float(score / 100.0))
                        st.caption(f"✍️ **Stroke Sequence #{idx + 1}:** Accounted for **{score:.2f}%** of model tracking stability.")
                        
    else:
        st.info("Awaiting active canvas draw commands to initiate spatiotemporal XAI algorithms.")