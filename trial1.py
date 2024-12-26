import streamlit as st
import numpy as np
import cv2
import os
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from datetime import datetime

# Ensure directories for saving images exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("predictions", exist_ok=True)
os.makedirs("ground_truths", exist_ok=True)

# Load the trained model
@st.cache_resource
def load_unet_model():
    return load_model('final_UNET_Butterfly_Segmentation.keras', compile=False)

unet_model = load_unet_model()



# Helper functions
def predict_single_image(image, model):
    """
    Predicts the segmentation mask for a single image and times the prediction.

    Args:
        image (array): Input image for prediction.
        model (keras.Model): Trained U-Net model.

    Returns:
        tuple: Predicted mask and inference time.
    """
    image = image.astype('float32') / 255.0  # Ensure image is float32 and normalized for prediction
    start_time = time.time()
    prediction = model.predict(np.expand_dims(image, axis=0))[0, :, :, 0]
    end_time = time.time()
    inference_time = end_time - start_time

    return prediction, inference_time

def visualize_segmentation(image, predicted_mask, ground_truth_mask=None):
    """
    Visualizes the original image, predicted mask, and ground truth mask.

    Args:
        image (array): Original image.
        predicted_mask (array): Predicted mask.
        ground_truth_mask (array, optional): Ground truth mask.
    """
    plt.figure(figsize=(16, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')

    plt.subplot(1, 3, 2)
    plt.imshow(predicted_mask, cmap='gray')
    plt.title('Predicted Mask')

    if ground_truth_mask is not None:
        plt.subplot(1, 3, 3)
        plt.imshow(ground_truth_mask, cmap='gray')
        plt.title('Ground Truth Mask')

    st.pyplot(plt)

def dice_coefficient(y_true, y_pred, smooth=1):
    """
    Calculate the Dice coefficient.

    Args:
        y_true (array): Ground truth mask.
        y_pred (array): Predicted mask.

    Returns:
        float: Dice coefficient score.
    """
    y_true_f = K.flatten(K.cast(y_true, 'float32'))
    y_pred_f = K.flatten(K.cast(y_pred, 'float32'))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def mean_iou(y_true, y_pred, smooth=1):
    """
    Calculate the mean Intersection over Union (IoU).

    Args:
        y_true (array): Ground truth mask.
        y_pred (array): Predicted mask.

    Returns:
        float: Mean IoU score.
    """
    y_true_f = K.flatten(K.cast(y_true, 'float32'))
    y_pred_f = K.flatten(K.cast(y_pred, 'float32'))
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# Streamlit UI
st.title("Butterfly Image Segmentation with U-Net")

uploaded_image = st.file_uploader("Upload a butterfly image for segmentation", type=["jpg", "png", "jpeg"])
ground_truth_image = st.file_uploader("Upload the ground truth mask (optional)", type=["jpg", "png", "jpeg"])

if uploaded_image:
    # Generate unique timestamped filename for saving
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load and save the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image_resized = cv2.resize(image, (256, 256))  # Resize image

    # Save the original uploaded image
    upload_path = f"uploads/image_{timestamp}.png"
    cv2.imwrite(upload_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # Convert back to BGR for saving

    # Predict the segmentation mask
    predicted_mask, inference_time = predict_single_image(image_resized, unet_model)

    # Save the predicted mask
    prediction_path = f"predictions/prediction_{timestamp}.png"
    cv2.imwrite(prediction_path, (predicted_mask * 255).astype(np.uint8))  # Scale mask to uint8 for saving

    # Show inference time
    st.write(f"Inference time: {inference_time:.4f} seconds")

    # Display segmentation
    if ground_truth_image:
        # Load and save the ground truth mask if provided
        gt_file_bytes = np.asarray(bytearray(ground_truth_image.read()), dtype=np.uint8)
        ground_truth = cv2.imdecode(gt_file_bytes, 0)  # Read as grayscale
        ground_truth_resized = cv2.resize(ground_truth, (256, 256)).astype('float32') / 255.0  # Resize and normalize
        
        # Save ground truth mask
        ground_truth_path = f"ground_truths/ground_truth_{timestamp}.png"
        cv2.imwrite(ground_truth_path, (ground_truth_resized * 255).astype(np.uint8))
        
        # Thank the user for providing the ground truth mask
        st.write("Thank you for providing the ground truth mask!")
    else:
        ground_truth_resized = None

    # Visualize results
    st.subheader("Segmentation Results")
    visualize_segmentation(image_resized, predicted_mask, ground_truth_resized)

    # Calculate and display evaluation metrics if ground truth is available
    if ground_truth_resized is not None:
        dice_score = dice_coefficient(ground_truth_resized, predicted_mask).numpy()
        mean_iou_score = mean_iou(ground_truth_resized, predicted_mask).numpy()
