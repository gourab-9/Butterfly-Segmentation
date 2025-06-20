# 🦋 Butterfly Image Segmentation App using U-Net

A **Streamlit web app** for semantic segmentation of butterfly images using a **pre-trained U-Net model**. Users can upload butterfly images and optionally ground truth masks to visualize predicted segmentations and evaluate performance with metrics like **Dice Coefficient** and **Mean IoU**.

---

## 📸 Features

- Upload a butterfly image for segmentation
- Upload ground truth mask (optional) for evaluation
- Predict segmentation masks using a pre-trained U-Net model
- Visualize original image, predicted mask, and ground truth mask side-by-side
- Calculate Dice Coefficient and Mean IoU for evaluation
- Displays inference time
- Saves uploaded images, predictions, and masks with timestamps

---

## 🗂️ Directory Structure

```text
.
├── app.py # Streamlit app script
├── final_UNET_Butterfly_Segmentation.keras # Pre-trained U-Net model
├── requirements.txt # Python dependencies
├── uploads/ # Uploaded butterfly images
├── predictions/ # Predicted segmentation masks
└── ground_truths/ # Uploaded ground truth masks
```

---

## 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/gourab-9/butterfly-segmentation-unet.git
cd butterfly-segmentation-unet

```
2. Install the required packages:

```bash
pip install -r requirements.txt
```
🧠 Run the App

```bash
streamlit run app.py
```

Make sure `final_UNET_Butterfly_Segmentation.keras` is in the same directory as `app.py`.

## 🧪 Model
- The model is a U-Net trained for binary segmentation of butterfly images.
- Input image size is 256x256x3.
- Output is a grayscale mask of size 256x256, with pixel values normalized between 0 and 1.

## 📋 Example Usage

1. **Upload** a butterfly image (JPG/PNG).
2. *(Optional)* **Upload the ground truth mask** for the same image.
3. The app will:
   - ✅ Predict the segmentation
   - ⏱️ Show inference time
   - 🖼️ Display original image, predicted mask, and ground truth mask
   - 📐 Calculate **Dice Score** and **IoU** *(if ground truth is provided)*

---

## 📊 Metrics

- **Dice Coefficient**  
  Measures the overlap between predicted and ground truth masks.

  $$
  \text{Dice} = \frac{2 \cdot |A \cap B|}{|A| + |B|}
  $$

- **Mean Intersection over Union (IoU)**  
  Measures the intersection over union between prediction and ground truth.

  $$
  \text{IoU} = \frac{|A \cap B|}{|A \cup B|}
  $$

---

## 📁 Saved Outputs

The app automatically saves the following files:

- `uploads/image_<timestamp>.png` — Original uploaded image
- `predictions/prediction_<timestamp>.png` — Predicted mask
- `ground_truths/ground_truth_<timestamp>.png` — Ground truth mask (if provided)

## Model Output
![Output Image](https://github.com/user-attachments/assets/b27e5367-1629-485b-9373-b781382fc9a3)

