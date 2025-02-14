import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import gradio as gr
import pandas as pd

model_path = "best.pt"

# Load YOLO model
model = YOLO(model_path)

def classify_stone_type(hu_value):
    """Classify the stone type based on the Hounsfield Units."""
    if 700 <= hu_value <= 1200:
        return "Calcium Oxalate"
    elif 300 <= hu_value <= 500:
        return "Uric Acid"
    elif 1000 <= hu_value <= 1200:
        return "Struvite"
    elif 200 <= hu_value <= 400:
        return "Cystine"
    else:
        return "Unknown"

def calculate_HU(cropped_image):
    """Calculate the Hounsfield Units (HU) based on pixel intensity."""
    if len(cropped_image.shape) > 2:
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    cropped_image = cropped_image.astype(np.uint16)
    max_intensity = np.max(cropped_image)
    hu_values = (cropped_image / max_intensity) * 4095 - 1024
    mean_hu = np.mean(hu_values) if hu_values.size > 0 else -1024

    masked_image = cv2.bitwise_and(cropped_image, cropped_image, mask=(cropped_image > 200).astype(np.uint8))

    return mean_hu, masked_image

def calculate_stone_size(masked_image):
    """Calculate the size of the stone based on white intensity regions."""
    masked_image_8bit = cv2.convertScaleAbs(masked_image)  # Convert to 8-bit image
    contours, _ = cv2.findContours(masked_image_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        x, y, w, h = cv2.boundingRect(largest_contour)
        diameter = max(w, h)
        return area, diameter, largest_contour
    return 0, 0, None

def color_to_hex(color):
    """Convert an RGB tuple to a hexadecimal color code."""
    return '#%02x%02x%02x' % color

def predict_image(image):
    results = model(image)
    boxes = results[0].boxes
    names = results[0].names
    confidences = boxes.conf
    xywh = boxes.xywh

    img = np.array(image)
    predictions = []
    masked_images = []

    # More colors for bounding boxes
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), 
        (255, 255, 0), (255, 0, 255), (0, 255, 255), 
        (128, 0, 0), (0, 128, 0), (0, 0, 128)
    ]

    for i, (x, y, w, h) in enumerate(xywh):
        class_id = int(boxes.cls[i].item())
        class_name = names[class_id]
        confidence = confidences[i].item()
        x1, y1, x2, y2 = x.item(), y.item(), (x + w).item(), (y + h).item()
        margin = 15
        x1_expanded = max(0, x1 - margin)
        y1_expanded = max(0, y1 - margin)
        x2_expanded = x2 + margin
        y2_expanded = y2 + margin
        cropped_image = img[int(y1_expanded):int(y2_expanded), int(x1_expanded):int(x2_expanded)]

        mean_hu, masked_image = calculate_HU(cropped_image)
        stone_type = classify_stone_type(mean_hu)
        area, diameter, contour = calculate_stone_size(masked_image)

        color = colors[i % len(colors)]
        thickness = 2
        cv2.rectangle(img, (int(x1_expanded), int(y1_expanded)), (int(x2_expanded), int(y2_expanded)), color, thickness)
        label = f"{stone_type}: {confidence:.2f}, HU: {mean_hu:.2f}, Size: {diameter}mm"
        cv2.putText(img, '', (int(x1_expanded), int(y1_expanded)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        color_hex = color_to_hex(color)
        color_swatch = f"■ {color_hex}"

        predictions.append({
            "Stone Type": stone_type,
            "Confidence": f"{confidence*100:.2f}%",
            "HU": f"{mean_hu:.2f}",
            "Size (mm)": f"{diameter:.2f}",
            "Color": color_swatch  # Display as Unicode block with hex code
        })

        if contour is not None:
            stone_image = cv2.drawContours(cropped_image.copy(), [contour], -1, (0, 255, 0), 2)
            masked_images.append(Image.fromarray(stone_image.astype(np.uint8)))

    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    df = pd.DataFrame(predictions)
    # return img_pil
    return img_pil, df, masked_images

# Gradio Interface with built-in DataFrame component
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Image(type="pil"),  # Original image with bounding boxes
        gr.Dataframe(headers=["Stone Type", "Confidence", "HU", "Size (mm)", "Color"], type="pandas"),
        gr.Gallery(label="High-Intensity Regions (Stone Only)")  # Display high-intensity regions
    ],
    live=True
)

iface.launch()
