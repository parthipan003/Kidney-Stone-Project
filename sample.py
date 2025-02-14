import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO as kronecker
import gradio as gr
import pandas as pd

model_path = "best.pt"

# Load kronecker model
model = kronecker(model_path)


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

        

        color = colors[i % len(colors)]
        thickness = 2
        cv2.rectangle(img, (int(x1_expanded), int(y1_expanded)), (int(x2_expanded), int(y2_expanded)), color, thickness)
       
        cv2.putText(img, '', (int(x1_expanded), int(y1_expanded)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        color_hex = color_to_hex(color)
        color_swatch = f"■ {color_hex}"


    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    return img_pil
    

# Gradio Interface with built-in DataFrame component
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Image(type="pil"),  # Original image with bounding boxes

    ],
    live=True
)

iface.launch()
