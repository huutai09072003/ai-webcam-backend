from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import cv2

# ✅ Dùng YOLOv8 từ ultralytics
from ultralytics import YOLO

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load both YOLO models
model_shape = YOLO("best.pt")        # bottle / can
model_material = YOLO("best2.pt")    # plastic / metal / glass

SHAPE_LABELS = model_shape.names
MATERIAL_LABELS = model_material.names

class ImageInput(BaseModel):
    image: str

# Mapping điểm
def get_point(shape, material):
    table = {
        ('bottle', 'plastic'): 5,
        ('bottle', 'glass'): 8,
        ('bottle', 'metal'): 10,
        ('can', 'plastic'): 4,
        ('can', 'glass'): 7,
        ('can', 'metal'): 9,
    }
    return table.get((shape, material), 0)

@app.post("/predict")
def predict(input: ImageInput):
    # Decode base64 image
    base64_data = input.image.split(",")[1]
    image_data = base64.b64decode(base64_data)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    img_array = np.array(image)

    # Detect shape (bottle / can)
    results_shape = model_shape(img_array, imgsz=640)[0]
    annotated_img = results_shape.plot()

    output = []
    total_points = 0

    if results_shape.boxes:
        for box in results_shape.boxes:
            cls_id = int(box.cls.item())
            shape_label = SHAPE_LABELS[cls_id]
            shape_conf = float(box.conf.item())

            if shape_conf < 0.7:
                continue  # Bỏ qua nếu không đủ tự tin

            xyxy = box.xyxy.cpu().numpy().astype(int)[0].tolist()
            x1, y1, x2, y2 = xyxy

            cropped = img_array[y1:y2, x1:x2]
            if cropped.size == 0:
                continue  # Tránh lỗi crop rỗng

            # Predict material
            results_material = model_material(cropped, imgsz=224)[0]
            material_label = "unknown"

            if results_material.boxes:
                best_box = max(results_material.boxes, key=lambda b: b.conf.item())
                material_conf = float(best_box.conf.item())

                if material_conf >= 0.7:
                    material_label = MATERIAL_LABELS[int(best_box.cls.item())]

            point = get_point(shape_label, material_label)
            total_points += point

            output.append({
                "label": shape_label,
                "material": material_label,
                "confidence": round(shape_conf, 3),
                "bbox": xyxy,
                "points": point
            })
    else:
        output = [{
            "label": "unknown",
            "material": "unknown",
            "confidence": 0,
            "bbox": [],
            "points": 0
        }]

    # Encode annotated image
    _, buffer = cv2.imencode(".jpg", annotated_img)
    encoded_img = base64.b64encode(buffer).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{encoded_img}"

    return {
        "detections": output,
        "total_points": total_points,
        "image": data_url
    }
