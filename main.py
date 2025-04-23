from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import cv2

from ultralytics import YOLO

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("type_newest_ver.pt")
NAMES = model.names if hasattr(model, "names") else ["Object"]

class ImageInput(BaseModel):
    image: str

@app.post("/predict")
def predict(input: ImageInput):
    # Decode base64 image
    base64_data = input.image.split(",")[1]
    image_data = base64.b64decode(base64_data)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    img_array = np.array(image)

    # Detect
    results = model(img_array, imgsz=640)[0]
    img_annotated = results.plot()

    # Parse results
    types = []
    detections = []

    if results.boxes:
        for box in results.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            label = NAMES[cls_id]
            types.append(label)

            xyxy = box.xyxy.cpu().numpy().astype(int)[0].tolist()
            detections.append({
                "label": label,
                "confidence": round(conf, 3),
                "bbox": xyxy  # [x1, y1, x2, y2]
            })
    else:
        types = ["Unknown"]

    # Encode annotated image to base64
    _, buffer = cv2.imencode(".jpg", img_annotated)
    encoded_img = base64.b64encode(buffer).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{encoded_img}"

    return {
        "types": types,
        "detections": detections,
        "image": data_url
    }
