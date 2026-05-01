from ultralytics import YOLO
import uuid
import os
from django.conf import settings

# load model once (important for performance)
MODEL_PATH = os.path.join(settings.BASE_DIR, "Models", "Face_Mask_Model.pt")
model = YOLO(MODEL_PATH)


def run_inference(image_path):
    results = model(image_path, conf=0.5)

    result = results[0]

    # Save annotated image
    output_filename = f"{uuid.uuid4()}.jpg"
    output_path = os.path.join(settings.MEDIA_ROOT, output_filename)

    result.save(filename=output_path)

    # Extract predictions
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        detections.append({
            "class": model.names[cls_id],
            "confidence": round(conf, 2)
        })

    return output_filename, detections