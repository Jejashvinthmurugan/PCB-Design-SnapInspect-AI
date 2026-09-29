from ultralytics import YOLO
import numpy as np


class PCBDetector:

    def __init__(self, model_path):

        self.model = YOLO(model_path)

    def detect(self, image, confidence=0.40):

        results = self.model.predict(
            source=image,
            conf=confidence,
            imgsz=640,
            verbose=False
        )

        result = results[0]

        detections = []

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(box.cls[0])

                confidence_score = float(
                    box.conf[0]
                )

                class_name = self.model.names[
                    class_id
                ]

                coordinates = box.xyxy[0].tolist()

                x1, y1, x2, y2 = map(
                    int,
                    coordinates
                )

                detections.append({
                    "defect": class_name,
                    "confidence": round(
                        confidence_score * 100,
                        2
                    ),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                })

        annotated = result.plot()

        annotated = np.array(
            annotated
        )[:, :, ::-1]

        return annotated, detections
