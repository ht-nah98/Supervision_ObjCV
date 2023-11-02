import numpy as np
import supervision as sv
from ultralytics import YOLO
import os
import cv2


video_path = os.path.join('.', 'data', 'morning_12.mp4')
weight = os.path.join('.', 'model_weights', 'last.pt')
model = YOLO(weight)

# using model.model.names to get all classes of the model

tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()  # create Bbox for obj
label_annotator = sv.LabelAnnotator()  # create label on Bbox
trace_annotator = sv.TraceAnnotator()  # crate trace for Bbox
line_annotator = sv.LineZoneAnnotator()


def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)  # detection format array[x y w h cfs cls track_id]

    labels = [
        # f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
        # for _, _, confidence, class_id, tracker_id in detections
    ]

    annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
    cv2.imshow("yolov8", frame)

    if (cv2.waitKey(30) == 27):
        pass
    return frame


sv.process_video(
    source_path=video_path,
    target_path=f"result3.mp4",
    callback=callback
)
