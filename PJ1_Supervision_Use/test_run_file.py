import cv2
from ultralytics import YOLO
import os
import supervision as sv

image_path = os.path.join('.','data','IMG-9266.jpg')

# create prediction model
model = YOLO("yolov8n.pt")
image = cv2.imread(image_path)
results = model(image)[0]

#  load into supervision to create and array[x y w h cfs cls track_id]
detections = sv.Detections.from_ultralytics(results)
print(detections)

# create Bbox using supervision and label for image
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

labels = [
    results.names[class_id]
    for class_id
    in detections.class_id
]

# bbox an image
annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
# label an image
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections, labels=labels)

sv.plot_image(annotated_image)