import cvzone
from ultralytics import YOLO
import cv2
import math
import os
import supervision as sv

video_path = os.path.join('.', 'data', 'morning_12.mp4')
weight = os.path.join('.', 'model_weights', 'last.pt')
# MASK = os.path.join('.', 'mask', 'mask2.png')

cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# initialize for video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('counter_vehicle_1.mp4', fourcc, fps, (frame_width, frame_height))

# create tracker by supervision
tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()  # create Bbox for obj
label_annotator = sv.LabelAnnotator()  # create label on Bbox
trace_annotator = sv.TraceAnnotator()  # crate trace for Bbox
model = YOLO(weight)

limits = [500, 100, 500, 350]
count_moto = []
count_car = []

while True:
    success, img = cap.read()
    # image_region = cv2.bitwise_and(img, MASK)
    results = model(img)[0]

    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    labels = [
        # f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
        # for _, _, confidence, class_id, tracker_id in detections
    ]

    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)

    cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 3)
    for result in detections:
        # print(result)
        x1, y1, x2, y2 = result[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        ID = result[-1]
        class_id = result[-2]

        w, h = x2-x1, y2-y1
        # bbox= x1, y1, w, h

        # create center point for each bbox
        cx, cy = x1 + w//2, y1+h//2

        cv2.circle(frame, (cx,cy), 5, (255, 255, 0), cv2.FILLED)
        if 480<cx<520 and 100<cy<350:
            if count_moto.count(ID) == 0 and class_id==0:
                count_moto.append(ID)
                cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 3)
            if count_car.count(ID) == 0 and class_id==1:
                count_car.append(ID)
                cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 3)

    cvzone.putTextRect(frame, f'moto:{len(count_moto)}', pos=(50,50), thickness=1,scale=2)
    cvzone.putTextRect(frame, f'car:{len(count_car)}', pos=(400, 50), thickness=1,scale=2)


    cv2.imshow('Image', frame)
    output.write(frame)
    if cv2.waitKey(60) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
