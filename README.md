from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')  # Downloaded automatically on first run

vehicle_labels = [
    'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'train', 'van', 'taxi', 
    'trailer', 'boat', 'airplane', 'helicopter', 'scooter', 'pickup', 'minivan', 
    'ambulance', 'fire truck', 'police car', 'tractor', 'limo'
]


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    results = model(frame)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        if label in vehicle_labels:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Vehicle Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
