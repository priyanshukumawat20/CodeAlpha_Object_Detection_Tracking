import cv2
from ultralytics import YOLO

# Load YOLO model (pre-trained)
model = YOLO('yolov8n.pt')

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run object detection on the frame
    results = model(frame, verbose=False)

    # Draw bounding boxes and labels on the frame
    annotated_frame = results[0].plot()

    # Display the result
    cv2.imshow('Object Detection and Tracking', annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
