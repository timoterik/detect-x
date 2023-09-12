import cv2

# Load the pre-trained YOLOv3 model and COCO dataset labels
net = cv2.dnn.readNet("var/yolov3.weights", "var/yolov3.cfg")
with open("var/coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Initialize the video capture
cap = cv2.VideoCapture(0)  # 0 for the default camera, change to a different index if needed

while True:
    ret, frame = cap.read()  # Read a frame from the camera

    if not ret:
        break

    # Perform object detection
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()

    # Process detected objects
    for detection in detections:
        for obj in detection:
            # Ensure that the object has enough elements to access scores
            if len(obj) >= 7:  # Check the length to ensure there are enough elements for confidence, x, y, w, h, and 2 additional values
                scores = obj[5:]
                class_id = scores.argmax()
                confidence = scores[class_id]

                if confidence > 0.5:  # You can adjust this confidence threshold
                    center_x = int(obj[0] * frame.shape[1])
                    center_y = int(obj[1] * frame.shape[0])
                    width = int(obj[2] * frame.shape[1])
                    height = int(obj[3] * frame.shape[0])

                    # Calculate coordinates for drawing the bounding box
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    label = f"{classes[class_id]}: {confidence:.2f}"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with detected objects
    cv2.imshow("Object Detection", frame)

    # Check for the 'q' key press and exit if pressed
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
