import cv2
import numpy as np
import urllib.request
import os

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, filename)
            print(f"Downloaded {filename}")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
            print(f"Please download {filename} manually from {url} and place it in the current directory.")
            return False
    return True

def main():
    # URLs for YOLOv3 model files
    config_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3.cfg"
    weights_url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov3.weights"
    classes_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

    config = "yolov3.cfg"
    weights = "yolov3.weights"
    classes_file = "coco.names"

    # Download model files if not present
    config_ok = download_file(config_url, config)
    weights_ok = download_file(weights_url, weights)
    classes_ok = download_file(classes_url, classes_file)

    if not (config_ok and weights_ok and classes_ok):
        print("Some model files could not be downloaded. Please download them manually and try again.")
        return

    # Load the pre-trained YOLOv3 model
    net = cv2.dnn.readNetFromDarknet(config, weights)

    # Load class labels
    with open(classes_file, 'r') as f:
        CLASSES = [line.strip() for line in f.readlines()]

    # Initialize video capture from webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting real-time object detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Get frame dimensions
        (h, w) = frame.shape[:2]

        # Prepare the frame for the neural network
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        # Pass the blob through the network
        net.setInput(blob)
        outs = net.forward(net.getUnconnectedOutLayersNames())

        # Initialize lists for detected bounding boxes, confidences, and class IDs
        class_ids = []
        confidences = []
        boxes = []

        # Loop over each output layer
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * w)
                    center_y = int(detection[1] * h)
                    width = int(detection[2] * w)
                    height = int(detection[3] * h)

                    # Rectangle coordinates
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)

                    boxes.append([x, y, width, height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maximum suppression to remove overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Draw the bounding boxes and labels
        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                x, y, w, h = box
                label = str(CLASSES[class_ids[i]])
                confidence = confidences[i]
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the frame
        cv2.imshow("Real-Time Object Tracker", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()