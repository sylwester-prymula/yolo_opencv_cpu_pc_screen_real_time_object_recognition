import numpy as np
import cv2
from PIL import ImageGrab

# Load Yolo
yolo_cfg = r'.\\cfg'
net = cv2.dnn.readNet(yolo_cfg + '\\yolov3.weights', yolo_cfg + '\\yolov3.cfg')
with open(yolo_cfg + '\\coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
    print(classes)
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
while True:
    # Capture computer screen
    # img1 = ImageGrab.grab(bbox=(960, 0, 1980, 1080))
    img1 = ImageGrab.grab()

    # Convert image to numpy array
    img_np = np.array(img1)
    # Convert color space from BGR to RGB
    img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing information on the screen
    height, width, channels = img.shape
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[int(class_ids[i])])
            print(label + ': ' + str(confidences[i]))
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            confidence = confidences[i]
            cv2.putText(img, label + ' ' + str(round(confidence, 2)), (x, y + 30), font, 2, color, 2)
    cv2.imshow('Image', img)
    print(' ')
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
