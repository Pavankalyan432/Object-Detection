import numpy as np
import time
import cv2
import os
import imutils
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import pyttsx3
from pytesseract import pytesseract

YOLO_PATH = "yolo"

labelsPath = os.path.sep.join([YOLO_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
CONFIDENCE = 0.3
THRESHOLD = 0.3
engine = pyttsx3.init()

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

weightsPath = os.path.sep.join([YOLO_PATH, "yolov3.weights"])
configPath = os.path.sep.join([YOLO_PATH, "yolov3.cfg"])

print("[INFO] loading YOLO from disk...")
net1 = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net1.getLayerNames()
ln = [ln[i - 1] for i in net1.getUnconnectedOutLayers()]
print("[INFO] loaded YOLO...")

prevText = ""

print("[INFO] loading Webcam...")
camera = cv2.VideoCapture(0)
time.sleep(2.0)
fps = FPS().start()

(W, H) = (None, None)
new_w, new_h = 320, 320
ratio_w, ratio_h = None, None
layer_names = ['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3']

print("[INFO] loading EAST text detector...")
net2 = cv2.dnn.readNet("frozen_east_text_detection.pb")
print("[INFO] Loading EAST text detector Complete...")

tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.tesseract_cmd = tesseract_path

wordsRecognized = []


def box_extractor(scores, geometry, min_confidence):
    num_rows, num_cols = scores.shape[2:4]
    rectangles = []
    confidences = []

    for y in range(num_rows):
        scores_data = scores[0, 0, y]
        x_data0 = geometry[0, 0, y]
        x_data1 = geometry[0, 1, y]
        x_data2 = geometry[0, 2, y]
        x_data3 = geometry[0, 3, y]
        angles_data = geometry[0, 4, y]

        for x in range(num_cols):
            if scores_data[x] < min_confidence:
                continue

            offset_x, offset_y = x * 4.0, y * 4.0

            angle = angles_data[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            box_h = x_data0[x] + x_data2[x]
            box_w = x_data1[x] + x_data3[x]

            end_x = int(offset_x + (cos * x_data1[x]) + (sin * x_data2[x]))
            end_y = int(offset_y + (cos * x_data2[x]) - (sin * x_data1[x]))
            start_x = int(end_x - box_w)
            start_y = int(end_y - box_h)

            rectangles.append((start_x, start_y, end_x, end_y))
            confidences.append(scores_data[x])

    return rectangles, confidences

try:
    while True:
        (grabbed, frame) = camera.read()

        if not grabbed:
            break
        if W is None or H is None:
            (H, W) = frame.shape[:2]
            ratio_w = W / float(new_w)
            ratio_h = H / float(new_h)

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net1.setInput(blob)
        start = time.time()
        layerOutputs = net1.forward(ln)
        end = time.time()

        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > CONFIDENCE:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE,
                                THRESHOLD)

        objects = set()
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                                           confidences[i])
                cv2.putText(frame, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                objects.add(LABELS[classIDs[i]])
            text = ""
            if len(objects) > 0:
                text = "Live image contains "
            for obj in objects:
                text += obj + ", "
            if prevText != text:
                prevText = text
                engine.say(text)
                engine.runAndWait()

        # Text recognition code
        image = imutils.resize(frame, width=1000)
        orig = image.copy()
        orig_h, orig_w = orig.shape[:2]
        image = cv2.resize(image, (new_w, new_h))
        blob = cv2.dnn.blobFromImage(image, 1.0, (new_w, new_h), (123.68, 116.78, 103.94),
                                     swapRB=True, crop=False)
        net2.setInput(blob)
        scores, geometry = net2.forward(layer_names)
        rectangles, confidences = box_extractor(scores, geometry, min_confidence=0.5)
        boxes = non_max_suppression(np.array(rectangles), probs=confidences)

        for (start_x, start_y, end_x, end_y) in boxes:
            start_x = int(start_x * ratio_w)
            start_y = int(start_y * ratio_h)
            end_x = int(end_x * ratio_w)
            end_y = int(end_y * ratio_h)

            dx = int(end_x - start_x)
            dy = int(end_y - start_y)

            start_x = max(0, start_x - dx)
            start_y = max(0, start_y - dy)
            end_x = min(orig_w, end_x + (dx * 2))
            end_y = min(orig_h, end_y + (dy * 2))

            roi = orig[start_y:end_y, start_x:end_x]
            config = '-l eng --oem 1 --psm 7 -c page_separator='''
            text = pytesseract.image_to_string(roi, config=config).replace("\n", "")
            wordsRecognized.extend(text.split(" "))

        cv2.imshow("Webcam", cv2.resize(frame, (800, 600)))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            fps.stop()
            break
        fps.update()
except Exception as e:
    raise e

print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
print("[INFO] Closing Webcam...")
camera.release()
print("Words found throughout the session are ", set(wordsRecognized))
cv2.destroyAllWindows()
print("[INFO] cleaning up...")
