from src.face_detector import YOLOv5
from src.FaceAntiSpoofing import AntiSpoof
import cv2
import numpy as np
import argparse

COLOR_REAL = (0, 255, 0)
COLOR_FAKE = (0, 0, 255)
COLOR_UNKNOWN = (127, 127, 127)

def increased_crop(img, bbox : tuple, bbox_inc : float = 1.5):
    # Crop face based on its bounding box
    real_h, real_w = img.shape[:2]
    
    x, y, w, h = bbox
    w, h = w - x, h - y
    l = max(w, h)
    
    xc, yc = x + w/2, y + h/2
    x, y = int(xc - l*bbox_inc/2), int(yc - l*bbox_inc/2)
    x1 = 0 if x < 0 else x 
    y1 = 0 if y < 0 else y
    x2 = real_w if x + l*bbox_inc > real_w else x + int(l*bbox_inc)
    y2 = real_h if y + l*bbox_inc > real_h else y + int(l*bbox_inc)
    
    img = img[y1:y2,x1:x2,:]
    img = cv2.copyMakeBorder(img, 
                             y1-y, int(l*bbox_inc-y2+y), 
                             x1-x, int(l*bbox_inc)-x2+x, 
                             cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img

def make_prediction(img, face_detector, anti_spoof):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 
    bbox = face_detector([img])[0]
    
    if bbox.shape[0] > 0:
        bbox = bbox.flatten()[:4].astype(int)
    else:
        return None

    pred = anti_spoof([increased_crop(img, bbox, bbox_inc=1.5)])[0]
    score = pred[0][0]
    label = np.argmax(pred)   
    
    return bbox, label, score

   
if __name__ == "__main__":
    # parsing arguments
    def check_zero_to_one(value):
        fvalue = float(value)
        if fvalue <= 0 or fvalue >= 1:
            raise argparse.ArgumentTypeError("%s is an invalid value" % value)
        return fvalue
    # Parsing arguments for the script
    p = argparse.ArgumentParser(description="Spoofing attack detection on a single image")
    p.add_argument("--input", "-i", type=str, required=True, help="Path to the image for predictions")
    p.add_argument("--output", "-o", type=str, default=None, help="Path to save the processed image")
    p.add_argument("--model_path", "-m", type=str, default="saved_models/AntiSpoofing_bin_1.5_128.onnx", help="Path to the ONNX model for anti-spoofing")
    p.add_argument("--threshold", "-t", type=check_zero_to_one, default=0.5, help="real face probability threshold above which the prediction is considered true")
    args = p.parse_args()

    # Initialize the face detector and anti-spoofing models
    face_detector = YOLOv5('saved_models/yolov5s-face.onnx')
    anti_spoof = AntiSpoof(args.model_path)

    # Load the image
    img = cv2.imread(args.input)
    if img is None:
        print("Error loading the image")
        exit()

    # Process the image
    pred = make_prediction(img, face_detector, anti_spoof)

    if pred is not None:
        (x1, y1, x2, y2), label, score = pred

        # Determine the label and draw the bounding box
        if label == 0:
            if score > args.threshold:
                res_text = "REAL {:.2f}".format(score)
                color = COLOR_REAL
            else:
                res_text = "UNKNOWN"
                color = COLOR_UNKNOWN
        else:
            res_text = "FAKE {:.2f}".format(score)
            color = COLOR_FAKE

        # Draw the bounding box and label on the image
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, res_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Save or display the result
        if args.output:
            cv2.imwrite(args.output, img)
        else:
            cv2.imshow('Processed Image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("No face detected in the image")
