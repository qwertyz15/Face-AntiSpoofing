from src.face_detector import YOLOv5
from src.FaceAntiSpoofing import AntiSpoof
import cv2
import glob
import os
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
    p.add_argument("--input", "-i", type=str, required=True, help="Path to the image direcotry for predictions")
    p.add_argument("--output", "-o", type=str, default=None, help="Path to save the processed image")
    p.add_argument("--model_path", "-m", type=str, default="saved_models/AntiSpoofing_bin_1.5_128.onnx", help="Path to the ONNX model for anti-spoofing")
    p.add_argument("--threshold", "-t", type=check_zero_to_one, default=0.5, help="real face probability threshold above which the prediction is considered true")
    args = p.parse_args()

    # Initialize the face detector and anti-spoofing models
    face_detector = YOLOv5('saved_models/yolov5s-face.onnx')
    anti_spoof = AntiSpoof(args.model_path)

    # Define the path to the directory containing images
    image_directory = args.input

    # List all image files in the directory with multiple extensions using glob.glob
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []


    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_directory, '**', ext), recursive=True))
    image_files.sort()

    correct = 0
    incorrect = 0
    total = 0

    for img_path in image_files:
        # Load the image
        img = cv2.imread(img_path)
        print(img_path)
        if img is None:
            print("Error loading the image")
            exit()

        # Process the image
        pred = make_prediction(img, face_detector, anti_spoof)

        # Get the full path of the parent directory
        parent_dir_full_path = os.path.dirname(img_path)

        # Get the name of the parent directory and convert it to lowercase
        org_label = os.path.basename(parent_dir_full_path).lower()
        
        if(org_label == 'live'):
            org_label = 0
        elif(org_label == 'spoof'):
            org_label = 1

        if pred is not None:
            (x1, y1, x2, y2), label, score = pred


            # print(int(org_label), int(label))

            # Determine the label and draw the bounding box
            if label == 0:
                if score > args.threshold:
                    LBL = 0
                else:
                    LBL = -1
            else:
                LBL = 1

            if(LBL != -1):
                total += 1

                if(int(LBL) == int(org_label)):
                    correct += 1
                    print(f"Correct: {correct} / {total}")
                else:
                    incorrect += 1
                    print(f"Incorrect: {incorrect} / {total}")

    print(f"Correct: {correct}")
    print(f"InCorrect: {incorrect}")
    print(f"Total: {total}")
    print(f"In General: {len(image_files)}")
    print(f"Acc : {correct/total}")

        #     # Draw the bounding box and label on the image
        #     cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        #     cv2.putText(img, res_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        #     # Save or display the result
        #     if args.output:
        #         cv2.imwrite(args.output, img)
        #     else:
        #         cv2.imshow('Processed Image', img)
        #         cv2.waitKey(0)
        #         cv2.destroyAllWindows()
        # else:
        #     print("No face detected in the image")
