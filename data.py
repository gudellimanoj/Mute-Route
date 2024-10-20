import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# Initialize Video Capture and Hand Detector
cap = cv2.VideoCapture(0) 
detector = HandDetector(maxHands=1)

# Parameters for cropping and resizing
offset = 20
imgSize = 300
counter = 0

# Folder path to save images (ensure this folder exists)
folder = "/Users/valla/OneDrive/Desktop/SignDetection/Data/No"
if not os.path.exists(folder):
    os.makedirs(folder)

# Main loop for video capture and hand detection
while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture video")
        break

    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        try:
            imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]
        except:
            continue  # If cropping fails, skip this frame
        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap: wCal + wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize

        # Display the cropped hand and the resized hand on a white background
        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    # Display the original video feed with hand detection
    cv2.imshow('Image', img)
    
    key = cv2.waitKey(1)
    
    # Save the image when 's' key is pressed
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(f"Saved image {counter}")

    # Break the loop with the 'q' key
    if key == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
