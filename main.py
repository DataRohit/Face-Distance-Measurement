import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import numpy as np

from sklearn.preprocessing import scale

cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)

sens = 20

textList = [
    "Face Distance", "Measurement", "Computer Vision",
    "If you like it", "Pls Star the Repo",
    "And do check my", "Other Projects"
]

while True:
    success, img = cap.read()
    imgText = np.zeros_like(img)
    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]

        pointLeft = face[145]
        pointRight = face[374]

        # cv2.line(img, pointLeft, pointRight, (0, 200, 0), 3)

        # cv2.circle(img, pointLeft, 5, (255, 0, 255), cv2.FILLED)
        # cv2.circle(img, pointRight, 5, (255, 0, 255), cv2.FILLED)

        # # Finding the Focal Length
        # w, _ = detector.findDistance(pointLeft, pointRight)
        # W = 6.3
        # d = 50

        # f = (w * d) / W 
        # # f = 682

        # Finding Distance
        f = 682
        W = 6.3
        w, _ = detector.findDistance(pointLeft, pointRight)

        d = int((W * f) // w)

        cvzone.putTextRect(img, f"Depth: {d}cm",
                           (face[10][0]-100, face[10][1]-50),
                           scale=2)

        for i, text in enumerate(textList):
            singleHeight = 20 + int(int((d/sens)*sens)/4)
            scale = 0.4 + int((d/sens)*sens)/75

            cv2.putText(imgText, text, (50, 50+(i*singleHeight)), cv2.FONT_ITALIC, scale, (255, 255, 255), [2 if i<=1 else 1][0])

    imgStacked = cvzone.stackImages([img, imgText], 2, 1)

    cv2.imshow("Image", imgStacked)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break