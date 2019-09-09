import cv2 as cv

face_network = cv.dnn.readNet("face-detection-retail-0004.bin",
                              'face-detection-retail-0004.xml')

cap = cv.VideoCapture(0)
hasFrame = True
while hasFrame:
    hasFrame, frame = cap.read()
    face_detector_blob = cv.dnn.blobFromImage(frame, size=(300, 300))
    face_network.setInput(face_detector_blob)
    face_network_out = face_network.forward()

    for detection in face_network_out.reshape(-1, 7):
        confidence = float(detection[2])

        if confidence > .2:
            frame = cv.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 30)
    cv.imshow("video", frame)
    cv.waitKey(1)
    cv.waitKey(1)