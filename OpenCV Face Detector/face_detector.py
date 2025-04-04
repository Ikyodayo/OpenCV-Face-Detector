import cv2

# Path ke file Haarcascade XML
face_cascade_path = "src\\face.xml"
hand_cascade_path = "src\\hand.xml"
eye_cascade_path = "src\\eye.xml"
mouth_cascade_path = "src\\mouth.xml"
nose_cascade_path = "src\\nose.xml"

# Load classifier
faceCascade = cv2.CascadeClassifier(face_cascade_path)
handCascade = cv2.CascadeClassifier(hand_cascade_path)
eyeCascade = cv2.CascadeClassifier(eye_cascade_path)
mouthCascade = cv2.CascadeClassifier(mouth_cascade_path)
noseCascade = cv2.CascadeClassifier(nose_cascade_path)

# Open webcam
video_capture = cv2.VideoCapture(0)

# Cek kamera
if not video_capture.isOpened():
    print("Error: Could not open camera.")
    exit()

# Font
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
thickness = 2

while True:
    ret, frame = video_capture.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah dan tangan
    faces = faceCascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    hands = handCascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        face_region_gray = gray[y:y + h, x:x + w]
        face_region_color = frame[y:y + h, x:x + w]

        # Mata
        eyes = eyeCascade.detectMultiScale(face_region_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face_region_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)
            cv2.putText(face_region_color, "Eyes", (ex, ey + eh + 20), font, font_scale, (0, 0, 255), thickness)

        # Mulut
        mouth_region_gray = gray[y + h // 2:y + h, x:x + w]
        mouths = mouthCascade.detectMultiScale(mouth_region_gray, 1.3, 5, minSize=(20, 20))
        if len(mouths) > 0:
            (mx, my, mw, mh) = mouths[0]
            cv2.rectangle(frame, (x + mx, y + h // 2 + my), (x + mx + mw, y + h // 2 + my + mh), (0, 165, 255), 2)
            cv2.putText(frame, "Mouth", (x + mx, y + h // 2 + my + mh + 20), font, font_scale, (0, 165, 255), thickness)

        # Hidung
        noses = noseCascade.detectMultiScale(face_region_gray, 1.1, 5, minSize=(20, 20))
        if len(noses) > 0:
            (nx, ny, nw, nh) = noses[0]
            cv2.rectangle(face_region_color, (nx, ny), (nx + nw, ny + nh), (255, 255, 0), 2)
            cv2.putText(face_region_color, "Nose", (nx, ny + nh + 20), font, font_scale, (255, 255, 0), thickness)

        # Label wajah
        cv2.putText(frame, "Face", (x, y + h + 20), font, font_scale, (0, 255, 0), thickness)

    # Tangan (maks 2)
    for i, (x, y, w, h) in enumerate(hands[:2]):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Hand", (x, y + h + 20), font, font_scale, (255, 0, 0), thickness)

    #Frame Kotak
    cv2.imshow('Video', frame)


    # Tekan q untuk keluar, s untuk screenshot
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite("screenshot.png", frame)
        print("Screenshot saved!")

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
