#pip install cmake
#pip install opencv-python
#pip install face_recognition
import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

#Loading known faces
image = face_recognition.load_image_file("faces/photo.jpeg")
image2 = face_recognition.load_image_file("faces/photo2.jpeg")
myImage_encoding = face_recognition.face_encodings(image)[0]
image2_encoding = face_recognition.face_encodings(image2)[0]
known_encodings = [myImage_encoding, image2_encoding]
known_face_names = ["shankar", "virat"]

#Expected faces
students = known_face_names.copy()

face_locations=[]
face_encodings=[]

#Get the current date and time

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    #Recognize faces

    face_locations = face_recognition.face_locations(rgb_small_frame, 1)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        match = face_recognition.compare_faces(known_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        if match[best_match_index]:
            name = known_face_names[best_match_index]

        #add the text if a person is present
        if name in known_face_names:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 100)
            fontScale = 1.5
            fontColor = (255, 0, 0)
            thickness = 3
            lineType = 2
            cv2.putText(frame, name + " is Present", bottomLeftCornerOfText, font, fontScale, fontColor,
                        thickness, lineType)

            if name in students:
                students.remove(name)
                current_time = now.strftime("%H-%M%S")
                lnwriter.writerow(([name, current_time]))

    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) and 0xFF == ord("q"):
        break
video_capture.release()
cv2.destroyAllWindows()
f.close()


