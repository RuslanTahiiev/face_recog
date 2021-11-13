from datetime import datetime
import numpy as np
import face_recognition
import cv2
import os
from settings import Settings


class FaceRecog:

    def __init__(self):
        self.images = []
        self.class_names = []

    def create(self):
        settings = Settings()
        my_list = settings.images()
        print(my_list)
        for cls in my_list:
            cur_img = cv2.imread(f'{settings.data_path}\\{cls}')
            self.images.append(cur_img)
            self.class_names.append(os.path.splitext(cls)[0])
        print(self.class_names)

    def find_encodings(self):
        encode_list = []
        for img in self.images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encode_list.append(encode)
        return encode_list

    def mark_attendence(self, name):
        with open('attendance.csv', 'r+') as file:
            data_list =  file.readlines()
            name_list = []
            for line in data_list:
                entry = line.split(',')
                name_list.append(
                    entry[0]
                )
            if name not in name_list:
                now = datetime.now().strftime('%H:%M:%S')
                file.writelines(f'\n{name}, {now}')

    def recognition(self):

        self.create()
        encoded_list_known = self.find_encodings()

        # print('Decoding end.')

        cap = cv2.VideoCapture(0)

        while True:
            success, img = cap.read()
            img_s = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)

            face_cur_frame = face_recognition.face_locations(img_s)
            encode_cur_frame = face_recognition.face_encodings(img_s, face_cur_frame)

            for encode_face, face_loc in zip(encode_cur_frame, face_cur_frame):
                matches = face_recognition.compare_faces(encoded_list_known, encode_face)
                face_dis = face_recognition.face_distance(encoded_list_known, encode_face)
                # print(face_dis)
                match_index = np.argmin(face_dis)

                if matches[match_index]:
                    name = self.class_names[match_index]
                    # print(name)
                    y1, x2, y2, x1 = face_loc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    self.mark_attendence(name)

            cv2.imshow('WebCam', img)
            cv2.waitKey(1)
