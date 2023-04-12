import math
import os, sys
import face_recognition_custom
import cv2
import numpy as np
import datetime

today = datetime.datetime.now()
day = today.date()

def write_logs(names):
    with open('check.txt') as f:
        lines = f.read().splitlines()
    list_log = []
    for i in range(len(lines)):
        lines[i] = lines[i].split(" ")
        if str(day) == lines[i][1]:
            if lines[i][0] not in list_log:
                list_log.append(lines[i][0])
    for name in names:
        full_logs = str(name) + " " + str(today)
        # print(full_logs)
        if name not in list_log:
            f = open("check.txt", "a")
            f.write(full_logs + "\n")
            f.close()


def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)
    
    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value,2)) + '%'
    
class FaceRegconition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True
    
    def __init__(self):
        self.encode_faces()
    
    def encode_faces(self):
        # for image in os.listdir('data_face'):
        #     print(image)
        for image in os.listdir('data_face'):
            face_image = face_recognition_custom.load_image_file(f"data_face/{image}")
            face_encoding = face_recognition_custom.face_encodings(face_image)[0]
            
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)           
        print(self.known_face_names) 
    
    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            sys.exit("Video not found ...")
            
        while True:
            ret, frame = video_capture.read()
            frame = cv2.flip(frame, 1)
            if self.process_current_frame:
                small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame[:, :, ::-1]
                
                self.face_locations = face_recognition_custom.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition_custom.face_encodings(rgb_small_frame, self.face_locations)     
                
                self.face_names = []
                short_names = []
                for face_encoding in self.face_encodings:
                    matches = face_recognition_custom.compare_faces(self.known_face_encodings, face_encoding)
                    name = "UNK"
                    confidence = "???"
                    
                    face_distances = face_recognition_custom.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])
                        
                    self.face_names.append(f'{name[:-4]} ({confidence})')
                    short_names.append(f'{name[:-4]}')
                
            self.process_current_frame = not self.process_current_frame
            
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                if name == "UNK":
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0, 0, 255), cv2.FILLED)
                else:
                    cv2.rectangle(frame, (left, top), (right, bottom), (50,205,50), 2)
                    cv2.rectangle(frame, (left, bottom -35), (right, bottom), (50,205,50), cv2.FILLED)
                    write_logs(short_names)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
                
            cv2.imshow('Face Recognition', frame)
            
            if cv2.waitKey(1) == ord('q'):
                break
            
        video_capture.release()
        cv2.destroyAllWindows()
            
if __name__ == '__main__':
    fr = FaceRegconition()
    fr.run_recognition()
    
    
           