import cv2
import numpy as np
# from tensorflow.keras.models import Model
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# from tensorflow.keras.preprocessing import image
import face_recognition
import pickle

class FaceRecognition:
    def __init__(self, database_file=None):
        self.known_face_encodings = []
        self.known_face_labels = []
        if database_file:
            self.load_data(database_file)
    
    def load_data(self, database_file):
        try:
            with open(database_file, "rb") as f:
                data = pickle.load(f)
                self.known_face_encodings = data["encodings"]
                self.known_face_labels = data["labels"]
                print("Face recognition: Load successfully!")
        except FileNotFoundError:
            print("Database file not found.")
    
    def save_data(self, database_file):
        # Save unique data
        data = {"encodings": self.known_face_encodings, "labels": self.known_face_labels}
        with open(database_file, "wb") as f:
            pickle.dump(data, f)
    
    def register_face(self, image, label, database_file=None):
        print("Registering face")
        resized_image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        
        encoding = face_recognition.face_encodings(rgb_image)[0]
        if encoding is not None and len(encoding) > 0:
            self.known_face_encodings.append(encoding)
            self.known_face_labels.append(label)
            
            if database_file:
                self.save_data(database_file)
                return True
        return False

    def get_registered_faces(self, database_file):
        self.load_data(database_file=database_file)
        return zip(self.known_face_encodings, self.known_face_labels)

    def get_encoding(self, image):
        # Resize image to 1/4 size for faster face recognition processing
        resized_image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        
        # Get face encodings using face_recognition lib
        encodings = face_recognition.face_encodings(rgb_image)
        if len(encodings) > 0:
            return encodings[0]
        else:
            return None
    
    def recognize_faces(self, image):
        # Resize image to 1/4 size for faster face recognition processing
        resized_image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        
        # Get face locations
        face_locations = face_recognition.face_locations(rgb_image)
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        # Compare face encodings with known face encodings
        recognized_faces = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = 'Unkown'
            
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_labels[first_match_index]
            
            recognized_faces.append(name)

        # Scale face locations back to original size
        face_locations = [(top*4, right*4, bottom*4, left*4) for (top, right, bottom, left) in face_locations]
        return face_locations, recognized_faces