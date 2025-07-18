import os
import cv2
import face_recognition
import pickle
from utils import load_images_from_folder

def encode_faces(known_faces_dir='known_faces', encoding_file='encodings.pickle'):
    known_encodings = []
    known_names = []

    images = load_images_from_folder(known_faces_dir)
    for img_path, name in images:
        image = cv2.imread(img_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model='hog')
        encodings = face_recognition.face_encodings(rgb, boxes)

        for enc in encodings:
            known_encodings.append(enc)
            known_names.append(name)

    data = {"encodings": known_encodings, "names": known_names}
    with open(encoding_file, "wb") as f:
        pickle.dump(data, f)

    print(f"[INFO] Encoded {len(known_names)} face(s) and saved to {encoding_file}")

if __name__ == "__main__":
    encode_faces()
