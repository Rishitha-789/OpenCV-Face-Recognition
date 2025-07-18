import cv2
import os

def collect_faces(name, output_dir="dataset"):
    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    person_dir = os.path.join(output_dir, name)
    os.makedirs(person_dir, exist_ok=True)

    count = 0
    print("Starting face capture. Press 'q' to quit.")
    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (100, 100))
            cv2.imwrite(f"{person_dir}/{str(count)}.jpg", face_img)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('Face Capture', frame)
        if cv2.waitKey(1) == ord('q') or count >= 50:
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"Saved {count} face images in '{person_dir}'.")

if __name__ == "__main__":
    person_name = input("Enter name: ")
    collect_faces(person_name)
