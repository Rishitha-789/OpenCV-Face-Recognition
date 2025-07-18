## Face Recognition with OpenCV & face\_recognition

A Python project for real-time face recognition using OpenCV and the `face_recognition` library. Known faces are loaded and encoded from a folder, then recognized via webcam.

---

---

### Requirements

Install dependencies:

```bash
pip install opencv-python face_recognition
```

---

### Folder Setup

Put your known face images in subfolders inside `known_faces/`, named after the person:

```
known_faces/
├── Alice/
│   └── alice1.jpg
├── Bob/
│   └── bob1.jpg
```

---

### How to Run

1. Encode Known Faces
   Run this once to create `encodings.pickle`:

   ```bash
   python encode_faces.py
   ```

2. Start Real-Time Recognition
   Launch webcam recognition:

   ```bash
   python recognize_faces.py
   ```

3. Press `q` to quit the webcam window.

---
