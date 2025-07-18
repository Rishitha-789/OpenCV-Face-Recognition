import os

def load_images_from_folder(folder):
    images = []
    for name in os.listdir(folder):
        person_dir = os.path.join(folder, name)
        if not os.path.isdir(person_dir):
            continue
        for file in os.listdir(person_dir):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(person_dir, file)
                images.append((img_path, name))
    return images
