# loader.py
import json, os
import tensorflow as tf

def load_json_annotations(json_folder):
    annotations = {}
    for fname in os.listdir(json_folder):
        if fname.endswith(".json"):
            with open(os.path.join(json_folder, fname), "r") as f:
                data = json.load(f)
                img = data["imagePath"]
                label = data["label"] if "label" in data else data.get("action", -1)
                annotations[img] = label
    return annotations

def load_dataset(txt_file, img_dir, json_dir, img_size=(224, 224)):
    with open(txt_file, "r") as f:
        lines = [line.strip() for line in f.readlines()]
    
    annotations = load_json_annotations(json_dir)

    image_paths = []
    labels = []

    for name in lines:
        img_file = f"{name}.jpg"
        img_path = os.path.join(img_dir, img_file)
        if img_file in annotations:
            label = annotations[img_file]
            if label is not None:
                image_paths.append(img_path)
                labels.append(int(label))

    def _load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, img_size)
        return image / 255.0, label

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(_load_image).batch(16).shuffle(100)
    return dataset
