import flwr as fl
import tensorflow as tf
import numpy as np
import os
import json

CLIENT_ID = os.environ.get("CLIENT_ID", "client_1")
IMG_DIR = "JPEGImages"
ANNOTATION_DIR = "Annotations"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_CLASSES = 9
train_list_path = f"{CLIENT_ID}.txt"

def load_data(file_path):
    with open(file_path) as f:
        filenames = [line.strip() for line in f.readlines()]

    images, labels = [], []
    for name in filenames:
        img_path = os.path.join(IMG_DIR, f"{name}.jpg")
        json_path = os.path.join(ANNOTATION_DIR, f"{name}.json")
        if not os.path.exists(img_path) or not os.path.exists(json_path):
            continue
        img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
        img = tf.keras.utils.img_to_array(img) / 255.0
        with open(json_path) as jf:
            label = json.load(jf).get("action")
        if label is None:
            continue
        images.append(img)
        labels.append(label)
    label_to_index = {name: i for i, name in enumerate(sorted(set(labels)))}
    int_labels = [label_to_index[l] for l in labels]
    y = tf.keras.utils.to_categorical(int_labels, num_classes=NUM_CLASSES)
    return tf.data.Dataset.from_tensor_slices((np.array(images), y)).shuffle(1000).batch(BATCH_SIZE)

train_ds = load_data(train_list_path)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def set_parameters(self, parameters):
        model.set_weights(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        model.fit(train_ds, epochs=1)
        return self.get_parameters(config), len(train_ds), {}

    def evaluate(self, parameters, config):
        return 0.0, 0, {}

fl.client.start_numpy_client(server_address="192.168.183.128:8080", client=FlowerClient())
