import flwr as fl
import tensorflow as tf
import os
from model import create_model
from loader import load_dataset

# 環境變數指定 client ID，例如：CLIENT_ID=client_1 python3 client.py
CLIENT_ID = os.environ.get("CLIENT_ID", "client_1")

# 目錄設定
IMG_DIR = "./JPEGImages"
JSON_DIR = "./Annotations"
SPLIT_FILE = f"./ImageSets/{CLIENT_ID}.txt"  # e.g., ./ImageSets/client_1.txt

# 載入資料
train_ds = load_dataset(SPLIT_FILE, IMG_DIR, JSON_DIR)

# 建立模型
model = create_model()

# 封裝成 Flower Client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(train_ds, epochs=1)
        return model.get_weights(), len(train_ds), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, acc = model.evaluate(train_ds)
        return loss, len(train_ds), {"accuracy": acc}

# 啟動 client
fl.client.start_numpy_client(server_address="192.168.183.128:8080", client=FlowerClient())
