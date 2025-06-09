import flwr as fl
from ultralytics import YOLO
import os
import torch
import gc

CLIENT_ID = os.environ.get("CLIENT_ID", "client_1")
DATA_DIR = f"./{CLIENT_ID}"
MODEL_PATH = "yolov8n.pt"

# 限制 CPU 使用
torch.set_num_threads(1)
torch.set_num_interop_threads(1)  # 限制 PyTorch 自行多執行緒

class YOLOClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = YOLO(MODEL_PATH)

    def get_parameters(self, config):
        return self.model.model.state_dict()

    def set_parameters(self, parameters):
        self.model.model.load_state_dict(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        self.model.train(
            data={
                "train": os.path.join(DATA_DIR, "images/train"),
                "val": os.path.join(DATA_DIR, "images/train"),
                "names": ["squat", "run", "sit", "stretch", "walk", "jump", "bendover", "stand", "lying"],
                "nc": 9
            },
            epochs=1,
            imgsz=224,     # 更小輸入尺寸
            batch=1,       # 最小 batch
            workers=0,     # 關掉 dataloader 執行緒
            device="cpu",
            close_mosaic=True  # 關閉 mosaic（減少記憶體）
        )

        # 強制釋放記憶體
        gc.collect()
        torch.cuda.empty_cache()

        return self.get_parameters(config={}), 1, {}

    def evaluate(self, parameters, config):
        return 0.0, 0, {}

fl.client.start_numpy_client(server_address="192.168.183.128:8080", client=YOLOClient())
