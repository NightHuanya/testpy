import argparse
import sys
import time
import platform
import cv2
import numpy as np
import dataclasses
from typing import List

# ✅ 新增：MySQL 連線
import mysql.connector

# 嘗試載入 tflite_runtime，若無則使用 tensorflow
try:
    from tflite_runtime.interpreter import Interpreter
    from tflite_runtime.interpreter import load_delegate
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    load_delegate = tf.lite.experimental.load_delegate

@dataclasses.dataclass
class ImageClassifierOptions:
    enable_edgetpu: bool = False
    max_results: int = 1
    num_threads: int = 1
    score_threshold: float = 0.0
    label_path: str = "labels.txt"

@dataclasses.dataclass
class Category:
    label: str
    score: float

def edgetpu_lib_name():
    return {
        'Darwin': 'libedgetpu.1.dylib',
        'Linux': 'libedgetpu.so.1',
        'Windows': 'edgetpu.dll',
    }.get(platform.system(), None)

class ImageClassifier:
    def __init__(self, model_path: str, options: ImageClassifierOptions):
        self._options = options
        self._load_labels(options.label_path)

        if options.enable_edgetpu:
            if edgetpu_lib_name() is None:
                raise OSError("當前 OS 不支援 Coral EdgeTPU")
            interpreter = Interpreter(
                model_path=model_path,
                experimental_delegates=[load_delegate(edgetpu_lib_name())],
                num_threads=options.num_threads
            )
        else:
            interpreter = Interpreter(model_path=model_path, num_threads=options.num_threads)

        interpreter.allocate_tensors()
        self._input_details = interpreter.get_input_details()
        self._output_details = interpreter.get_output_details()
        self._input_height = self._input_details[0]['shape'][1]
        self._input_width = self._input_details[0]['shape'][2]
        self._is_quantized_input = self._input_details[0]['dtype'] == np.uint8
        self._is_quantized_output = self._output_details[0]['dtype'] == np.uint8
        self._interpreter = interpreter

    def _load_labels(self, label_path: str):
        try:
            with open(label_path, "r") as f:
                self._label_list = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            print(f"警告：無法找到標籤檔 {label_path}，請確保標籤檔存在")
            self._label_list = []

    def _set_input_tensor(self, image: np.ndarray):
        tensor_index = self._input_details[0]['index']
        input_tensor = self._interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        input_tensor = cv2.resize(image, (self._input_width, self._input_height))
        if not self._is_quantized_input:
            input_tensor = np.float32(input_tensor) / 255.0
        return input_tensor

    def classify(self, image: np.ndarray) -> List[Category]:
        image = self._preprocess(image)
        self._set_input_tensor(image)
        self._interpreter.invoke()
        output_tensor = np.squeeze(self._interpreter.get_tensor(self._output_details[0]['index']))
        return self._postprocess(output_tensor)

    def _postprocess(self, output_tensor: np.ndarray) -> List[Category]:
        if self._is_quantized_output:
            scale, zero_point = self._output_details[0]['quantization']
            output_tensor = scale * (output_tensor - zero_point)

        if output_tensor.ndim == 0 or len(output_tensor) == 1:
            score = float(output_tensor) if output_tensor.ndim == 0 else float(output_tensor[0])
            label = self._label_list[1] if score > 0.5 else self._label_list[0]
            return [Category(label=label, score=score)]
        else:
            max_index = np.argmax(output_tensor)
            max_score = float(output_tensor[max_index])
            label = self._label_list[max_index] if max_index < len(self._label_list) else "Unknown"
            return [Category(label=label, score=max_score)]

def run(model: str, label_path: str, max_results: int, num_threads: int, enable_edgetpu: bool, ip: str,
        camera_id: int, width: int, height: int) -> None:
    options = ImageClassifierOptions(
        num_threads=num_threads,
        max_results=max_results,
        enable_edgetpu=enable_edgetpu,
        label_path=label_path
    )
    classifier = ImageClassifier(model, options)

    # ✅ 連線 MySQL
    db = mysql.connector.connect(
        host="192.168.64.128",
        user="root",
        password="",
        database="mqtt_test"
    )
    cursor = db.cursor()

    url = f"http://{ip}:4747/video"
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit('ERROR: 無法讀取攝影機影像，請檢查攝影機設定。')

        image = cv2.flip(image, 1)
        categories = classifier.classify(image)

        for category in categories:
            result_text = f"{category.label} ({round(category.score, 2)})"
            cv2.putText(image, result_text, (24, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

            # ✅ 將 score 寫入資料庫
            sql = "INSERT INTO messages (topic, message) VALUES (%s, %s)"
            val = ("image_classification", str(round(category.score, 4)))
            cursor.execute(sql, val)
            db.commit()

        if cv2.waitKey(1) == 27:
            break
        cv2.imshow('image_classification', image)

    cap.release()
    cv2.destroyAllWindows()
    db.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--maxResults', type=int, default=1)
    parser.add_argument('--numThreads', type=int, default=4)
    parser.add_argument('--enableEdgeTPU', action='store_true')
    parser.add_argument('--ip', required=True)
    parser.add_argument('--cameraId', type=int, default=0)
    parser.add_argument('--frameWidth', type=int, default=640)
    parser.add_argument('--frameHeight', type=int, default=480)
    args = parser.parse_args()

    run(args.model, args.labels, args.maxResults, args.numThreads, args.enableEdgeTPU, args.ip,
        args.cameraId, args.frameWidth, args.frameHeight)

if __name__ == '__main__':
    main()
