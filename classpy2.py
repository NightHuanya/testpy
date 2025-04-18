import sys
import platform
import os
import cv2
import numpy as np
import mysql.connector
import argparse
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

def edgetpu_lib_name():
    return {
        'Darwin': 'libedgetpu.1.dylib',
        'Linux': 'libedgetpu.so.1',
        'Windows': 'edgetpu.dll',
    }.get(platform.system(), None)

class YOLOv8TFLiteDetector:
    def __init__(self, model_path: str, score_threshold: float = 0.3, num_threads: int = 1, enable_edgetpu: bool = False):
        print(f"🔍 正在載入模型：{model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 找不到模型檔案：{model_path}")

        if enable_edgetpu:
            if edgetpu_lib_name() is None:
                raise OSError("❌ 當前 OS 不支援 Coral EdgeTPU")
            self.interpreter = Interpreter(
                model_path=model_path,
                experimental_delegates=[load_delegate(edgetpu_lib_name())],
                num_threads=num_threads
            )
        else:
            self.interpreter = Interpreter(model_path=model_path, num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_height = self.input_details[0]['shape'][1]
        self.input_width = self.input_details[0]['shape'][2]
        self.score_threshold = score_threshold

    def detect(self, image: np.ndarray):
        input_tensor = cv2.resize(image, (self.input_width, self.input_height))
        input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32) / 255.0
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]  # (5, 8400)

        preds = output.transpose()  # (8400, 5)
        boxes = []

        for row in preds:
            x_center, y_center, w, h = row[0:4]
            confidence = row[4]
            if confidence > self.score_threshold:
                x1 = int((x_center - w / 2) * image.shape[1])
                y1 = int((y_center - h / 2) * image.shape[0])
                x2 = int((x_center + w / 2) * image.shape[1])
                y2 = int((y_center + h / 2) * image.shape[0])
                boxes.append((x1, y1, x2, y2, confidence))

        return boxes

def run(model_path: str, ip: str, width: int = 640, height: int = 480, threshold: float = 0.3):
    detector = YOLOv8TFLiteDetector(model_path, score_threshold=threshold, num_threads=2, enable_edgetpu=False)

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
            print("❌ 無法讀取攝影機畫面")
            break

        boxes = detector.detect(image)
        count = len(boxes)

        for x1, y1, x2, y2, score in boxes:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(image, f"round: {count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("YOLOv8 TFLite Webcam Detection", image)

        sql = "INSERT INTO messages (topic, message) VALUES (%s, %s)"
        val = ("wheel_count", str(count))
        cursor.execute(sql, val)
        db.commit()

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    db.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, required=True, help="手機的 IP 位址")
    parser.add_argument('--model', type=str, default="best_float32.tflite", help="TFLite 模型路徑")
    parser.add_argument('--threshold', type=float, default=0.6, help="信心門檻 (0~1)")
    args = parser.parse_args()

    run(model_path=args.model, ip=args.ip, threshold=args.threshold)
