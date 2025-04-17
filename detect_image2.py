import sys
import platform
import os
import cv2
import numpy as np
import mysql.connector
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate


def edgetpu_lib_name():
    return {
        'Darwin': 'libedgetpu.1.dylib',
        'Linux': 'libedgetpu.so.1',
        'Windows': 'edgetpu.dll',
    }.get(platform.system(), None)


class YOLOv8TFLiteDetector:
    def __init__(self, model_path: str, score_threshold: float = 0.25, num_threads: int = 1, enable_edgetpu: bool = False):
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
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        boxes = []
        for row in output:
            # 確保至少有6個欄位再取
            if len(row) >= 6:
                x_center, y_center, w, h, confidence, class_id = row[:6]
                class_id = int(class_id)
                print(f"[DEBUG] class_id={class_id}, confidence={confidence:.3f}")
                if confidence > self.score_threshold and class_id == 0:
                    x1 = int((x_center - w / 2) * image.shape[1])
                    y1 = int((y_center - h / 2) * image.shape[0])
                    x2 = int((x_center + w / 2) * image.shape[1])
                    y2 = int((y_center + h / 2) * image.shape[0])
                    boxes.append((x1, y1, x2, y2, confidence))
        return boxes


def main():
    model_path = "best_float32.tflite"
    image_path = "test.jpg"
    enable_edgetpu = False
    num_threads = 2

    print(f"📷 載入圖片：{image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print("❌ 無法讀取圖片")
        sys.exit(1)

    detector = YOLOv8TFLiteDetector(model_path, score_threshold=0.25, num_threads=num_threads, enable_edgetpu=enable_edgetpu)
    boxes = detector.detect(image)
    count = len(boxes)
    print(f"✅ 輪圈數量（符合 confidence 條件者）: {count}")

    for x1, y1, x2, y2, score in boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(image, f"{score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.putText(image, f"round: {count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Detection Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    try:
        db = mysql.connector.connect(
            host="192.168.64.128",
            user="root",
            password="",
            database="mqtt_test"
        )
        cursor = db.cursor()
        sql = "INSERT INTO messages (topic, message) VALUES (%s, %s)"
        val = ("wheel_count", str(count))
        cursor.execute(sql, val)
        db.commit()
        db.close()
        print("✅ 結果已寫入資料庫")
    except Exception as e:
        print(f"❌ 資料庫連線或寫入錯誤：{e}")


if __name__ == '__main__':
    main()
