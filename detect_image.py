import sys
import platform
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
    def __init__(self, model_path: str, score_threshold: float = 0.3, num_threads: int = 1, enable_edgetpu: bool = False):
        if enable_edgetpu:
            if edgetpu_lib_name() is None:
                raise OSError("當前 OS 不支援 Coral EdgeTPU")
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
            if row[4] > self.score_threshold:
                x_center, y_center, w, h = row[0:4]
                x1 = int((x_center - w / 2) * image.shape[1])
                y1 = int((y_center - h / 2) * image.shape[0])
                x2 = int((x_center + w / 2) * image.shape[1])
                y2 = int((y_center + h / 2) * image.shape[0])
                boxes.append((x1, y1, x2, y2, row[4]))
        return boxes


def main():
    model_path = "best_float32.tflite"         # ✅ 模型路徑
    image_path = "test.jpg"                    # ✅ 圖片檔案名稱
    enable_edgetpu = False                     # ✅ 如果有 Coral 可以改 True
    num_threads = 2                            # ✅ 可依 Pi 核心數調整

    # 連線到資料庫
    db = mysql.connector.connect(
        host="192.168.64.128",
        user="root",
        password="",
        database="mqtt_test"
    )
    cursor = db.cursor()

    # 載入圖片
    image = cv2.imread(image_path)
    if image is None:
        print("❌ 無法讀取圖片")
        sys.exit(1)

    # 偵測輪圈
    detector = YOLOv8TFLiteDetector(model_path, score_threshold=0.3, num_threads=num_threads, enable_edgetpu=enable_edgetpu)
    boxes = detector.detect(image)
    count = len(boxes)
    print(f"✅ 輪圈數量：{count}")

    # 畫出輪框
    for x1, y1, x2, y2, score in boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(image, f"{score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # 顯示數量
    cv2.putText(image, f"round: {count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 顯示圖片
    cv2.imshow("Detection Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 寫入 MySQL
    sql = "INSERT INTO messages (topic, message) VALUES (%s, %s)"
    val = ("wheel_count", str(count))
    cursor.execute(sql, val)
    db.commit()
    db.close()
    print("✅ 結果已寫入資料庫")

if __name__ == '__main__':
    main()
