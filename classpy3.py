import argparse
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


def run(model: str, num_threads: int, enable_edgetpu: bool, ip: str, camera_id: int, width: int, height: int):
    detector = YOLOv8TFLiteDetector(model, score_threshold=0.3, num_threads=num_threads, enable_edgetpu=enable_edgetpu)

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
        boxes = detector.detect(image)

        # 顯示輪圈數量
        count = len(boxes)
        cv2.putText(image, f"\u8f2a\u5708\u6578\u91cf: {count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        for x1, y1, x2, y2, score in boxes:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(image, f"{score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        sql = "INSERT INTO messages (topic, message) VALUES (%s, %s)"
        val = ("wheel_count", str(count))
        cursor.execute(sql, val)
        db.commit()

        cv2.imshow("Wheel Detection", image)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    db.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--numThreads', type=int, default=4)
    parser.add_argument('--enableEdgeTPU', action='store_true')
    parser.add_argument('--ip', required=True)
    parser.add_argument('--cameraId', type=int, default=0)
    parser.add_argument('--frameWidth', type=int, default=640)
    parser.add_argument('--frameHeight', type=int, default=480)
    args = parser.parse_args()

    run(args.model, args.numThreads, args.enableEdgeTPU, args.ip, args.cameraId, args.frameWidth, args.frameHeight)


if __name__ == '__main__':
    main()
