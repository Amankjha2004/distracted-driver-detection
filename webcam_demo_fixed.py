from pathlib import Path
import time
import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(r"D:\AI_FINAL_PROJECT")
MODEL_PATH = PROJECT_ROOT / "outputs" / "yolov12_cnn" / "runs" / "yolov12n_classification" / "weights" / "best.pt"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found:\n{MODEL_PATH}")

    print("Loading trained YOLOv12 classification model...")
    model = YOLO(str(MODEL_PATH))
    print("Model loaded.")
    print("Class names:", model.names)
    print("Opening webcam. Press Q or ESC in the camera window to exit.")

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam. Check Windows camera permissions and close other camera apps.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    previous_time = time.time()
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read a webcam frame.")
                break

            result = model.predict(source=frame, verbose=False)[0]

            # Make a writable image before drawing OpenCV text.
            display_frame = np.array(result.plot(), copy=True)

            now = time.time()
            fps = 1.0 / max(now - previous_time, 1e-6)
            previous_time = now

            cv2.putText(
                display_frame,
                f"YOLOv12 classification | FPS: {fps:.1f}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_frame,
                "Press Q or ESC to quit",
                (10, display_frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("Distracted Driver Detection - Webcam Demo", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Webcam closed.")


if __name__ == "__main__":
    main()
