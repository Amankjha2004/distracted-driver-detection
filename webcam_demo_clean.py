
from pathlib import Path
import time
import cv2
from ultralytics import YOLO

# ==========================================
# YOLOv12 DRIVER ACTIVITY - CLEAN WEBCAM DEMO
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "yolov12_cnn_v2"
    / "runs"
    / "yolov12n_classification_v2"
    / "weights"
    / "best.pt"
)

WINDOW_NAME = "Distracted Driver Detection - Clean Webcam Demo"
CAMERA_INDEX = 0
IMAGE_SIZE = 224

# Visual settings
GREEN = (50, 255, 70)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def main():
    # 1. Check that the V2 model exists
    if not MODEL_PATH.is_file():
        print("\nERROR: V2 model checkpoint was not found.")
        print("Expected path:")
        print(MODEL_PATH)
        print("\nCheck that best.pt exists in the V2 weights folder.")
        return

    # 2. Load the model
    print("Loading YOLOv12 V2 model...")
    print(f"Model: {MODEL_PATH}")

    try:
        model = YOLO(str(MODEL_PATH))
    except Exception as error:
        print(f"Could not load model: {error}")
        return

    print("Model loaded successfully!")
    print("Class names:", model.names)

    # 3. Start webcam
    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        print("Close other apps using your camera and try again.")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    # 4. FPS tracking
    previous_time = time.perf_counter()
    fps = 0.0

    print("\nWebcam started.")
    print("Press Q or ESC to exit.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read a frame from the webcam.")
                break

            # Mirror the image for a natural webcam experience
            frame = cv2.flip(frame, 1)

            # 5. Run classification
            results = model.predict(
                source=frame,
                imgsz=IMAGE_SIZE,
                verbose=False
            )

            activity = "Unknown"
            confidence = 0.0

            if results and results[0].probs is not None:
                probabilities = results[0].probs

                class_id = int(probabilities.top1)
                confidence = float(probabilities.top1conf)

                # Use the class names saved with the trained model
                activity = model.names[class_id]

                # Convert the class label into readable text
                activity = activity.replace("_", " ").title()

            # 6. Calculate FPS
            current_time = time.perf_counter()
            elapsed = current_time - previous_time

            if elapsed > 0:
                instant_fps = 1.0 / elapsed
                fps = instant_fps if fps == 0 else (
                    0.8 * fps + 0.2 * instant_fps
                )

            previous_time = current_time

            # 7. Draw the top information panel
            height, width = frame.shape[:2]

            overlay_height = 115
            overlay = frame.copy()

            cv2.rectangle(
                overlay,
                (0, 0),
                (width, overlay_height),
                BLACK,
                -1
            )

            frame = cv2.addWeighted(
                overlay, 0.65, frame, 0.35, 0
            )

            cv2.putText(
                frame,
                "YOLOv12 DRIVER ACTIVITY - V2",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                GREEN,
                2,
                cv2.LINE_AA
            )

            prediction_text = (
                f"{activity} | Confidence: {confidence * 100:.1f}%"
            )

            cv2.putText(
                frame,
                prediction_text,
                (20, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                WHITE,
                2,
                cv2.LINE_AA
            )

            # 8. Draw FPS and exit instructions
            cv2.putText(
                frame,
                "Q / ESC: Quit",
                (15, height - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                WHITE,
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (width - 155, height - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                GREEN,
                2,
                cv2.LINE_AA
            )

            # 9. Display the webcam
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

    except KeyboardInterrupt:
        print("\nWebcam stopped by user.")

    except Exception as error:
        print(f"\nRuntime error: {error}")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Webcam closed.")


if __name__ == "__main__":
    main()
