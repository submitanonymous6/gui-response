import os
import pickle
import sys

import cv2
import tensorflow as tf

from utils_common import Frame, ScreenTap
from utils_common import output_data_to_json


class TouchDetection:
    def __init__(self, video_path, model):
        self.touch_detections = []
        self.model_path = model
        self.video_path = video_path
        self.detections = []

    def execute_with_cache(self):
        cur_path = self.video_path
        video_dir, video_file = os.path.split(cur_path)
        video_name, video_extension = os.path.splitext(video_file)
        cur_dir_path = os.path.join(os.path.dirname(cur_path), video_name)

        if not os.path.exists(cur_path):
            sys.exit("ERROR: Specified video does not exist " + cur_path)

        if not os.path.exists(cur_dir_path):
            os.mkdir(cur_dir_path)

        # Execute touch detection
        pkl_path = os.path.join(cur_dir_path, "detections.pkl")
        if os.path.exists(pkl_path):
            print("⚡ Skipping touch detection; loading from pickle...", flush=True)
            with open(pkl_path, "rb") as f:
                complete_detections = pickle.load(f)
            self.touch_detections = complete_detections
        else:
            print("⚡ Computing touch detection; no pickle...", flush=True)
            self.execute_detection()
            complete_detections = self.get_touch_detections()

            with open(pkl_path, "wb") as f:
                pickle.dump(complete_detections, f)
            output_data_to_json(complete_detections, os.path.join(cur_dir_path, "detections.json"))

    def execute_detection(self):
        print("📦 Loading TF2 SavedModel from:", self.model_path)
        model = tf.saved_model.load(self.model_path)
        infer = model.signatures['serving_default']

        # === Prepare paths ===
        video_file = os.path.basename(self.video_path)
        video_name, _ = os.path.splitext(video_file)
        video_dir = os.path.dirname(self.video_path)
        detection_output_path = os.path.join(video_dir, video_name, "detected_frames")
        os.makedirs(detection_output_path, exist_ok=True)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"❌ Cannot open video: {self.video_path}")

        frame_index = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = tf.convert_to_tensor(rgb_frame[None, ...], dtype=tf.uint8)
            outputs = infer(input_tensor)

            boxes = outputs['detection_boxes'].numpy()[0]
            scores = outputs['detection_scores'].numpy()[0]

            im_height, im_width, _ = frame.shape

            detection = Frame(frame_index)
            for i in range(len(boxes)):
                score = scores[i]
                if score > 0.5:
                    box = boxes[i]
                    yMin, xMin, yMax, xMax = box
                    xMin = int(xMin * im_width)
                    xMax = int(xMax * im_width)
                    yMin = int(yMin * im_height)
                    yMax = int(yMax * im_height)

                    # ✅ Draw bounding box using OpenCV
                    cv2.rectangle(frame, (xMin, yMin), (xMax, yMax), (0, 255, 0), 2)
                    label = f"{score:.2f}"
                    cv2.putText(frame, label, (xMin, yMin - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    x = xMin + ((xMax - xMin) / 2.0)
                    y = yMin + ((yMax - yMin) / 2.0)
                    detection.add_tap(ScreenTap(x, y, float(score)))

            # Save detection if taps exist
            if len(detection.get_screen_taps()) > 0:
                self.touch_detections.append(detection)

            output_file = os.path.join(detection_output_path, f"bbox-{frame_index:04d}.jpg")
            cv2.imwrite(output_file, frame)

            frame_index += 1
            if frame_index % 50 == 0:
                print(f"📊 Processed {frame_index}/{total_frames} frames...")

        cap.release()
        print(f"📦 Total detection frames with tap: {len(self.touch_detections)}")

    def get_touch_detections(self):
        return self.touch_detections


if __name__ == "__main__":
    current_file = os.path.abspath(__file__)
    project_path = os.path.abspath(os.path.join(current_file, "../.."))

    if project_path not in sys.path:
        sys.path.insert(0, project_path)

    video_path = os.path.join(project_path, "examples", "demo.mp4")
    model = os.path.join(project_path, "src", "touch_model")
    print(video_path)
    touch_detection = TouchDetection(video_path, model)
    touch_detection.execute_with_cache()

    print(f"✅ detect tap completed successfully for {video_path}\n")
