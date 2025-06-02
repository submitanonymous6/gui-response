import argparse
import os
import sys

from screen_record import ScreenRecord


def process_video(video_file, output_dir):
    app_type = 'mobile'

    # During video recording, some unnecessary regions may be captured, which can affect similarity calculations.
    # For example, the status bar at the top of the mobile screen might show a changing clock even when the UI is static,
    # leading to false detection of visual changes between otherwise identical frames.
    # It is recommended to specify the effective screen region (top-left coordinates, width, height).
    frame_x = None
    frame_y = None
    frame_w = None
    frame_h = None

    # Call the interface to process the video and compute performance metrics
    app_screen_record = ScreenRecord(video_file, output_dir, app_type, frame_x, frame_y, frame_w, frame_h)
    app_screen_record.calculate_perf_metrics()


if __name__ == '__main__':
    current_file = os.path.abspath(__file__)
    project_path = os.path.abspath(os.path.join(current_file, "../.."))

    if project_path not in sys.path:
        sys.path.insert(0, project_path)

    parser = argparse.ArgumentParser(description="Process a video and extract frames.")
    parser.add_argument('--video', type=str, required=True, help='Path to the input video')
    parser.add_argument('--output', type=str, required=True, help='Directory to save the output frames')
    args = parser.parse_args()
    process_video(args.video, args.output)


