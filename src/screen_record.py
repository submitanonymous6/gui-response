import os.path
import pandas as pd
from sklearn.ensemble import IsolationForest

from touch_detection import TouchDetection
from user_interaction import UserInteraction
from utils_common import *


class ScreenRecord:
    SIMILARITY_CACHE_FILE = '_similarity.pickle'
    HAS_TAP_CACHE_FILE = '_tap.pickle'
    METRICS_CSV_FILE = u'_metrics.csv'
    ALL_VIDEO_FRAMES_PATH_NAME = u'AllFrames'
    USER_INTERACTION_FRAMES_PATH_NAME = u'UserInteractionFrames'

    SIMILARITY_CSV_FILE = u'_ssim.csv'
    HAS_TAP_CSV_FILE = u'_tap.csv'

    def __init__(self, video_file, output_dir, app_type, frame_x=None, frame_y=None, frame_w=None, frame_h=None):
        self.video_file = video_file
        head_dir, tail_video_file_with_ext = os.path.split(video_file)
        head_video_name, tail_ext = os.path.splitext(tail_video_file_with_ext)
        self.video_name = head_video_name
        self.output_dir = output_dir
        self.app_type = app_type

        self.all_video_frames_dir = os.path.join(output_dir, self.video_name, ScreenRecord.ALL_VIDEO_FRAMES_PATH_NAME)
        self.user_interaction_frames_dir = os.path.join(output_dir, self.video_name, ScreenRecord.USER_INTERACTION_FRAMES_PATH_NAME)

        frame_width, frame_height, frame_count, timestamps = get_video_frame_size_and_timestamps(video_file)
        if frame_x is not None:
            self.frame_x = frame_x
            self.frame_y = frame_y
            self.frame_w = frame_w
            self.frame_h = frame_h
        elif frame_x is None:
            self.frame_x = 0
            self.frame_y = int(frame_height * 0.035)
            self.frame_w = frame_width
            self.frame_h = int(frame_height * 0.965)

        self.frame_count = frame_count
        self.timestamps = timestamps

        # The ssim score list of the video
        # each value of the list is float which indicates the ssim score between this frame and previous frame
        self.frame_diffs_ssim_score = list()

        # The indicator of outlier frame
        # each value of the list indicate whether the frame is outlier
        # -1 for outlier, 1 for not outlier
        self.is_outlier_frame_list = list()

        # The [is frame has tap] list of the video
        # each value of the list is bool which indicates whether the correspond frame has a tap
        self.is_frame_has_tap_list = list()

        self.fps = get_video_fps(self.video_file)

        self.user_interaction_list = list()

        print("video file:", video_file)
        print("video name:", self.video_name)
        print("video fps:", self.fps)
        print("video frame height:", frame_height)
        print("video frame width:", frame_width)
        print("video frame count:", frame_count)
        print("output dir:", self.output_dir)
        print("user interaction frames dir:", self.user_interaction_frames_dir)

    def detect_outlier_frame(self):
        print('detect outlier frame')
        assert len(self.frame_diffs_ssim_score) > 0
        self.frame_diffs_ssim_score = [min(score, 1.0) for score in self.frame_diffs_ssim_score]
        reshaped_frame_diffs_ssim_score = np.array(self.frame_diffs_ssim_score).reshape(-1, 1)
        clf = IsolationForest(random_state=0).fit(reshaped_frame_diffs_ssim_score)
        self.is_outlier_frame_list = clf.predict(reshaped_frame_diffs_ssim_score)
        assert len(self.frame_diffs_ssim_score) == len(self.is_outlier_frame_list)

    def detect_outlier_frame_using_threshold(self):
        print('detect outlier frame using threshold')
        assert len(self.frame_diffs_ssim_score) > 0
        for ssim in self.frame_diffs_ssim_score:
            if ssim < 0.99:
                self.is_outlier_frame_list.append(-1)
            else:
                self.is_outlier_frame_list.append(1)
        assert len(self.frame_diffs_ssim_score) == len(self.is_outlier_frame_list)
        self.collect_ssim_outlier()

    def calculate_frames_similarity_with_cache(self):
        video_similarity_cache_file = os.path.join(self.output_dir, self.video_name, self.video_name + ScreenRecord.SIMILARITY_CACHE_FILE)
        if os.path.isfile(video_similarity_cache_file):
            self.frame_diffs_ssim_score = load_variable(video_similarity_cache_file)
            # self.frame_diffs_ssim_score = self.get_three_average_ssim_score_with_previous()
        else:
            mkdir(os.path.join(self.output_dir, self.video_name))
            frame_diffs_ssim_score = self.calculate_frames_similarity_without_cache()
            save_variable(frame_diffs_ssim_score, video_similarity_cache_file)

            # write ssim to csv file
            df = pd.DataFrame(self.frame_diffs_ssim_score, columns=['ssim'])
            df.to_csv(os.path.join(self.output_dir, self.video_name, self.video_name + ScreenRecord.SIMILARITY_CSV_FILE), index=False)

    def write_all_frames_to_disk(self):
        cap = cv2.VideoCapture(self.video_file)
        success, frame = cap.read()
        mkdir(self.all_video_frames_dir)

        i = 1
        while success:
            frame_file_path = os.path.join(self.all_video_frames_dir, r"%d.jpg" % i)
            image_save(frame_file_path, frame)
            i = i + 1
            success, frame = cap.read()
        cap.release()

    def calculate_frames_similarity_without_cache(self):
        cap = cv2.VideoCapture(self.video_file)
        curr_frame = None
        prev_frame = None
        self.frame_diffs_ssim_score.append(1)  # the first ssim score is 1
        success, frame = cap.read()

        i = 1
        while success:
            curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            if curr_frame is not None and prev_frame is not None:
                ssim_score = calculate_ssim(curr_frame, prev_frame, self.frame_x, self.frame_y, self.frame_w, self.frame_h, 1)
                self.frame_diffs_ssim_score.append(ssim_score)
                if i % 100 == 0:
                    print('calculating similarity for frames, index: ' + str(i) + ', total frames: ' + str(self.frame_count))

            prev_frame = curr_frame
            i += 1
            success, frame = cap.read()
        cap.release()

        return self.frame_diffs_ssim_score

    def split_user_interaction(self):
        """
        Split the video to multiple user interaction.
        """
        # self.is_frame_has_tap_list
        # [False, False, False, lse, False, False, False, False, False, False, True, False, False, False, False, False ...]
        index_user_operation_frame = list()
        for index, has_tap in enumerate(self.is_frame_has_tap_list):
            if self.is_frame_has_tap_list[index] is True:
                index_user_operation_frame.append(index)

        # for segmentation
        index_user_operation_frame.append(index)
        print('index_user_operation_frame: ', index_user_operation_frame)

        # split the video frames to user interactions
        for i in range(0, len(index_user_operation_frame) - 1):
            start_frame = index_user_operation_frame[i]  # user operation frame which contains a tap
            end_frame = index_user_operation_frame[i + 1] - 1
            user_interaction = UserInteraction(start_frame, end_frame, self.frame_diffs_ssim_score, self.is_outlier_frame_list, self.is_frame_has_tap_list, self.fps,
                                               self.timestamps)
            user_interaction.calculate_perf_metrics_forest()
            self.user_interaction_list.append(user_interaction)

    def save_user_interaction_frames(self):
        cap = cv2.VideoCapture(self.video_file)
        success, frame = cap.read()
        frame_index = 0
        clear_and_mkdir(self.user_interaction_frames_dir)

        while success:
            for interaction_id, interaction in enumerate(self.user_interaction_list, 1):
                if interaction.index_operation_frame <= frame_index <= interaction.end_frame:
                    # Create directory for the current user interaction if it's the first frame
                    if frame_index == interaction.index_operation_frame:
                        image_dir = os.path.join(self.user_interaction_frames_dir, str(interaction_id))
                        mkdir(image_dir)
                        assert os.path.exists(image_dir)
                        # first_image_path = os.path.join(self.user_interaction_frames_dir, str(interaction_id), str(frame_index + 1) + u'.jpg')

                    # Save current frame
                    image_filename = f"{frame_index + 1}.jpg"
                    image_path = os.path.join(self.user_interaction_frames_dir, str(interaction_id), image_filename)
                    image_save(image_path, frame)
                    break  # Ensure a frame is only saved once

            # Read next frame from outer loop
            success, frame = cap.read()
            frame_index = frame_index + 1

        cap.release()
        return self.user_interaction_list

    def export_gui_responsiveness_metrics(self):
        """
        Export GUI responsiveness metrics for each user interaction to a CSV file.
        This includes the user operation frame (i.e., start frame), finish frame, end frame,
        and whether a keyboard is shown during the interaction.

        Note: Frame indices are converted to 1-based for readability in the CSV output.
        """
        print(f'Exporting GUI responsiveness metrics...')

        df_frames_metrics = pd.DataFrame(columns=[
            'index',
            'index of user operation frame',
            'index of response frame',
            'index of finish frame',
            'index of end frame',
            'response time(ms)',
            'finish time(ms)'
        ])

        for index, user_interaction in enumerate(self.user_interaction_list, 1):
            index_operation_frame, index_response_frame, index_finish_frame, index_end_frame, response_time, finish_time = user_interaction.get_perf_metrics()

            # Convert None values to empty string for cleaner CSV
            index_operation_frame = index_operation_frame if index_operation_frame is not None else ''
            index_response_frame = index_response_frame if index_response_frame is not None else ''
            index_finish_frame = index_finish_frame if index_finish_frame is not None else ''
            index_end_frame = index_end_frame if index_end_frame is not None else ''

            # Append a new row to the DataFrame
            df_frames_metrics.loc[len(df_frames_metrics)] = [
                index,
                index_operation_frame,
                index_response_frame,
                index_finish_frame,
                index_end_frame,
                response_time,
                finish_time
            ]

        # Write the results to a CSV file under the output directory
        if mkdir(os.path.join(self.output_dir, self.video_name)):
            df_frames_metrics.to_csv(os.path.join(self.output_dir, self.video_name, self.video_name + ScreenRecord.METRICS_CSV_FILE), index=False)

    def detect_user_operation_tap_with_cache(self):
        # detect
        current_file = os.path.abspath(__file__)
        project_path = os.path.abspath(os.path.join(current_file, "../.."))
        model = os.path.join(project_path, "src", "touch_model")
        touch_detection = TouchDetection(self.video_file, model)
        touch_detection.execute_with_cache()
        touch_detections = touch_detection.get_touch_detections()

        # Initialize tap list (binary tap flags per frame)
        frame_count = self.frame_count
        self.is_frame_has_tap_list = [False] * frame_count
        operation_frame_list = self.extract_first_frames(touch_detections)

        for frame_idx in operation_frame_list:
            if 0 <= frame_idx < frame_count:
                self.is_frame_has_tap_list[frame_idx] = True
            else:
                print(f"⚠️ Warning: frame index {frame_idx} out of range for video {self.video_name}")

        # Convert Frame objects into simplified dicts
        parsed = []
        for frame in touch_detections:
            taps = frame.get_screen_taps()
            for tap in taps:
                parsed.append({'id': frame.get_id(), 'taps': [str(tap)]})

        # Extract interaction types: "Tap" or "Swipe"
        self.interaction_types = extract_interaction_types(parsed)

    def extract_first_frames(self, touch_detections):
        frame_ids = [frame.get_id() for frame in touch_detections]
        groups = []
        current_group = [frame_ids[0]]

        for i in range(1, len(frame_ids)):
            if frame_ids[i] == frame_ids[i - 1] + 1:
                current_group.append(frame_ids[i])
            else:
                groups.append(current_group)
                current_group = [frame_ids[i]]
        groups.append(current_group)

        first_frames = [group[0] for group in groups]
        return first_frames

    def extract_interaction_types(self, touch_detections):
        from math import dist

        groups = []
        current_group = [touch_detections[0]]

        for i in range(1, len(touch_detections)):
            if touch_detections[i].id == touch_detections[i - 1].id + 1:
                current_group.append(touch_detections[i])
            else:
                groups.append(current_group)
                current_group = [touch_detections[i]]
        groups.append(current_group)

        types = []
        for group in groups:
            coords = []
            for frame in group:
                try:
                    tap_str = str(frame.taps[0])
                    x = float(tap_str.split("x=")[1].split(",")[0])
                    y = float(tap_str.split("y=")[1].replace("}", ""))
                    coords.append((x, y))
                except Exception as e:
                    print(f"⚠️ Failed to parse tap in frame {frame.id}: {e}")
            if len(coords) >= 2:
                movement = dist(coords[0], coords[-1])
                types.append("Swipe" if movement > 10 else "Tap")
            else:
                types.append("Tap")
        return types


    def calculate_perf_metrics(self):
        self.detect_user_operation_tap_with_cache()  # detect operation tap
        self.calculate_frames_similarity_with_cache()  # calculate similarity
        # self.write_all_frames_to_disk()  # write all frames to disk
        self.detect_outlier_frame()  # detect outlier frames of the videos
        self.split_user_interaction()  # split the video to user interactions and calculate performance metrics
        # self.save_user_interaction_frames()  # write user interaction frames as JPEG images to disc
        self.export_gui_responsiveness_metrics()  # calculate and write performance metrics to csv file
