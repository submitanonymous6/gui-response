class UserInteraction:
    def __init__(self, start_frame, end_frame, frame_diffs_ssim_score, is_outlier_frame_list, is_frame_has_tap_list, fps, timestamps):
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.index_operation_frame = start_frame
        self.index_response_frame = start_frame
        self.index_finish_frame = start_frame

        self.response_time = None
        self.finish_time = None

        self.frame_diffs_ssim_score = frame_diffs_ssim_score
        self.is_outlier_frame_list = is_outlier_frame_list
        self.is_frame_has_tap_list = is_frame_has_tap_list
        self.fps = fps

        # time stamp
        self.timestamps = timestamps

    def get_start_frame(self):
        return self.start_frame

    def get_duration_milliseconds(self):
        return self.finish_time

    def calculate_perf_metrics_forest(self):
        # detect the outlier frame from the front to back to identify response frame
        for i in range(self.start_frame, self.end_frame, 1):
            if self.is_outlier_frame_list[i] == -1:  # the corresponding frame is outlier
                self.index_response_frame = i
                break

        # detect the outlier frame from the back to front to identify finish frame
        for i in range(self.end_frame, self.start_frame -1, -1):
            # if self.is_outlier_frame_list[i] == -1:  # the corresponding frame is outlier
            # if i < 0 or i >= len(self.is_outlier_frame_list):
            #     print(f"[PerfMetrics] start_frame={self.start_frame}, end_frame={self.end_frame}")
            #     print(f"[PerfMetrics] index_user_operation_frame: {self.index_operation_frame}")
            #     print(f"[PerfMetrics] len(is_outlier_frame_list)={len(self.is_outlier_frame_list)}")
            #     print('exception in is_outlier_frame_list')
            if self.is_outlier_frame_list[i] == -1:  # the corresponding frame is outlier
                self.index_finish_frame = i
                break

            # if i < 0 or i >= len(self.frame_diffs_ssim_score):
            #     print(f"[PerfMetrics] Invalid index: i={i}, start_frame={self.start_frame}, end_frame={self.end_frame}")
            #     continue
            #
            # if self.frame_diffs_ssim_score[i] <= self.FINISH_THRESHOLD:
            #     self.index_finish_frame = i
            #     break

        assert (self.index_operation_frame <= self.index_response_frame <= self.index_finish_frame)
        if (
                self.index_finish_frame < len(self.timestamps)
                and self.index_operation_frame < len(self.timestamps)
                and self.index_response_frame < len(self.timestamps)
        ):
            self.finish_time = self.timestamps[self.index_finish_frame] - self.timestamps[self.index_operation_frame]
            self.response_time = self.timestamps[self.index_response_frame] - self.timestamps[self.index_operation_frame]
        else:
            print(f"⚠️ Index out of bounds: operation={self.index_operation_frame}, response={self.index_response_frame}, finish={self.index_finish_frame}")
            self.finish_time = -1
            self.response_time = -1

        return self.index_operation_frame, self.index_response_frame, self.index_finish_frame, self.response_time, self.finish_time


    def calculate_perf_metrics(self):
        # detect the outlier frame from the front to back to identify response frame
        # for i in range(self.start_frame, self.end_frame, 1):
        #     if self.is_outlier_frame_list[i] == -1:  # the corresponding frame is outlier
        #         self.index_response_frame = i
        #         break
        for i in range(self.start_frame, self.end_frame + 1):
            if self.frame_diffs_ssim_score[i] <= self.RESPONSE_THRESHOLD:
                self.index_response_frame = i
                break

        # detect the outlier frame from the back to front to identify finish frame
        for i in range(self.end_frame, self.start_frame -1, -1):
            # if self.is_outlier_frame_list[i] == -1:  # the corresponding frame is outlier
            # if i < 0 or i >= len(self.is_outlier_frame_list):
            #     print(f"[PerfMetrics] start_frame={self.start_frame}, end_frame={self.end_frame}")
            #     print(f"[PerfMetrics] index_user_operation_frame: {self.index_operation_frame}")
            #     print(f"[PerfMetrics] len(is_outlier_frame_list)={len(self.is_outlier_frame_list)}")
            #     print('exception in is_outlier_frame_list')
            # if self.is_outlier_frame_list[i] == -1 and (not self.noise_frame(i)):  # the corresponding frame is outlier
            #     self.index_finish_frame = i
            #     break

            if i < 0 or i >= len(self.frame_diffs_ssim_score):
                print(f"[PerfMetrics] Invalid index: i={i}, start_frame={self.start_frame}, end_frame={self.end_frame}")
                continue

            if self.frame_diffs_ssim_score[i] <= self.FINISH_THRESHOLD:
                self.index_finish_frame = i
                break

        assert (self.index_operation_frame <= self.index_response_frame <= self.index_finish_frame)

        if (
                self.index_finish_frame < len(self.timestamps)
                and self.index_operation_frame < len(self.timestamps)
                and self.index_response_frame < len(self.timestamps)
        ):
            self.finish_time = self.timestamps[self.index_finish_frame] - self.timestamps[self.index_operation_frame]
            self.response_time = self.timestamps[self.index_response_frame] - self.timestamps[self.index_operation_frame]
        else:
            print(f"⚠️ Index out of bounds: operation={self.index_operation_frame}, response={self.index_response_frame}, finish={self.index_finish_frame}")
            self.finish_time = -1
            self.response_time = -1

        return self.index_operation_frame, self.index_response_frame, self.index_finish_frame, self.response_time, self.finish_time

    def get_perf_metrics(self):
        return self.index_operation_frame, self.index_response_frame, self.index_finish_frame, self.end_frame, self.response_time, self.finish_time

