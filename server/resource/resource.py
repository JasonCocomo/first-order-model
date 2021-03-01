from abc import ABC, abstractmethod
import cv2
import logging
import numpy as np
from server.util.setup_logging import get_logger

logger = get_logger(__name__, level=logging.DEBUG)

RESOLUTION_LIMIT = 1280 * 720


class ResolutionError(Exception):
    def __init__(self):
        super().__init__("Resolution Error, greater than 1280 * 720. ")


class MaxImageSizeExceedError(Exception):
    def __init__(self, max_img_size, current_image_size):
        super().__init__(
            "Max Image Size Exceed Error, config image size: "
            f"{max_img_size}, current image size {current_image_size}")


class FrameData:
    def __init__(self, frame_id, landmark68,
                 normed_kp_landmarks, kp_by_detector):
        self.frame_id = frame_id
        self.landmark68 = landmark68
        self.normed_kp_landmarks = normed_kp_landmarks
        self.kp_by_detector = kp_by_detector


class Resource(ABC):
    def __init__(self, file_path):
        self.file_path = file_path
        self.material = []

    def get_frame(self, idx):
        idx = int(idx)
        return self.material[idx]

    def get_count(self):
        return len(self.material)


class VideoResource(Resource):
    """
    Video Resource Management
    """

    def __init__(self, material_file_path):
        super().__init__(file_path=material_file_path)

    def load_material(self):
        """
        Load a full list of frames in memory.

        """
        cap = cv2.VideoCapture(self.file_path)
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.frame_width * self.frame_height > RESOLUTION_LIMIT:
            raise ResolutionError()

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        res = []
        while True:
            ret, frame = cap.read()
            if ret is False:
                break
            res.append(frame[:, :, 0:3])
        self.material = res
        self.total_frames = len(res)
        cap.release()
        return self.material
