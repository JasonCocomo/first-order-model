import numpy as np
import torch
from .resource import FrameData
from scipy.spatial import ConvexHull


class PPDGenerator:
    def __init__(self, kp_detector, fa):
        self.kp_detector = kp_detector
        self.fa = fa

    def generate(self, resource, ppd_file_path):
        def normalize_kp(kp):
            kp = kp - kp.mean(axis=0, keepdims=True)
            area = ConvexHull(kp[:, :2]).volume
            area = np.sqrt(area)
            kp[:, :2] = kp[:, :2] / area
            return kp
        ppd_file = {}
        ppd_file['width'] = resource.frame_width
        ppd_file['height'] = resource.frame_height
        ppd_file['frame_count'] = len(resource.material)
        frame_data_list = []
        for frame_id, frame in enumerate(resource.material):
            # bgr to rgb
            frame = frame[:, :, ::-1].copy()
            kp_by_landmarks = self.fa.get_landmarks(frame)[0]
            normed_kp_landmarks = normalize_kp(kp_by_landmarks)

            frame = torch.tensor(frame[np.newaxis].astype(np.float32)) \
                .permute(0, 3, 1, 2).cuda() / 255.
            kp_by_detector = self.kp_detector(frame)

            frame_data = FrameData(
                frame_id, kp_by_landmarks, normed_kp_landmarks, kp_by_detector)
            frame_data_list.append(frame_data)
        ppd_file['frame_data_list'] = frame_data_list
        return ppd_file
