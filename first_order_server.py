import torch
import cv2
import numpy as np
from server.util.model_util import load_checkpoints
from animate import normalize_kp
from server import process_code
from server.resource.resouce_manager import ResourceManager
from scipy.spatial import ConvexHull
from server.util.image_common import resize_image

from argparse import ArgumentParser


class SourceFace:
    def __init__(self, face_index, bbox, face_img, kp_source=None):
        self.face_index = face_index
        self.bbox = bbox
        self.face_img = face_img
        self.kp_source = kp_source

    def update_kp_source(self, kp_source):
        self.kp_source = kp_source


class TaskMeta:

    def __init__(self, task_id,
                 template_id,
                 material_file_path,
                 ppd_file_path,
                 source_img_path):
        self.task_id = task_id
        self.template_id = template_id
        self.material_file_path = material_file_path
        self.ppd_file_path = ppd_file_path
        self.source_img_path = source_img_path


class DataFeeder:

    def __init__(self, kp_detector, device_id=0,
                 scale_factor=0.2, max_cache_count=10):
        self.device = 'cpu' if device_id < 0 else f'cuda:{device_id}'
        import face_alignment
        self.kp_detector = kp_detector
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
                                               flip_input=True,
                                               device=self.device)
        self.scale_factor = scale_factor
        self.resource_manager = ResourceManager(max_count=max_cache_count)

    def get_best_frame(self, source_face, frame_data_list):

        def ppd_normalize_kp(kp):
            kp = kp - kp.mean(axis=0, keepdims=True)
            area = ConvexHull(kp[:, :2]).volume
            area = np.sqrt(area)
            kp[:, :2] = kp[:, :2] / area
            return kp
        kp_source = self.fa.get_landmarks(source_face.face_img)[0]
        kp_source = ppd_normalize_kp(kp_source)
        frame_id = 0
        norm = float('inf')
        for frame_data in frame_data_list:
            kp_driving = frame_data.normalize_kp
            new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
            if new_norm < norm:
                norm = new_norm
                frame_id = frame_data.frame_id
        return frame_id

    def get_adjusted_bbox(self, bbox):
        x1, y1, x2, y2 = bbox[:4]

    def get_source_faces(self, source_img):
        bboxes = self.fa.face_detector.detect_from_image(source_img)
        source_faces = []
        for face_index, bbox in enumerate(bboxes):
            adjusted_bbox = self.get_adjusted_bbox(bbox)
            x1, y1, x2, y2 = adjusted_bbox
            face_img = source_img[x1:x2, y1:y2]
            source_face = SourceFace(face_index, adjusted_bbox, face_img)
            source_faces.append(source_face)
        return source_faces

    def update_kp_sources(self, source_faces):
        for source_face in source_faces:
            # bgr to rgb
            source = source_face.face_img[:, :, ::-1].copy()
            source = torch.tensor(source[np.newaxis].astype(
                np.float32)).permute(0, 3, 1, 2) / 255.
            kp_source = self.kp_detector(source)
            source_face.kp_source = kp_source

    def do_feeding(self,
                   task_meta: TaskMeta,
                   generator):

        source_img = cv2.imread(task_meta.source_img_path)
        source_img = resize_image(source_img)
        source_faces = self.get_source_faces(source_img)

        if len(source_faces) == 0:
            return process_code.NO_AVAILABLE_FACE

        self.update_kp_sources(source_faces)

        resource, frame_data_list = self.resource_manager.get_cache_data(
            task_meta.material_file_path, task_meta.ppd_file_path)

        predictions = []
        for source_face in source_faces:
            best_frame_id = self.get_best_frame(source_face, frame_data_list)

            best_frame = resource.material[best_frame_id]
            best_frame = best_frame[:, :, ::-1].copy()
            frame = torch.tensor(best_frame[np.newaxis].astype(np.float32)) \
                .permute(0, 3, 1, 2).cuda() / 255.
            kp_driving_initial = self.kp_detector(frame)

            relative = True
            adapt_movement_scale = True
            for frame_data in frame_data_list:
                kp_norm = normalize_kp(kp_source=source_face.kp_source,
                                       kp_driving=frame_data.kp_by_detector,
                                       kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                       use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
                out = self.generator(source_face.face_img,
                                     kp_source=source_face.kp_source,
                                     kp_driving=kp_norm)
                predictions.append(np.transpose(
                    out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
        return predictions


def main(args):
    material_file_path = ''
    ppd_file_path = ''
    source_img_path = ''
    task_meta = TaskMeta('11', 1, material_file_path,
                         ppd_file_path, source_img_path)
    config_path = args.config
    checkpoint_path = args.checkpoint
    generator, kp_detector = load_checkpoints(config_path, checkpoint_path)
    data_feeder = DataFeeder(kp_detector)
    output_imgs = data_feeder.do_feeding(task_meta, generator)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", default='vox-cpk.pth.tar',
                        help="path to checkpoint to restore")
    args = parser.parse_args()
    main(args)
