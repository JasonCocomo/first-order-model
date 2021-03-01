import cv2
import numpy as np
import logging
from server.util.setup_logging import get_logger

logger = get_logger(__name__, level=logging.DEBUG)
try:
    import PyNvCodec as nvc
    using_pynvcodec = True
except:
    using_pynvcodec = False
    logger.info('VPF failed to init, please check is it installed.')


class MixedVideoWriter:

    def __init__(self, counterpart_mod, no_audio_file_path, size, fps, device_id=-1):
        self.no_audio_file_path = no_audio_file_path
        if counterpart_mod:
            size = (size[0] * 2, size[1])
        if device_id < 0 or not using_pynvcodec:
            self.use_opencv = True
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            self.vout = cv2.VideoWriter()
            self.vout.open(no_audio_file_path, fourcc, fps, size, True)
        else:
            self.use_opencv = False
            width = size[0]
            height = size[1]
            res = f'{width}x{height}'
            self.nvEnc = nvc.PyNvEncoder(
                {'preset': 'hq', 'codec': 'h264', 's': res, 'fps': str(fps)}, device_id)
            self.nvUpl = nvc.PyFrameUploader(int(width), int(
                height), nvc.PixelFormat.YUV420, device_id)
            self.nvCvt = nvc.PySurfaceConverter(int(width), int(
                height), nvc.PixelFormat.YUV420, nvc.PixelFormat.NV12, device_id)
            self.vout = open(no_audio_file_path, "wb")
        self.open_status = True

    def write(self, bgr_frame):
        if self.use_opencv:
            self.vout.write(bgr_frame)
        else:
            success, encByteArray = self.convert_numpy_to_nv12(bgr_frame)
            if not success:
                return False
            self.vout.write(encByteArray)
        return True

    def close(self):
        if not self.open_status:
            return
        if self.use_opencv:
            self.vout.release()
        else:
            self.vout.close()
        self.open_status = False

    def convert_numpy_to_nv12(self, bgr_frame):
        cv2.setNumThreads(3)
        rawFrameYUV = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2YUV_I420)
        rawSurface = self.nvUpl.UploadSingleFrame(rawFrameYUV)
        if (rawSurface.Empty()):
            logger.error(
                f'VPF failed to upload video frame to GPU. file: {self.no_audio_file_path}')
            return False, None
        cvtSurface = self.nvCvt.Execute(rawSurface)
        if (cvtSurface.Empty()):
            logger.error(
                f'VPF failed to do color conversion. file: {self.no_audio_file_path}')
            return False, None
        encFrame = np.ndarray(shape=(0), dtype=np.uint8)
        success = self.nvEnc.EncodeSingleSurface(
            cvtSurface, encFrame, sync=False)
        if success:
            encByteArray = bytearray(encFrame)
            return True, encByteArray
        return False, None
