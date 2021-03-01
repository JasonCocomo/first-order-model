# 处理正常
OK = 0
# 未知错误
UNKNOWN_ERROR = 1
# 参数错误
PARAMETER_ERROR = 2

# ===== 文件相关 =====
# 读取不了文件
CAN_NOT_READ_FILE = 11
# 下载文件失败
DOWNLOAD_FAILED = 12
# 上传文件失败
UPLOAD_FAILED = 13
# 生成文件失败
GENERATE_FAILED = 14
# 文件下载路径被锁
FILE_LOCKED = 15

# ==== 处理过程相关 ====
# 素材中没有人脸
NO_AVAILABLE_FACE = 101
# 像素值过大
RESOLUTION_ERR = 102
# 人脸检测异常
FACE_DETECT_ERR = 110
# 单帧监测到过多人脸，默认10个
EXCEED_FACE_LIMIT = 111
# 人脸68点Landmark提取异常
LANDMARKS68_ERR = 140
