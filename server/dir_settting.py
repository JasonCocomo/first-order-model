import os
from functools import wraps
from server import config_dict


def check_file_exist(file_path):
    return os.path.exists(file_path)


def makedir_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        dirpath = func(*args, **kwargs)
        if not check_file_exist(dirpath):
            os.makedirs(dirpath)
        return dirpath
    return wrapper


cache_dir = config_dict.get('local.cache.dir')
oss_preprocess_root = config_dict.get('oss.preprocess.root')
oss_shifted_root = config_dict.get('oss.shifted.root')
preprocess_version = config_dict.get('preprocess.version')


source_video_dir = os.path.join(cache_dir, 'video')

if not check_file_exist(source_video_dir):
    os.makedirs(source_video_dir)


@makedir_wrapper
def get_task_label_input_image_dir(task_id, label):
    task_label_source_image_dir = os.path.join(
        get_task_dir(task_id), 'source', str(label))
    return task_label_source_image_dir


@makedir_wrapper
def get_task_dir(task_id):
    task_label_source_image_dir = os.path.join(cache_dir, 'task', task_id)
    return task_label_source_image_dir


@makedir_wrapper
def get_pre_processed_dir(template_id: str, pre_process_version: int = None):
    pre_processed_data_dir = os.path.join(
        cache_dir, 'processed', f"V_{pre_process_version}",
        template_id)
    return pre_processed_data_dir
