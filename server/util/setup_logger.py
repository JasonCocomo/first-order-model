import logging
import yaml
import logging.config
import os


def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.level = level
    return logger


def get_logconfig_path():
    logyaml_path = "./server/config/log.yml"
    return logyaml_path


def config_logger(level=logging.INFO, srv_id=None):
    logyaml_path = get_logconfig_path()
    if os.path.exists(logyaml_path):
        with open(file=logyaml_path, mode='r', encoding="utf-8") as file:
            try:
                logging_yaml = yaml.load(stream=file, Loader=yaml.FullLoader)
                if logging_yaml:
                    handlers = logging_yaml['handlers']
                    for key, value in handlers.items():
                        if 'filename' in value:
                            log_dir, log_file = (
                                os.path.split(value['filename']))
                            if srv_id is not None:
                                log_dir = os.path.join(log_dir, str(srv_id))
                            if not os.path.exists(log_dir):
                                os.makedirs(log_dir)
                            handlers[key]['filename'] = \
                                os.path.join(log_dir, log_file)
                    logging_yaml['handlers'] = handlers
                    # 配置logging日志：主要从文件中读取handler的配置、formatter（格式化日志样式）、logger记录器的配置
                    logging.config.dictConfig(config=logging_yaml)
                else:
                    logging.basicConfig(
                        level=level,
                        format='%(asctime)s %(levelname)s %(message)s')
            except Exception as e:
                print(e)
                print('Error in Logging Configuration. Using default configs')
                logging.basicConfig(
                    level=level,
                    format='%(asctime)s %(levelname)s %(message)s')
    else:
        logging.basicConfig(
            level=level,
            format='%(asctime)s %(levelname)s %(message)s')
        print('Failed to load configuration file. Using default configs')
