import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger():
    """配置工程级日志模块"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger("SmartLock")
    logger.setLevel(logging.DEBUG)

    # 格式化器：包含时间、级别、模块名和具体消息
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'
    )

    # 文件处理器：每个文件最大 5MB，保留 5 个备份
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "doorlock.log"),
        maxBytes=5*1024*1024, 
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)

    # 控制台处理器：方便调试查看
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 全局初始化
log = setup_logger()