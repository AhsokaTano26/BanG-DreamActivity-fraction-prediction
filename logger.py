import logging
from logging.handlers import RotatingFileHandler


# 全局变量，记录已经配置过的日志器名称
_configured_loggers = set()

def setup_logger(name="BDP", level=logging.DEBUG):
    """配置日志系统"""
    logger = logging.getLogger(name)
    if name in _configured_loggers:
        return logger
    logger.setLevel(level)
    # 如果日志器已经有处理器，先清除它们
    if logger.handlers:
        logger.handlers.clear()

    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 创建文件处理器
    file_handler = RotatingFileHandler("log/BDP.log", maxBytes=3 * 1024 * 1024, encoding='utf-8', mode='w')
    file_handler.setFormatter(formatter)

    # 添加处理器到日志器
    logger.addHandler(file_handler)

    return logger