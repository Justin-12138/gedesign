import logging
import os
from .const import log_path

class Logger:
    _instance = None

    def __new__(cls, log_file=log_path):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialize(log_file)
        return cls._instance

    def _initialize(self, log_file):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        self.logger = logging.getLogger("AppLogger")
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")

        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)

        if not self.logger.handlers:
            self.logger.addHandler(file_handler)

    def info(self, message):
        self.logger.info(message)

    def debug(self, message):
        self.logger.debug(message)

    def error(self, message):
        self.logger.error(message)

    def warning(self, message):
        self.logger.warning(message)


# 方便其他模块直接导入使用
logger = Logger()
