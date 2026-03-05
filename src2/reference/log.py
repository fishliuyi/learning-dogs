import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import time
from functools import wraps


class Log:
    def __init__(self, name: str, max_bytes: int = 50*1024*1024, backup_count: int = 5,
                 level: int = logging.INFO, log_dir: str = "logs"):
        """
        初始化日志系统

        Args:
            name: 日志文件名前缀
            max_bytes: 单个日志文件最大大小（字节）默认50MB
            backup_count: 保留的备份文件数量，默认5个
            level: 日志级别，默认INFO
            log_dir: 日志目录，默认"logs"
        """
        # 确保日志目录存在（使用绝对路径）
        log_path = Path(log_dir).resolve()
        log_path.mkdir(parents=True, exist_ok=True)
        filename = log_path / f"{name}.log"

        # 配置日志格式
        log_format = '%(asctime)s  %(name)s  %(levelname)s: %(message)s'

        # 创建或获取logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # 避免重复添加handler
        if not self.logger.handlers:
            # 创建RotatingFileHandler（按大小轮转）
            file_handler = RotatingFileHandler(
                filename=str(filename),
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8"
            )
            file_handler.setFormatter(logging.Formatter(log_format))

            # 创建控制台handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(log_format))

            # 添加handlers到logger
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

        # 避免重复添加handler到根logger
        self.logger.propagate = False

    def debug(self, message: str):
        """记录DEBUG级别日志"""
        self.logger.debug(message)

    def info(self, message: str):
        """记录INFO级别日志"""
        self.logger.info(message)

    def warning(self, message: str):
        """记录WARNING级别日志"""
        self.logger.warning(message)

    def error(self, message: str):
        """记录ERROR级别日志"""
        self.logger.error(message)

    def critical(self, message: str):
        """记录CRITICAL级别日志"""
        self.logger.critical(message)

    def exception(self, message: str):
        """记录异常信息（包含堆栈跟踪）"""
        self.logger.exception(message)

    def set_level(self, level: int):
        """动态设置日志级别"""
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)

    def performance_monitor(self, operation_name: str):
        """
        性能监控装饰器

        Args:
            operation_name: 操作名称

        Usage:
            @log.performance_monitor("特征提取")
            def extract_features():
                # 你的代码
                pass
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                self.info(f"{operation_name} 启动")
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    end_time = time.time()
                    duration = end_time - start_time
                    self.info(f"{operation_name} 完成，耗时: {duration:.2f}秒")
                    return result
                except Exception as e:
                    end_time = time.time()
                    duration = end_time - start_time
                    self.error(
                        f"{operation_name} 失败，耗时: {duration:.2f}秒，错误: {str(e)}")
                    raise
            return wrapper
        return decorator


# 全局日志工厂函数
def get_logger(name: str, **kwargs) -> logging.Logger:
    """
    获取日志记录器的便捷函数

    Args:
        name: 日志名称
        **kwargs: 传递给Log构造函数的参数

    Returns:
        logging.Logger实例
    """
    return Log(name, **kwargs).logger


# 预定义的日志级别
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}
