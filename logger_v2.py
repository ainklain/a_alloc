
import logging.handlers
import os


def get_level(level_str):
    level_dict = {'debug': logging.DEBUG,
                  'info': logging.INFO,
                  'warning': logging.WARNING,
                  'error': logging.ERROR,
                  'critical': logging.CRITICAL}
    return level_dict[level_str.lower()]


class Logger:
    def __init__(self, name):
        # logger
        self.logger = logging.getLogger(name)
        self.log_maxbytes = 10 * 1024 * 1024
        self.log_backupcount = 10
        self.log_format = "%(asctime)s[%(levelname)s|%(name)s,%(lineno)s] %(message)s"

    def debug(self, *args, **kwargs):
        return self.logger.debug(*args, **kwargs)

    def critical(self, *args, **kwargs):
        return self.logger.critical(*args, **kwargs)

    def error(self, *args, **kwargs):
        return self.logger.error(*args, **kwargs)

    def warning(self, *args, **kwargs):
        return self.logger.warning(*args, **kwargs)

    def info(self, *args, **kwargs):
        return self.logger.info(*args, **kwargs)

    def set_handler(self, log_level, filename, outpath='./', use_stream_handler=False):
        self.reset_handler()

        # 로거 & 포매터 & 핸들러 생성
        if use_stream_handler:
            stream_formatter = logging.Formatter(self.log_format)
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(stream_formatter)
            self.logger.addHandler(stream_handler)

        formatter = logging.Formatter(self.log_format)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=os.path.join(outpath, '{}.log'.format(filename)),
            maxBytes=self.log_maxbytes,
            backupCount=self.log_backupcount)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # 로거 레벨 설정
        self.logger.setLevel(get_level(log_level))

    def reset_handler(self):
        while self.logger.hasHandlers():
            self.logger.removeHandler(self.logger.handlers[0])

