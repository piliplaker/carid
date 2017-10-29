# -*- coding: utf-8 -*
"""
这里提供一个修饰器,直接在控制台显示运行什么函数,调用了什么参数,运行了多少时间
用法 : 直接在被修饰函数上面写 @timelog 函数即可
"""
import logging
import sys
import time

# create logger
logger_name = "state_track"
logger = logging.getLogger(logger_name)
logger.setLevel(logging.DEBUG)

# create stream handler
sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.INFO)

# create file handler
log_path = "./dpoc.log"
fh = logging.FileHandler(log_path)
fh.setLevel(logging.WARN)

# create formatter
fmt = "%(asctime)-15s %(levelname)-8s  %(filename)s %(lineno)d  %(message)s"
datefmt = "%d %b %H:%M:%S "
formatter = logging.Formatter(fmt, datefmt)

# add handler and formatter to logger
fh.setFormatter(formatter)
sh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(sh)


def timelog(fn):  # this is a decporator
    def wraper(*arg):
        start = time.clock()
        ret = fn(*arg)
        elapsed = (time.clock() - start)
        logger.info("-->" + fn.__name__ + " runs for " + str(elapsed) + "s")
        return ret

    return wraper
