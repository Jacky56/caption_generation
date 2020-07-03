from functools import wraps
import logging


class Logger(object):
    def __init__(self, filename, directory='./', level=logging.INFO):
        self.filename = filename
        self.log_setup = logging.getLogger(self.filename)
        self.fileHandler = logging.FileHandler('{}/{}.log'.format(directory, filename), mode='a')

        # set format
        self.formatter = logging.Formatter('%(levelname)s: %(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        self.fileHandler.setFormatter(self.formatter)

        self.log_setup.setLevel(level)
        self.log_setup.addHandler(self.fileHandler)

    def __call__(self, method):
        @wraps(method)
        def wrapper(*args, **kwargs):
            log = logging.getLogger(self.filename)
            msg = '{} : {} {}'.format(method.__name__, args, kwargs)
            result = method(*args, **kwargs)
            log.info(msg)
            return result

        return wrapper

# def logger(method):
#     import logging
#     import time
#     logging.basicConfig(filename='generator.log', level=logging.DEBUG)
#     @wraps(method)
#     def wrapper(*args, **kwargs):
#
#         msg = '{} - {}: {} {}'.format(time.time(), method.__name__,  args, kwargs)
#         logging.debug(msg)
#         return method(*args, **kwargs)
#     return wrapper

