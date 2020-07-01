from functools import wraps


class Logger(object):
    def __init__(self, filename):
        self.filename = filename

    def __call__(self, method):

        @wraps(method)
        def wrapper(*args, **kwargs):
            import logging
            import datetime
            logging.basicConfig(filename='{}.log'.format(self.filename), level=logging.DEBUG)
            msg = '{} - {}: {} {}'.format(datetime.datetime.now(), method.__name__, args, kwargs)
            logging.debug(msg)
            return method(*args, **kwargs)

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