
import time
def run_time(func):
    def inner(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        result = end_time - start_time
        print(func.__name__," time is %.3fs" % result)
        return res

    return inner