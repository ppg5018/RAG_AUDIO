import time
import functools

def timer_decorator(func):
    """
    A decorator that prints how long a function took to run.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"  {func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper