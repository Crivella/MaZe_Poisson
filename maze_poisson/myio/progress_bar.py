from . import get_enabled

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *args, **kwargs):
        return x

class ProgressBar:
    def __init__(self, n: int):
        self.n = n
        
    def __iter__(self):
        if get_enabled():
            return iter(tqdm(range(self.n)))
        return iter(range(self.n))