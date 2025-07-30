from . import get_enabled_io

try:
    from tqdm import tqdm
except ImportError:
    class ProgressBar:
        def __init__(self, n: int):
            self.n = n
            self.current = 0
            
        def __iter__(self):
            return self
        def __next__(self):
            if self.current >= self.n:
                raise StopIteration
            self.current += 1
            if self.current % 100 == 0:
                if get_enabled_io():
                    print(f"Progress: {self.current}/{self.n}")
            return self.current - 1
else:
    class ProgressBar:
        def __init__(self, n: int):
            self.n = n
            
        def __iter__(self):
            if get_enabled_io():
                return iter(tqdm(range(self.n)))
            return iter(range(self.n))
