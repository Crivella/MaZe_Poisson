import signal
import sys

from . import get_enabled_io

try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False

try:
    from rich.progress import Progress
    from rich.text import Text
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        MofNCompleteColumn,
        ProgressColumn,
    )
    HAVE_RICH = True

    class IterationSpeedColumn(ProgressColumn):
        """Renders iteration speed."""

        def render(self, task):
            if task.finished:
                return Text(f"{self.last_seppd:>6.2f} it/s", style="green")
            self.last_seppd = speed = task.speed or 0.0
            
            return Text(f"{speed:>6.2f} it/s", style="cyan")

except ImportError:
    HAVE_RICH = False

class BaseProgressBar:
    def __init__(self, n: int, description: str = "Progress"):
        self.n = n
        self.description = description
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

class TQDMProgressBar:
    def __init__(self, n: int, description: str = "Progress"):
        self.n = n
        self.description = description
        
    def __iter__(self):
        if get_enabled_io():
            return iter(tqdm(range(self.n)))
        return iter(range(self.n))

class RichProgressBar:
    def __init__(self, n: int, description: str = "Progress"):
        self.n = n
        self.current = 0
        self.description = description
        self.task = None
        self.progress = None

        for sig in [signal.SIGINT]:
            signal.signal(sig, lambda s, f: self.terminate(s, f))
    
    def __iter__(self):
        if get_enabled_io():
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                IterationSpeedColumn(),
                TimeElapsedColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            )
            self.task = self.progress.add_task(self.description, total=self.n)
            self.progress.start()
            return self
        return iter(range(self.n))

    def __next__(self):
        if self.current >= self.n:
            self.clear()
            raise StopIteration

        self.progress.update(self.task, advance=1)
        res = self.current
        self.current += 1
        return res

    def terminate(self, signum, frame):
        print("\nProgress interrupted. Cleaning up...")
        self.clear()
        sys.exit(signum)

    def clear(self):
        if self.progress is not None:
            self.progress.stop()
        self.progress = None
        self.task = None

if HAVE_RICH:
    ProgressBar = RichProgressBar
elif HAVE_TQDM:
    ProgressBar = TQDMProgressBar
else:
    ProgressBar = BaseProgressBar