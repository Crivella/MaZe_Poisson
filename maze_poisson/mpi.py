import numpy as np


class Singleton(type):
    """Singleton metaclass."""
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

try:
    from mpi4py import MPI
except ImportError:
    class MPIBase(metaclass=Singleton):
        """Dummy class for MPI."""
        def __init__(self):
            self.comm = None
            self.size = 1
            self.rank = 0
            self.master = True
            self.prev = 0
            self.nxt_ = 0
        def __bool__(self):
            return False
        def get_bot_top(self, data: np.ndarray, bot: np.ndarray = None, top: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
            """Get the top and bottom slices of the data."""
            if bot is not None:
                bot[:] = data[-1]
            else:
                bot = data[-1]
            if top is not None:
                top[:] = data[0]
            else:
                top = data[0]
            return bot, top
        def all_reduce(self, data, *args, **kwargs):
            return data
        def all_reduce_inplace(self, data, *args, **kwargs):
            return data
        def get_n_start(self, n: int) -> int:
            return 0
        def get_n_loc(self, n: int) -> int:
            return n
else:
    class MPIBase(metaclass=Singleton):
        """Base class for MPI classes."""
        def __init__(self):
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
            self.__bool = self.size > 1
            self.master = self.rank == 0

            self.prev = (self.rank - 1) % self.size
            self.nxt_ = (self.rank + 1) % self.size

            self.comm_address = MPI._addressof(self.comm)

            # div = self.N // self.mpi.size
            # rem = self.N % self.mpi.size
            # self.N_loc = div + (1 if self.mpi.rank < rem else 0)
            # self.N_loc_start = div * self.mpi.rank + min(self.mpi.rank, rem)

        def __bool__(self):
            return self.__bool

        def send_previous(self, data: np.ndarray):
            """Send data to the previous rank."""
            self.comm.Isend(data, dest=self.prev, tag=0)

        def send_next(self, data: np.ndarray):
            """Send data to the next rank non-blocking."""
            self.comm.Isend(data, dest=self.nxt_, tag=1)

        def recv_previous(self, buffer: np.ndarray):
            """Receive data from the previous rank."""
            self.comm.Recv(buffer, source=self.prev, tag=1)

        def recv_next(self, buffer: np.ndarray):
            """Receive data from the next rank."""
            self.comm.Recv(buffer, source=self.nxt_, tag=0)

        def get_bot_top(self, data: np.ndarray, bot: np.ndarray = None, top: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
            """Get the top and bottom slices of the data."""
            if bot is None:
                bot = np.empty_like(data[0])
            if top is None:
                top = np.empty_like(data[0])
            self.comm.Sendrecv(
                data[-1], dest=self.nxt_, sendtag=0,
                recvbuf=bot, source=self.prev, recvtag=0
            )
            self.comm.Sendrecv(
                data[0], dest=self.prev, sendtag=1,
                recvbuf=top, source=self.nxt_, recvtag=1
            )
            return bot, top

        def all_reduce(self, data, op=MPI.SUM):
            """Perform an all reduce operation."""
            return self.comm.allreduce(data, op)

        def all_reduce_inplace(self, data: np.ndarray, op=MPI.SUM):
            """Perform an all reduce operation in place."""
            self.comm.Allreduce(MPI.IN_PLACE, [data, MPI.DOUBLE], op)

        def get_n_start(self, n: int) -> int:
            div = n // self.size
            rem = n % self.size
            return div * self.rank + min(self.rank, rem)
        def get_n_loc(self, n: int) -> int:
            div = n // self.size
            rem = n % self.size
            return div + (1 if self.rank < rem else 0)
