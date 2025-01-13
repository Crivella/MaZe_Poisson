import atexit

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
    MPI = None
    class MPIBase(metaclass=Singleton):
        """Dummy class for MPI."""
        def __init__(self):
            self.size = 1
            self.rank = 0
        def send_previous(self, *args):
            pass
        def recv_previous(self, *args):
            pass
        def send_next(self, *args):
            pass
        def recv_next(self, *args):
            pass
        def all_reduce(self, data, *args, **kwargs):
            return data
        def barrier(self):
            pass
        def finalize(self):
            pass
else:
    class MPIBase(metaclass=Singleton):
        """Base class for MPI classes."""
        # _instance = None
        # def __new__(class_, *args, **kwargs):
        #     if not isinstance(class_._instance, class_):
        #         class_._instance = object.__new__(class_, *args, **kwargs)
        #     return class_._instance

        def __init__(self):
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
            self.master = self.rank == 0

            self.prev = (self.rank - 1) % self.size
            self.nxt_ = (self.rank + 1) % self.size

            # atexit.register(self.finalize)

        def send_previous(self, data: np.ndarray):
            """Send data to the previous rank."""
            # print(f'RANK {self.rank} send_prev TO {self.prev} DATA {data[:5, :5]}')
            self.comm.Isend(data, dest=self.prev, tag=0)

        def recv_previous(self, buffer: np.ndarray):
            """Receive data from the previous rank."""
            self.comm.Recv(buffer, source=self.prev, tag=1)
            # print(f'RANK {self.rank} recv_prev FROM {self.prev} DATA {buffer[:5, :5]}')

        def send_next(self, data: np.ndarray):
            """Send data to the next rank non-blocking."""
            # self.comm.Send(data, dest=self.nxt_)
            # print(f'RANK {self.rank} send_nxt  TO {self.nxt_} DATA {data[:5, :5]}')
            self.comm.Isend(data, dest=self.nxt_, tag=1)

        def recv_next(self, buffer: np.ndarray):
            """Receive data from the next rank."""
            self.comm.Recv(buffer, source=self.nxt_, tag=0)
            # print(f'RANK {self.rank} recv_nxt  FROM {self.nxt_} DATA {buffer[:5, :5]}')

        def all_reduce(self, data: float, op=MPI.SUM):
            """Perform an all reduce operation."""
            return self.comm.allreduce(data, op)

        def barrier(self):
            """Barrier synchronization."""
            self.comm.Barrier()

        def finalize(self):
            """Finalize the MPI environment."""
            # self.comm.Barrier()
            # MPI.Finalize()
            if self.master:
                MPI.Finalize()