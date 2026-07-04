from mpi4py import MPI
import numpy as np

class Communicator(object):
    def __init__(self, comm: MPI.Comm):
        self.comm = comm
        self.total_bytes_transferred = 0

    def Get_size(self):
        return self.comm.Get_size()

    def Get_rank(self):
        return self.comm.Get_rank()

    def Barrier(self):
        return self.comm.Barrier()

    def Allreduce(self, src_array, dest_array, op=MPI.SUM):
        assert src_array.size == dest_array.size
        src_array_byte = src_array.itemsize * src_array.size
        self.total_bytes_transferred += src_array_byte * 2 * (self.comm.Get_size() - 1)
        self.comm.Allreduce(src_array, dest_array, op)

    def Allgather(self, src_array, dest_array):
        src_array_byte = src_array.itemsize * src_array.size
        dest_array_byte = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_array_byte * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_array_byte * (self.comm.Get_size() - 1)
        self.comm.Allgather(src_array, dest_array)

    def Reduce_scatter(self, src_array, dest_array, op=MPI.SUM):
        src_array_byte = src_array.itemsize * src_array.size
        dest_array_byte = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_array_byte * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_array_byte * (self.comm.Get_size() - 1)
        self.comm.Reduce_scatter_block(src_array, dest_array, op)

    def Split(self, key, color):
        return __class__(self.comm.Split(key=key, color=color))

    def Alltoall(self, src_array, dest_array):
        nprocs = self.comm.Get_size()

        # Ensure that the arrays can be evenly partitioned among processes.
        assert src_array.size % nprocs == 0, (
            "src_array size must be divisible by the number of processes"
        )
        assert dest_array.size % nprocs == 0, (
            "dest_array size must be divisible by the number of processes"
        )

        # Calculate the number of bytes in one segment.
        send_seg_bytes = src_array.itemsize * (src_array.size // nprocs)
        recv_seg_bytes = dest_array.itemsize * (dest_array.size // nprocs)

        # Each process sends one segment to every other process (nprocs - 1)
        # and receives one segment from each.
        self.total_bytes_transferred += send_seg_bytes * (nprocs - 1)
        self.total_bytes_transferred += recv_seg_bytes * (nprocs - 1)

        self.comm.Alltoall(src_array, dest_array)

    def myAllreduce(self, src_array, dest_array, op=MPI.SUM):
        """
        A manual implementation of all-reduce using a reduce-to-root
        followed by a broadcast.
        
        Each non-root process sends its data to process 0, which applies the
        reduction operator (by default, summation). Then process 0 sends the
        reduced result back to all processes.
        
        The transfer cost is computed as:
          - For non-root processes: one send and one receive.
          - For the root process: (n-1) receives and (n-1) sends.
        """
        rank = self.comm.Get_rank()
        size = self.comm.Get_size()

        # Map the MPI op to a numpy reduction so we can combine locally at root.
        def reduce_pair(acc, other):
            if op == MPI.SUM:
                return acc + other
            if op == MPI.MIN:
                return np.minimum(acc, other)
            if op == MPI.MAX:
                return np.maximum(acc, other)
            if op == MPI.PROD:
                return acc * other
            raise NotImplementedError(f"Unsupported op: {op}")

        if rank == 0:
            # Root starts with its own data, then folds in every other rank.
            acc = np.array(src_array, copy=True)
            recv_buf = np.empty_like(src_array)
            for src in range(1, size):
                self.comm.Recv(recv_buf, source=src, tag=0)
                self.total_bytes_transferred += recv_buf.itemsize * recv_buf.size
                acc = reduce_pair(acc, recv_buf)
            # Broadcast the reduced result back to everyone (including self).
            dest_array[...] = acc
            for dst in range(1, size):
                self.comm.Send(acc, dest=dst, tag=1)
                self.total_bytes_transferred += acc.itemsize * acc.size
        else:
            # Non-root: send our contribution to root, receive the final result.
            self.comm.Send(src_array, dest=0, tag=0)
            self.total_bytes_transferred += src_array.itemsize * src_array.size
            self.comm.Recv(dest_array, source=0, tag=1)
            self.total_bytes_transferred += dest_array.itemsize * dest_array.size

    def myAlltoall(self, src_array, dest_array):
        """
        A manual implementation of all-to-all where each process sends a
        distinct segment of its source array to every other process.
        
        It is assumed that the total length of src_array (and dest_array)
        is evenly divisible by the number of processes.
        
        The algorithm loops over the ranks:
          - For the local segment (when destination == self), a direct copy is done.
          - For all other segments, the process exchanges the corresponding
            portion of its src_array with the other process via Sendrecv.

        The total data transferred is updated for each pairwise exchange.
        """
        rank = self.comm.Get_rank()
        size = self.comm.Get_size()

        seg_len = src_array.size // size
        # Flat views so segment i is src[i*seg_len:(i+1)*seg_len].
        src_flat = src_array.reshape(-1)
        dst_flat = dest_array.reshape(-1)

        for other in range(size):
            s0, s1 = other * seg_len, (other + 1) * seg_len
            if other == rank:
                # Local segment: direct copy, no network transfer.
                dst_flat[s0:s1] = src_flat[s0:s1]
            else:
                send_seg = np.ascontiguousarray(src_flat[s0:s1])
                recv_seg = np.empty(seg_len, dtype=dest_array.dtype)
                # Exchange: we send the segment destined for `other`, and receive
                # from `other` the segment they destined for us.
                self.comm.Sendrecv(
                    sendbuf=send_seg, dest=other, sendtag=0,
                    recvbuf=recv_seg, source=other, recvtag=0,
                )
                dst_flat[s0:s1] = recv_seg
                self.total_bytes_transferred += send_seg.itemsize * send_seg.size
                self.total_bytes_transferred += recv_seg.itemsize * recv_seg.size

        # Write back in case dest_array wasn't a view (reshape may copy).
        dest_array[...] = dst_flat.reshape(dest_array.shape)
