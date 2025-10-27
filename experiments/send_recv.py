import torch
import torch.nn.functional as F
import torch.distributed as dist

import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="All Reduce Experiment")
    args = parser.parse_args()

    device = torch.device("cpu")
    backend = "gloo"

    # init process group
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()


    # rank 0 receives from all other ranks
    if rank == 0:
        recv_tensor = torch.zeros(2, dtype=torch.int64, device=device)
        for src in range(dist.get_world_size()):
            if src == rank: # skip self
                continue

            dist.recv(recv_tensor, src=src)
            print(f"recv | rank: {rank}, received from rank {src}, {recv_tensor.numpy()}")
    else:
        # print(f"send | rank: {rank}, {tensor.numpy()}")
        tensor = torch.arange(2, dtype=torch.int64, device=device) + 1 + 2 * rank 
        dist.send(tensor, dst=0)