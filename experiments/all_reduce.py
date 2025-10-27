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
    
    tensor = torch.arange(2, dtype=torch.int64, device=device) + 1 + 2 * rank 
    print(f"Before reduce | rank: {rank}, {tensor.numpy()}")

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"After reduce | rank: {rank}, {tensor.numpy()}")
    assert tensor.equal(torch.tensor([4, 6], dtype=torch.int64, device=device))