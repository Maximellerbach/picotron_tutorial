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
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_tensors, tensor)

    print(f"Gathered: | rank: {rank}, {[t.numpy() for t in gathered_tensors]}")
