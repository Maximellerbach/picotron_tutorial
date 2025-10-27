import torch
import torch.nn.functional as F
import torch.distributed as dist

import os

if __name__ == "__main__":
    device = torch.device("cpu")
    backend = "gloo"

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # init process group
    dist.init_process_group(backend=backend, rank=global_rank, world_size=world_size)

    tensor = torch.arange(2, dtype=torch.int64, device=device) + 1 + 2 * global_rank
    print(f"Before reduce | rank: {global_rank}, {tensor.numpy()}")

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"After reduce | rank: {global_rank}, {tensor.numpy()}")
    assert tensor.equal(torch.tensor([4, 6], dtype=torch.int64, device=device))