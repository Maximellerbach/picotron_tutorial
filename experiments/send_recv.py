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

    # rank 0 receives from all other ranks
    if global_rank == 0:
        recv_tensor = torch.zeros(2, dtype=torch.int64, device=device)
        for src in range(dist.get_world_size()):
            if src == global_rank: # skip self
                continue

            dist.recv(recv_tensor, src=src)
            print(f"recv | rank: {global_rank}, received from rank {src}, {recv_tensor.numpy()}")
    else:
        # print(f"send | rank: {global_rank}, {tensor.numpy()}")
        tensor = torch.arange(2, dtype=torch.int64, device=device) + 1 + 2 * global_rank
        dist.send(tensor, dst=0)