import torch
import torch.nn.functional as F
import torch.distributed as dist

from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
from torch.distributed.device_mesh import init_device_mesh

import os

class SimpleModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(4, 4)

    def forward(self, x):
        x = self.linear1(x)
        return x


def col_parallel_example(model, mesh):
    sharded = parallelize_module(
        model,
        mesh,
        {
            "linear1": ColwiseParallel(),
        },
    )

    if global_rank == 0:
        input_tensor = torch.randn(2, 4, device=device)
    else:
        input_tensor = torch.empty(2, 4, device=device)

    dist.broadcast(input_tensor, src=0)

    input_shard = torch.chunk(input_tensor, dist.get_world_size(), dim=0)[global_rank]
    sharded_out_partial = sharded(input_shard) 

    output_list = [torch.empty_like(sharded_out_partial) for _ in range(dist.get_world_size())]
    dist.all_gather(output_list, sharded_out_partial)
    sharded_out = torch.cat(output_list, dim=1)

    print(f"rank {global_rank} | output: {sharded_out.detach().numpy()}")


def row_parallel_example(model, mesh):
    sharded = parallelize_module(
        model,
        mesh,
        {
            "linear1": RowwiseParallel(),
        },
    )

    # full input on rank 0 only
    if global_rank == 0:
        full_input = torch.randn(2, 4, device=device)
    else:
        full_input = torch.empty(2, 4, device=device)

    # broadcast full input to all ranks
    dist.broadcast(full_input, src=0)

    input_shard = torch.chunk(full_input, dist.get_world_size(), dim=1)[global_rank]
    sharded_out = sharded(input_shard)
    print(f"shape after local forward: {sharded_out.shape}")

    # RowParallel outputs partial results that must be reduced across ranks
    dist.all_reduce(sharded_out)
    print(f"rank {global_rank} | output: {sharded_out.detach().numpy()}")


if __name__ == "__main__":
    """"""
    device = torch.device("cpu")
    backend = "gloo"

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # init process group
    dist.init_process_group(backend=backend, rank=global_rank, world_size=world_size)
    world_size = dist.get_world_size()

    model = SimpleModule().to(device)
    mesh = init_device_mesh("cpu", (world_size,))

    # col_parallel_example(model, mesh)
    row_parallel_example(model, mesh)