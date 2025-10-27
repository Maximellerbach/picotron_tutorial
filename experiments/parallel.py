import torch
import torch.nn.functional as F
import torch.distributed as dist

from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel
from torch.distributed.device_mesh import init_device_mesh

import os

class SimpleModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(4, 4)

    def forward(self, x):
        x = self.linear1(x)
        return x

if __name__ == "__main__":
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

    sharded = parallelize_module(
        model,
        mesh,
        {
            "linear1": ColwiseParallel(),
        },
    )

    input_tensor = torch.randn(2, 4, device=device)
    sharded_out = sharded(input_tensor) # shape (2, 2)
    print(f"rank: {global_rank}, output: {sharded_out.detach().numpy()}")

    # reconstruct full output for verification
    full_output = torch.zeros(2, 4, device=device)
    dist.all_gather(
        [full_output[:, i*2:(i+1)*2] for i in range(world_size)],
        sharded_out,
    )

    if global_rank == 0:
        print(f"Reconstructed full output: {full_output.detach().numpy()}")
