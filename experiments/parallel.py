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
        x = F.relu(x)
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

    input_tensor = torch.randn(4, 4, device=device)
    output = sharded(input_tensor)
    print(f"rank: {global_rank}, output: {output.detach().numpy()}")
