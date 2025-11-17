"""Custom tensor parallel example with a simple linear layer."""
import torch
import torch.distributed as dist
import os
from model import CombinedLinear

if __name__ == "__main__":
    device = torch.device("cpu")
    backend = "gloo"

    rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.distributed.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if rank == 0:
        X = torch.randn(2, 4)
        Y = torch.randn(2, 8)
    else:
        X = torch.empty(2, 4)
        Y = torch.empty(2, 8)

    dist.broadcast(X, src=0)
    dist.broadcast(Y, src=0)

    model = CombinedLinear(4, 8).to(device)
    loss_fn = torch.nn.MSELoss()
    pred = model(X)
    loss = loss_fn(pred, Y)
    loss.backward()

    print(f"Rank {rank}, Loss: {loss.item()}")