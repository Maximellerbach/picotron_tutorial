"""Custom tensor parallel example with a simple linear layer."""
import torch
import torch.distributed as dist
import os
from model import CombinedLinear, Attention

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

    model = Attention(4, 8).to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # training loop
    for it in range(100):

        pred = model(X)
        loss = loss_fn(pred, Y)
        loss.backward()

        optimizer.step() 
        optimizer.zero_grad()

        # in case of data parrallel,
        # sum gradients across data parallel ranks

        if rank == 0 and it % 10 == 0:
            print(f"Iter {it}, Loss: {loss.item()}")
    