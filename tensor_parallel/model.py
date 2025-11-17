import torch.distributed as dist
import torch
import torch.nn.functional as F

class ColwiseLinear(torch.nn.Module):
    def __init__(self, in_f, out_f, gather_output=True):
        super().__init__()
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.local_out = out_f // self.world
        self.W = torch.nn.Parameter(torch.randn(in_f, self.local_out) * 0.02)
        self.b = torch.nn.Parameter(torch.zeros(self.local_out))

        self.gather_output = gather_output

    def forward(self, x):  # x: [B, in_f]
        return F.linear(x, self.W, self.b)

class RowwiseLinear(torch.nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.local_in = in_f // self.world
        self.W = torch.nn.Parameter(torch.randn(self.local_in, out_f))
        self.b = torch.nn.Parameter(torch.zeros(out_f))

    def forward(self, x):
        output_parallel = F.linear(x, self.W)
        dist.all_reduce(output_parallel, op=dist.ReduceOp.SUM)
        return output_parallel + self.b


class CombinedLinear(torch.nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.col = ColwiseLinear(in_f, out_f)
        self.row = RowwiseLinear(out_f, out_f)

    def forward(self, x):
        x = self.col(x)
        x = self.row(x)
        return x
