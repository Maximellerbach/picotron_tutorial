import torch
import torch.distributed as dist
from parallel import (
    AllGather, AllReduce
)


class ColumnParallelLinear(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(ColumnParallelLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight = torch.nn.Parameter(
            torch.randn(input_size, output_size // dist.get_world_size()),
            requires_grad=True,
        )
        self.weight.retain_grad()

    def forward(self, X):
        return (X @ self.weight)


class RowParallelLinear(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(RowParallelLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight = torch.nn.Parameter(
            torch.randn(output_size // dist.get_world_size(), input_size),
            requires_grad=True,
        )
        self.weight.retain_grad()

    def forward(self, X):
        Y = (X @ self.weight)
        Y = AllReduce.apply(Y)
        return Y


class CombinedLinear(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(CombinedLinear, self).__init__()
        self.col_linear = ColumnParallelLinear(input_size, output_size)
        self.row_linear = RowParallelLinear(output_size, output_size)

    def forward(self, X):
        Y_col = self.col_linear.forward(X)
        Y = self.row_linear.forward(Y_col)
        return Y
