import torch.distributed as dist
import torch
import torch.nn.functional as F


class AllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = input.clone()
        if dist.get_world_size() == 1:
            return output

        dist.all_reduce(output, op=dist.ReduceOp.SUM)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        world_size = dist.get_world_size()
        gathered = [torch.zeros_like(input) for _ in range(world_size)]
        dist.all_gather(gathered, input)
        return torch.cat(gathered, dim=-1)
    
    @staticmethod
    def backward(ctx, grad_output):
        world_size = dist.get_world_size()
        if world_size == 1:
            return grad_output

        grad_chunks = torch.chunk(grad_output, world_size, dim=-1)
        return grad_chunks[dist.get_rank()].contiguous()


def split_tensor(tensor, dim):
    world_size = dist.get_world_size()
    local_rank = dist.get_rank()
    return torch.chunk(tensor, world_size, dim)[local_rank].contiguous()

def row_parallel_linear_forward(X_row, W_row):
    Y = X_row @ W_row
    Y = AllReduce.apply(Y)
    return Y

def row_parallel_linear_backward(Y_grad, X_row, W_row, use_all_gather=True):
    """
    TODO: fix this

    #Given: Y = X @ W
    # We get the derivatives: 
    dY/dX = W
    dY/dW = X

    # Applying chain rule:
    dL/dX = dL/dY @ dY/dX = dL/dY @ W
    dL/dW = dL/dY @ dY/dW = dL/dY @ X
    """
    # 1. Compute dL/dX_row = dL/dY @ dY/dX = dL/dY @ W_row
    X_local_grad = Y_grad @ W_row.t()
    if use_all_gather:
        X_grad = torch.zeros_like(X_local_grad)
        dist.all_gather(X_grad, X_local_grad)
        X_grad = torch.cat(X_grad, dim=1)
    else:
        X_grad = X_local_grad

    # 2. Compute dL/dW_row = dL/dY @ dY/dW = dL/dY @ X_row
    W_grad = Y_grad.t() @ X_row

    return X_grad, W_grad

def column_parallel_linear_forward(X, W_col):
    Y_col = X @ W_col
    return Y_col

def column_parallel_linear_backward(Y_local_grad, X, W_col):
    """
    TODO: fix this

    #Given: Y = X @ W
    # We get the derivatives: 
    dY/dX = W
    dY/dW = X

    # Applying chain rule:
    dL/dX = dL/dY @ dY/dX = dL/dY @ W
    dL/dW = dL/dY @ dY/dW = dL/dY @ X
    """
    # 1. Compute dL/dX = dL/dY_col @ W_col
    #    Each rank computes partial gradient, then all_reduce to sum contributions
    X_grad = Y_local_grad @ W_col.t()
    # Use custom AllReduce function for autograd support
    X_grad = AllReduce.apply(X_grad)
    
    # 2. Compute dL/dW_col = dL/dY_col @ X
     #    Each rank computes gradient for its column slice
    W_local_grad = Y_local_grad.t() @ X
    
    return X_grad, W_local_grad

