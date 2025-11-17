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
