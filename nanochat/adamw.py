"""
Borrowed from modded-nanogpt. By Keller, @vagrawal, et al.
Not a general optimizer! But works for our specific use.
"""
import torch
import torch.distributed as dist
from torch import Tensor

"""
在优化器的step函数中完成梯度同步
采用类ZeRO2的优化，分别保存各自梯度和优化器状态

采用多种优化器，这里仅是adamw，与adam不同的是把权重衰减与梯度更新解耦
"""
class DistAdamW(torch.optim.Optimizer):
    """
    Distributed AdamW optimizer.
    In the style of ZeRO-2, i.e. sharded optimizer states and gradient reduction
    """
    def __init__(self, param_groups, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(param_groups, defaults)

    @torch.compile              # 可通过compile模式优化计算
    @torch.no_grad()            # 梯度更新时，需要关闭计算的梯度产生，调用no_grad修饰
    def step(self):
        rank = dist.get_rank()      # 获取当前进程rank编号
        world_size = dist.get_world_size()      # 获取所有进程数
        reduce_scatter_futures: list[torch.Future] = []
        all_reduce_futures: list[torch.Future] = []
        grad_slices = []
        for group in self.param_groups:     # 遍历权重
            params: list[Tensor] = group["params"]
            for base_i in range(len(params)):   # 遍历梯度tensor
                assert params[base_i].shape[0] % world_size == 0, f"First dim of parameter shape {params[base_i].shape} must be divisible by world size {world_size}"
                grad = params[base_i].grad
                rank_size = grad.shape[0] // world_size         # 在第0维切分tensor，第0维长度需要满足可被总进程数整除
                grad_slice = torch.empty_like(grad[:rank_size]) # 申请一块空间存储reduce scatter的结果
                reduce_scatter_futures.append(dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future())   # reduce scatter，通算异步进行
                grad_slices.append(grad_slice)          # 记录通信任务的列表

        idx = 0
        for group in self.param_groups:
            beta1, beta2 = group['betas']       # 取出权重对应的梯度更新的参数
            eps = group['eps']
            wd = group['weight_decay']
            params = group['params']            # 取出权重
            for base in range(len(params)):
                reduce_scatter_futures[idx].wait()  # 等待对应异步的reduce scatter通信完成，才能继续all gather
                p = params[base]
                rank_size = p.shape[0] // world_size    # 同样切分
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]        # 取出本进程对应的那部分权重
                lr = group['lr'] * getattr(p, "lr_mul", 1.0)
                state = self.state[p]               # 取出优化器状态
                g_slice = grad_slices[idx]          # 按照列表顺序取出reduce scatter得到的梯度
                # State init
                if not state:                       # state初始化，都为0，大小与本进程所存储梯度对应
                    state['step'] = torch.tensor(0, dtype=torch.int64, device=p.device)
                    state['exp_avg'] = torch.zeros_like(p_slice)
                    state['exp_avg_sq'] = torch.zeros_like(p_slice)
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1              # 累计step次数
                t = state['step']
                # weight decay
                if wd != 0:
                    eff_weight_decay = lr * wd * getattr(p, "wd_mul", 1.0)
                    p_slice.mul_(1 - eff_weight_decay)      # 先对权重进行权重衰减
                # update running averages
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)                  # beta1 * exp_avg + (1 - beta1) * grad
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)  # beta2 * exp_avg_sq + (1 - beta2) * grad * grad
                # bias corrections，误差纠正，确保符合期望
                bias1 = 1 - beta1 ** t
                bias2 = 1 - beta2 ** t
                # compute step
                denom = exp_avg_sq.sqrt().add_(eps)                                 # sqrt(exp_avg_sq) + eps
                step_size = lr * (torch.sqrt(bias2) / bias1)
                update = exp_avg.div(denom).mul_(step_size)                         # (exp_avg / (1 - beta1 ^ t)) / (sqrt(exp_avg_sq) + eps) * (1 - beta2 ^ t) * lr
                p_slice.add_(other=update, alpha=-1.0)                              # 更新权重, p = p - g
                idx += 1
                all_reduce_futures.append(dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future())  # 权重进行all gather，所有进程同步获得最新权重
        torch.futures.collect_all(all_reduce_futures).wait()    # 等待异步通信完成
