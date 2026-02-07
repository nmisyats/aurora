from typing import Optional, Iterable, Callable, Dict, List, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import tqdm


LossEvalOutput = Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]
LossEvalFn = Callable[[nn.Module], LossEvalOutput]

def train_loop(
    model: nn.Module,
    iter_loss: LossEvalFn,
    optimizer: Optimizer,
    num_iters: int,
    start_iter: int = 0,
    lr_scheduler: Optional[LRScheduler] = None,
    progress_bar: bool = True
):
    stop_iter = start_iter + num_iters
    if progress_bar:
        iterator = tqdm.trange(start_iter, stop_iter)
    else:
        iterator = range(start_iter, stop_iter)
    
    history = {}

    for iter in iterator:
        optimizer.zero_grad()
        
        step_result = iter_loss(model)
        if isinstance(step_result, tuple):
            loss, loss_dict = step_result
        else:
            loss = step_result
            loss_dict = {"loss": loss.item()}
        
        loss.backward()
        
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()
        
        for name, val in loss_dict.items():
            loss_hist = history.setdefault(name, [])
            loss_hist.append(val)
        
        if progress_bar:
            loss_str = " ".join([f"{name}:{val:.2f}" for name, val in loss_dict.items()])
            postfix = [loss_str]
            if lr_scheduler is not None:
                last_lr = lr_scheduler.get_last_lr()[0]
                postfix.append(f"lr:{last_lr:.2e}")
            iterator.set_postfix_str(" ".join(postfix))
    
    return history

def train(
    model: nn.Module,
    iter_loss: LossEvalFn,
    num_iters: int,
    start_iter: int = 0,
    lr: float = 5e-5,
    weight_decay: float = 1.0,
    progress_bar: bool = True
):
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    return train_loop(
        model=model,
        iter_loss=iter_loss,
        optimizer=optimizer,
        num_iters=num_iters,
        start_iter=start_iter,
        progress_bar=progress_bar
    )
