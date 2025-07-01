from typing import Optional, Iterable, Callable, Dict, List, Union

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import tqdm


def train(
        *,
        step: Callable[[], Dict[str, float]],
        num_iters: int,
        start_iter: int = 0,
        optimizer: Optional[torch.optim.Optimizer] = None,
        modules: Optional[Union[nn.Module, Iterable[nn.Module]]] = None,
        lr: float = 5e-5,
        weight_decay: float = 1.0,
        scheduler: Optional[lr_scheduler.LRScheduler] = None,
        progress_bar: bool = True
    ) -> Dict[str, List[float]]:
    """
    Train reconstruction with flexible loss terms.
    
    Args:
        [TODO]
    
    Returns:
        Dictionary mapping loss term names to their training history
    """
    if optimizer is None:
        if modules is None:
            raise ValueError("Trained modules must be provided for default optimizer")
        if isinstance(modules, nn.Module):
            params = modules.parameters()
        else:
            params = set()
            for module in modules:
                params = params.union(module.parameters())
        optimizer = torch.optim.Adam(
            params=params,
            lr=lr,
            weight_decay=weight_decay
        )
    
    if progress_bar:
        iterator = tqdm.trange(start_iter, start_iter + num_iters)
    else:
        iterator = range(start_iter, start_iter + num_iters)
    
    history = {}
    
    for iter in iterator:
        optimizer.zero_grad()
        
        loss_dict = step()
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        for name, val in loss_dict.items():
            loss_hist = history.setdefault(name, [])
            loss_hist.append(val)
        
        if progress_bar:
            loss_str = " ".join([f"{name}:{val:.2f}" for name, val in loss_dict.items()])
            postfix = [loss_str]
            if scheduler is not None:
                last_lr = scheduler.get_last_lr()[0]
                postfix.append(f"lr:{last_lr:.2e}")
            iterator.set_postfix_str(" ".join(postfix))
    
    return history

