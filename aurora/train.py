import torch
import torch.optim.lr_scheduler as lr_scheduler
import tqdm

from aurora.reconstruction import Reconstruction
from aurora.loss import Loss, CombinedLoss


def train(
    recon: Reconstruction,
    loss_fns: dict[str, Loss] = {},
    loss_weights: dict[str, Loss] = {},
    num_iters: int = 1000,
    lr: float = 5e-5,
    weight_decay: float = 1.0,
    lr_step_size: int = 5000,
    lr_gamma: float = 0.1
) -> dict[str, list[float]]:
    
    recon.train()

    if len(loss_fns) == 0:
        raise ValueError("At least one loss function must be provided provided.")
    
    eval_loss = CombinedLoss(loss_fns, loss_weights)

    optimizer = torch.optim.Adam(recon.flux_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    tq = tqdm.trange(num_iters)
    for iter in tq:
        optimizer.zero_grad()
        
        loss = eval_loss(recon)
        loss.backward()

        optimizer.step()
        scheduler.step()

        tq.set_postfix(
            loss=f"{loss.item():.0f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}"
        )
    tq.close()

    return eval_loss.loss_history
