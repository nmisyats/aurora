import torch
import torch.optim.lr_scheduler as lr_scheduler
import tqdm

from aurora.losses.loss_term import LossTerm


def minimize(
        *loss_terms: LossTerm,
        iters: int = 1000,
        lr: float = 5e-5,
        weight_decay: float = 1.0,
        lr_step: int = 1000,
        lr_decay: float = 0.5,
        progress_bar: bool = True
    ) -> dict[str, list[float]]:
    """
    Train reconstruction with flexible loss terms.
    
    Args:
        loss_terms: List of LossTerm objects defining the loss function
        ...: [TODO]
    
    Returns:
        Dictionary mapping loss term names to their training history
    """
    if not loss_terms:
        raise ValueError("At least one loss term must be provided")
    
    params = set()
    for term in loss_terms:
        params = params.union(term.parameters())
    
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_decay)
    
    total_history = []
    iterator = tqdm.trange(iters) if progress_bar else range(iters)
    
    for iter in iterator:
        optimizer.zero_grad()
        
        # Compute all loss terms
        term0 = loss_terms[0]
        total_loss = term0(term0.sample_batch())
        for term in loss_terms[1:]:
            if term.weight == 0.0:
                continue
            term_loss = term(term.sample_batch())
            total_loss = total_loss + term_loss
        
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_history.append(total_loss.item())
        
        if progress_bar:
            # Show recent losses in progress bar
            recent_losses = {term.name: term.history[-1] for term in loss_terms if term.history}
            loss_str = " ".join([f"{name}:{val:.2f}" for name, val in recent_losses.items()])
            iterator.set_postfix_str(f"total:{total_loss.item():.2f} {loss_str} lr:{scheduler.get_last_lr()[0]:.2e}")
    
    # Return all histories
    history = {"total": total_history}
    for term in loss_terms:
        history[term.name] = term.history
    
    return history