import torch
import torch.nn as nn

from aurora.losses.loss_term import LossTerm
from aurora.datasets.ray_data import RayDataset
from aurora.models.flux_model import FluxModel
from aurora.sampling import sample_random_in_bins


class RayLoss(LossTerm):
    def __init__(
            self,
            flux_model: FluxModel,
            ray_data: RayDataset, 
            batch_size: int = 4096,
            weight: float = 1.0, 
            num_ray_bins: int = 64,
            name: str = "ray_loss"
        ):
        super().__init__(weight, batch_size, name)
        
        self.flux_model = flux_model
        self.ray_data = ray_data
        self.num_ray_bins = num_ray_bins
        self.loss_fn = nn.MSELoss()
    
    def sample_batch(self):
        return self.ray_data.random_sample(self.batch_size)
    
    def eval_raw_loss(self, batch_in: RayDataset.SampleType):
        ro, rd, tn, tf, g_target = batch_in
        t = sample_random_in_bins(tn, tf, self.num_ray_bins)
        g_pred = self.flux_model.int_emis_ray(ro, rd, t)
        return self.loss_fn(g_pred, g_target)