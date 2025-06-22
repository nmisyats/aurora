import torch
import torch.nn as nn

from aurora.losses.loss_term import LossTerm
from aurora.datasets.radar_data import RadarDataset
from aurora.models.flux_model import FluxModel


class RadarLoss(LossTerm):
    def __init__(
            self,
            flux_model: FluxModel,
            radar_data: RadarDataset,
            batch_size: int = 1024,
            weight: float = 1.0,
            name: str = "radar_loss"
        ):
        super().__init__(weight, batch_size, name)

        self.flux_model = flux_model
        self.radar_data = radar_data
        self.loss_fn = nn.MSELoss()
    
    def sample_batch(self):
        return self.radar_data.random_sample(self.batch_size)
    
    def eval_raw_loss(self, batch_in: RadarDataset.SampleType):
        p, d_target = batch_in
        d_pred = self.flux_model.elec_dens(p)
        return self.loss_fn(d_pred, d_target)
