from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossMseCfg:
    weight: float


@dataclass
class LossMseCfgWrapper:
    mse: LossMseCfg


class LossMse(Loss[LossMseCfg, LossMseCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        valid_mask: Tensor | None = None,
    ) -> Float[Tensor, ""]:
        delta = prediction.color - batch["target"]["image"]
        if valid_mask is not None:
            mask = valid_mask.to(delta.device)
            mask = mask.to(delta.dtype)
            denom = mask.sum().clamp(min=1.0)
            return self.cfg.weight * ((mask * (delta**2)).sum() / denom)
        return self.cfg.weight * (delta**2).mean()

    def dynamic_forward(
        self, prediction: DecoderOutput, gt_image: Tensor, global_step: int, weight: float | None = None
    ) -> Float[Tensor, ""]:
        weight = self.cfg.weight if weight is None else weight
        delta = prediction.color - gt_image
        return weight * (delta**2).mean()
