from abc import ABC, abstractmethod
from typing import Optional
import torch


class AbstractTransform(ABC):

    @abstractmethod
    def transform(
            self,
            points: torch.Tensor,
            planar_symmetries: Optional[torch.Tensor] = None,
            axis_continue_symmetries: Optional[torch.Tensor] = None,
            axis_discrete_symmetries: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        pass

    @abstractmethod
    def inverse_transform(
            self,
            points: torch.Tensor,
            planar_symmetries: Optional[torch.Tensor] = None,
            axis_continue_symmetries: Optional[torch.Tensor] = None,
            axis_discrete_symmetries: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        pass

    def __call__(
            self,
            points: torch.Tensor,
            planar_symmetries: Optional[torch.Tensor] = None,
            axis_continue_symmetries: Optional[torch.Tensor] = None,
            axis_discrete_symmetries: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.transform(points, planar_symmetries, axis_continue_symmetries, axis_discrete_symmetries)
