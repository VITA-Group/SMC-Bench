import math
import torch
from torch import Tensor
class EmpiricalBlockFisherInverse:
    def __init__(
        self,
        num_grads: int,
        fisher_block_size: int,
        num_weights: int,
        damp: float,
        device: torch.device,
    ):
        self.m = num_grads
        self.B = fisher_block_size
        self.d = num_weights
        self.damp = damp
        self.dev = device

        self.num_blocks = math.ceil(self.d / self.B)
        self.F_inv = (
            (1.0 / self.damp * torch.eye(n=self.B, device=self.dev))
            .unsqueeze(0)
            .repeat(self.num_blocks, 1, 1)
        )  # takes O(d x B) memory on a device

    def add_grad(self, g: Tensor):
        """
        Updates empirical Fisher inverse with a new gradient
        :param g: a collected gradient
        """
        # if 'd / B' is not integer, pad with zeros for batch calculations
        if g.numel() < self.num_blocks * self.B:
            g = torch.cat(
                [g, torch.zeros(self.num_blocks * self.B - g.numel(), device=g.device)]
            )

        # prepare grad for batch calculations
        g = g.view(self.num_blocks, self.B)

        # batched F_inv x g: (batch, B, B) x (batch, B) -> (batch, B)
        Finv_g = torch.einsum("bij,bj->bi", self.F_inv, g)

        # scalar denominator for each batch: (batch)
        alpha = (self.m + torch.einsum("bi,bi->b", g, Finv_g)).sqrt().unsqueeze(1)
        Finv_g /= alpha

        # update F_inv with new outer product: (batch, B) x (batch, B) -> (batch, B, B)
        self.F_inv.baddbmm_(Finv_g.unsqueeze(2), Finv_g.unsqueeze(1), alpha=-1)

    def diag(self) -> Tensor:
        """
        :return: diagonal of the Fisher inverse matrix
        """
        return self.F_inv.diagonal(dim1=1, dim2=2).flatten()[: self.d]

    def mul(self, v: Tensor) -> Tensor:
        """
        Computes matrix-vector product of the Fisher inverse matrix and a vector
        :param v: a vector to compute matrix-vector product with
        :return: result of the matrix-vector multiplication
        """
        if v.numel() < self.num_blocks * self.B:
            v = torch.cat(
                [v, torch.zeros(self.num_blocks * self.B - v.numel(), device=v.device)]
            )
        return torch.bmm(
            self.F_inv, v.view(self.num_blocks, self.B).unsqueeze_(2)
        ).flatten()[: self.d]