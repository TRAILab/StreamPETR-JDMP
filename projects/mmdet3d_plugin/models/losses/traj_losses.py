import torch
import mmcv
import torch.nn as nn
from mmdet.models import LOSSES
from mmdet.models.losses.utils import weighted_loss

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def gaussian_nll_loss(pred, target, eps=1e-6):
    """ Computes negative log likelihood of ground truth trajectory under a
    predictive distribution with a single mode,
    with a bivariate Gaussian distribution predicted at each time in the
    prediction horizon.

    Args:
        pred (torch.Tensor): Predictions. [batch_size, num_modes, time_steps, 5]
        target (torch.Tensor): Ground truth trajectory. [batch_size, num_modes, time_steps, 2]

    Returns:
        torch.Tensor: Calculated loss [batch_size, num_modes, time_steps]
    """
    assert pred.size(-1) == 5 or pred.size(-1) == 4
    if pred.size(-1) == 5:
        mu_x = pred[..., 0]
        mu_y = pred[..., 1]
        x = target[..., 0]
        y = target[..., 1]
        sig_x = pred[..., 2]
        sig_y = pred[..., 3]
        rho = pred[..., 4]
        ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
        loss = 0.5 * torch.pow(ohr, 2) * \
            (torch.pow(sig_x, 2) * torch.pow(x - mu_x, 2) + torch.pow(sig_y, 2) *
            torch.pow(y - mu_y, 2) - 2 * rho * torch.pow(sig_x, 1) *
            torch.pow(sig_y, 1) * (x - mu_x) * (y - mu_y)) - \
            torch.log(sig_x * sig_y * ohr) + 1.8379
        loss[loss.isnan()] = 0
        loss[loss.isinf()] = 0
    else:
        input = pred[..., :2]
        var = pred[..., 2:]
        assert var.size() == input.size()
        assert torch.all(var >= 0)
        var = var.clone()
        with torch.no_grad():
            var.clamp_(min=eps)
        loss = 0.5 * (torch.log(var) + (input - target) ** 2 / var)
        loss = loss.sum(dim=-1)
    return loss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def laplace_nll_loss(pred, target, eps=1e-6):
    """
    Computes the negative log-likelihood loss assuming a Laplace distribution.

    Args:
        pred: Tensor of shape [batch_size, num_modes, time_steps, 4], where:
              - pred[..., :2] are the predicted means (y_hat)
              - pred[..., 2:] are the predicted scales (b)
        target: Tensor of shape [batch_size, num_modes, time_steps, 2], representing the true values.
        eps: Small constant to prevent division by zero and log instability.

    Returns:
        torch.Tensor: Calculated loss [batch_size, num_modes, time_steps]
    """
    assert pred.size(-1) == 4
    mean = pred[..., :2]
    with torch.no_grad():
        scale = pred[..., 2:].clone().clamp(min=eps)
    loss = torch.log(2 * scale) + torch.abs(target - mean) / scale
    loss = torch.sum(loss, dim=-1)
    return loss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def l2_loss(pred, target):
    """ Computes average displacement error for the best trajectory is a set,
    with respect to ground truth.

    Args:
        pred (torch.Tensor): Predictions. [batch_size, num_modes, time_steps, 2]
        target (torch.Tensor): Ground truth trajectory. [batch_size, num_modes, time_steps, 2]

    Returns:
        torch.Tensor: Calculated loss [batch_size, num_modes, time_steps]
    """
    assert pred.size(-1) == 2
    loss = torch.norm(pred[..., :2] - target[..., :2], p=2, dim=-1)
    return loss

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def l1_loss(pred, target, beta=1.0):
    """ Computes average displacement error for the best trajectory is a set,
    with respect to ground truth.

    Args:
        pred (torch.Tensor): Predictions. [batch_size, num_modes, time_steps, 2]
        target (torch.Tensor): Ground truth trajectory. [batch_size, num_modes, time_steps, 2]

    Returns:
        torch.Tensor: Calculated loss [batch_size, num_modes, time_steps]
    """
    assert pred.size(-1) == 2
    loss = torch.abs(pred[..., :2] - target[..., :2]).sum(dim=-1)
    # diff = torch.abs(pred[..., :2] - target[..., :2]) # smooth l1 loss
    # loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
    #                    diff - 0.5 * beta).sum(dim=-1)
    return loss

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def min_fde_loss(pred, target):
    """ Computes final displacement error for the best trajectory is a set,
    with respect to ground truth.

    Args:
        pred (torch.Tensor): Predictions. [batch_size, num_modes, time_steps, 2]
        target (torch.Tensor): Ground truth trajectory. [batch_size, num_modes, time_steps, 2]

    Returns:
        torch.Tensor: Calculated loss [batch_size, num_modes, time_steps]
    """
    assert pred.size(-1) >= 2
    loss = torch.norm(pred[..., -1, :2] - target[..., -1, :2], p=2, dim=-1) # B, M
    loss = torch.min(loss, dim=-1)[0] # B
    loss = loss.unsqueeze(-1).unsqueeze(-1).expand(pred.shape[0], pred.shape[1], pred.shape[2]) # B, M, T
    return loss


@LOSSES.register_module()
class TrajLoss(nn.Module):
    def __init__(self, losses=['l2'], reduction='mean', loss_weights=[0.5]):
        super(TrajLoss, self).__init__()
        self.losses = losses
        self.reduction = reduction
        self.loss_weights = loss_weights
        self.loss_functions = { # all are minADE based losses except min_fde
            'gaussian_nll': gaussian_nll_loss,
            'laplace_nll': laplace_nll_loss,
            'l2': l2_loss,
            'l1': l1_loss,
            'min_fde': min_fde_loss
        }
        assert len(self.losses) == len(self.loss_weights)
        assert all(loss in self.loss_functions for loss in self.losses)

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        assert pred.shape[:-1] == target.shape[:-1], "All dimensions except last should be equal"
        assert target.size(-1) == 2

        reduction = (
            reduction_override if reduction_override else self.reduction)
        weight = weight[..., 0] if weight is not None else None
        total_loss = 0.0
        for loss, loss_weight in zip(self.losses, self.loss_weights):
            loss_value = self.loss_functions[loss](pred, target, weight, reduction, avg_factor)
            if loss_value is not None:
                total_loss += loss_value * loss_weight
            
        return total_loss
