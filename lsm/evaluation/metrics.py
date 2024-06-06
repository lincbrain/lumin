import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanBCELoss(nn.Module):
    """
    mean binary cross entropy loss
    """

    def __init__(self):
        super(MeanBCELoss, self).__init__()

    def forward(self, predictions, targets):
        mask = targets >= 0
        # apply mask to tensors
        predictions_masked = predictions[mask]
        targets_masked = targets[mask]
        # return mean loss
        return F.binary_cross_entropy(predictions_masked, targets_masked).mean()


class DiceLoss(nn.Module):
    """
    dice loss
    """

    def __init__(self, reduce=True, smooth=100.0, power=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduce = reduce
        self.power = power

    def dice_loss(self, pred, target):
        loss = 0.0

        for index in range(pred.size()[0]):
            iflat = pred[index].contiguous().view(-1)
            tflat = target[index].contiguous().view(-1)
            intersection = (iflat * tflat).sum()
            if self.power == 1:
                loss += 1 - (
                    (2.0 * intersection + self.smooth)
                    / (iflat.sum() + tflat.sum() + self.smooth)
                )
            else:
                loss += 1 - (
                    (2.0 * intersection + self.smooth)
                    / (
                        (iflat**self.power).sum()
                        + (tflat**self.power).sum()
                        + self.smooth
                    )
                )

        return loss / float(pred.size()[0])

    def dice_loss_batch(self, pred, target):
        iflat = pred.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()

        if self.power == 1:
            loss = 1 - (
                (2.0 * intersection + self.smooth)
                / (iflat.sum() + tflat.sum() + self.smooth)
            )
        else:
            loss = 1 - (
                (2.0 * intersection + self.smooth)
                / (
                    (iflat**self.power).sum()
                    + (tflat**self.power).sum()
                    + self.smooth
                )
            )
        return loss

    def forward(self, pred, target, weight_mask=None):
        if not (target.size() == pred.size()):
            raise ValueError(
                "Target size ({}) must be the same as pred size ({})".format(
                    target.size(), pred.size()
                )
            )

        if self.reduce:
            loss = self.dice_loss(pred, target)
        else:
            loss = self.dice_loss_batch(pred, target)
        return loss


class KLDivergence(nn.Module):
    """

    KL-divergence of 2 distributions
    """

    def __init__(self):
        super(KLDivergence, self).__init__()

    def forward(self, predictions, targets):
        mask = targets >= 0
        # clamp values to fix log error
        target = torch.clamp(targets[mask], min=torch.finfo(torch.float32).eps, max=1.0)
        predictions = torch.clamp(
            targets[mask], min=torch.finfo(torch.float32).eps, max=1.0
        )

        bce_target = F.binary_cross_entropy(targets, targets, reduction="none")
        bce_pred = F.binary_cross_entropy(predictions, targets, reduction="none")

        return torch.mean(bce_pred - bce_target)


class MeanAverageError(nn.Module):
    """
    weighted mean average error
    """

    def __init__(self):
        super(MeanAverageError, self).__init__()

    def forward(self, predictions, targets, weight=None):
        loss = F.l1_loss(predictions, targets, reduction="none")
        loss = loss * weight
        return loss


class MeanSquareError(nn.Module):
    """
    weighted mean squared error
    """

    def __init__(self):
        super(MeanSquareError, self).__init__()

    def mse(self, predictions, target, weight=None):
        s1 = torch.prod(torch.tensor(predictions.size()[2:]).float())
        s2 = predictions.size()[0]
        norm_term = (s1 * s2).to(predictions.device)
        if weight is None:
            return torch.sum((predictions - target) ** 2) / norm_term

        return torch.sum(weight * (pred - target) ** 2) / norm_term

    def forward(self, predictions, targets, weight=None):
        return self.mse(predictions=predictions, target=targets, weight=weight)


if __name__ == "__main__":
    print(f"hello")
