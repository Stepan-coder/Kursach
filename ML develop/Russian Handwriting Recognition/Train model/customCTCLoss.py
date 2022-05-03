import math
import torch


class CustomCTCLoss(torch.nn.Module):
    def __init__(self, dim=2):
        super().__init__()
        self.dim = dim
        self.ctc_loss = torch.nn.CTCLoss(reduction='mean', zero_infinity=True)

    def forward(self, logits, labels,
                prediction_sizes, target_sizes):
        loss = self.sanitize(self.ctc_loss(logits, labels, prediction_sizes, target_sizes))
        return self.debug(loss, logits, labels, prediction_sizes, target_sizes)

    def sanitize(self, loss):
        if abs(loss.item()) > 99999:
            return torch.zeros_like(loss, requires_grad=True)
        if math.isnan(loss.item()):
            return torch.zeros_like(loss, requires_grad=True)
        return loss

    def debug(self, loss, logits, labels, prediction_sizes, target_sizes):
        if math.isnan(loss.item()):
            print(f"Loss: {loss}, "
                  f"logits: {logits}, "
                  f"labels: {labels}, "
                  f"prediction_sizes: {prediction_sizes}, "
                  f"target_sizes: {target_sizes}")
            raise Exception("NaN loss obtained.")
        return loss


