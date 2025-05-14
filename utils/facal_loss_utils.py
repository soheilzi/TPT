import torch
import torch.nn.functional as F
from torch import nn
from transformers import Trainer

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)          # (B, C)
        probs      = log_probs.exp()                       # pt  in the paper
        tgt_log_p  = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt         = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        loss = -self.alpha * (1.0 - pt).pow(self.gamma) * tgt_log_p
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss            # 'none'

def focal_loss(logits, labels, alpha=1.0, gamma=2):
    # Move labels to the correct device to enable model parallelism
    labels = labels.to(logits.device)
    # Shift so that tokens < n predict n
    # Logits shape: (Batch, T, Vocab_size)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Compute log probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    # Gather the log probabilities of the correct labels
    log_p_t = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    
    # Compute probabilities of the correct labels
    p_t = log_p_t.exp()

    # Compute the modulating factor
    modulating_factor = alpha * (1 - p_t) ** gamma
    
    # Stop gradient for modulating factor
    # modulating_factor = modulating_factor.detach() # Look into this to check if it is correct.
    
    # Compute the focal loss per token
    loss = -modulating_factor * log_p_t
    # Take the mean loss over all tokens and batches
    loss = loss.mean()
    return loss

class FocalTrainer(Trainer):
    """
    Drop‑in replacement for `Trainer` that computes focal loss instead of the
    loss returned by the model.
    """
    def __init__(self, *args, alpha=1.0, gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")          # (B, S,)
        outputs = model(**inputs)              # forward pass
        logits  = outputs.logits               # (B, S, C)
        import pdb; pdb.set_trace()
        loss = focal_loss(logits, labels)      # focal loss

        return (loss, outputs) if return_outputs else loss


def clipped_loss(logits, labels, alpha=1.0, gamma=0.8):
    # Move labels to the correct device to enable model parallelism
    labels = labels.to(logits.device)
    # Shift so that tokens < n predict n
    # Logits shape: (Batch, T, Vocab_size)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Compute log probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)
    # Gather the log probabilities of the correct labels
    log_p_t = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1) 
    # Compute probabilities of the correct labels
    p_t = log_p_t.exp()
    # Compute the modulating factor
    p_t_clipped = torch.where(p_t > gamma, torch.tensor(0.0, dtype=p_t.dtype, device=p_t.device), torch.tensor(1.0, dtype=p_t.dtype, device=p_t.device))
    modulating_factor = alpha * p_t_clipped
    
    # Stop gradient for modulating factor
    modulating_factor = modulating_factor.detach() # Look into this to check if it is correct.

    # Compute the focal loss per token
    loss = -modulating_factor * log_p_t    
    # Take the mean loss over all tokens and batches
    loss = loss.mean()
    return loss

class ClipLoss(nn.Module):
    def __init__(self, clip_val: float = 0.9, reduction: str = "mean"):
        super().__init__()
        self.clip_val = clip_val
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)          # (B, C)
        probs      = log_probs.exp()                       # pt  in the paper
        tgt_log_p  = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt         = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        loss = -tgt_log_p * torch.where(pt > self.clip_val, 0.0, 1.0) 
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss            # 'none'

class ClipTrainer(Trainer):
    """
    Drop‑in replacement for `Trainer` that computes focal loss instead of the
    loss returned by the model.
    """
    def __init__(self, *args, clip_val=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip_loss = ClipLoss(clip_val=clip_val)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")          # (B,)
        outputs = model(**inputs)              # forward pass
        logits  = outputs.logits               # (B, C)

        loss = clipped_loss(logits, labels)      # focal loss

        return (loss, outputs) if return_outputs else loss