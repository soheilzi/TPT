import torch
import torch.nn.functional as F
from torch import nn
from transformers import Trainer

IGNORE_INDEX = -100



def focal_loss(outputs, labels, alpha=1.0, ignore_index=IGNORE_INDEX, gamma=2, num_items_in_batch=1):
    # Move labels to the correct device to enable model parallelism
    logits = outputs.logits
    labels = labels.to(logits.device)
    # Shift so that tokens < n predict n
    # Logits shape: (Batch, T, Vocab_size)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Compute log probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    # Gather the log probabilities of the correct labels
    mask = shift_labels.ne(ignore_index)

    safe_labels = shift_labels.masked_fill(~mask, 0)

    log_probs = F.log_softmax(shift_logits, dim=-1)
    log_p_t   = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)

    # Compute probabilities of the correct labels
    p_t = log_p_t.exp()

    # Compute the modulating factor
    modulating_factor = alpha * (1 - p_t) ** gamma
    
    # Stop gradient for modulating factor
    # modulating_factor = modulating_factor.detach() # Look into this to check if it is correct.
    
    # Compute the focal loss per token
    loss = -modulating_factor * log_p_t
    # Take the mean loss over all tokens and batches
    loss = loss.masked_select(mask).sum() / torch.sum(mask.float())
    return loss



def clipped_loss(outputs, labels, alpha=1.0, gamma=0.9, ignore_index=IGNORE_INDEX, num_items_in_batch=1):
    # Move labels to the correct device to enable model parallelism
    logits = outputs.logits
    labels = labels.to(logits.device)
    # Shift so that tokens < n predict n
    # Logits shape: (Batch, T, Vocab_size)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    mask = shift_labels.ne(ignore_index)

    safe_labels = shift_labels.masked_fill(~mask, 0)

    log_probs = F.log_softmax(shift_logits, dim=-1)
    log_p_t   = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    # Compute probabilities of the correct labels
    p_t = log_p_t.exp()
    # Compute the modulating factor
    p_t_clipped = torch.where(p_t > gamma, torch.tensor(0.0, dtype=p_t.dtype, device=p_t.device), torch.tensor(1.0, dtype=p_t.dtype, device=p_t.device))
    modulating_factor = alpha * p_t_clipped
    
    # Stop gradient for modulating factor
    modulating_factor = modulating_factor.detach() # Look into this to check if it is correct.

    # import pdb; pdb.set_trace()
    # Compute the focal loss per token
    loss = -modulating_factor * log_p_t    
    # loss = -log_p_t
    # Take the mean loss over all tokens and batches with mask
    loss = loss.masked_select(mask).sum() / torch.sum(mask.float())
    # loss = torch.sum(loss * mask.float(), dim=-1) / torch.sum(mask.float(), dim=-1)
    # loss = loss.mean()
    return loss
