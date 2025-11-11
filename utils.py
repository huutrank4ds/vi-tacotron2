import numpy as np
import torch


def get_mask_from_lengths(lengths):
    """Tạo mask từ lengths - an toàn với cả CPU và GPU"""
    max_len = torch.max(lengths).item()
    device = lengths.device
    ids = torch.arange(0, max_len, dtype=torch.long, device=device)
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def to_gpu(x, rank):
    x = x.contiguous()
    if torch.cuda.is_available():
        # Truyền rank (số nguyên) hoặc device object vào hàm .to()
        x = x.to(rank, non_blocking=True)
    return x
