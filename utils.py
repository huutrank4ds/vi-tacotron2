import numpy as np
import torch
import os

def load_checkpoint(checkpoint_path, model, device, optimizer=None, strict=True):
    """
    Tải checkpoint từ đường dẫn và khôi phục trạng thái cho model và optimizer (nếu có).
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint_dict = torch.load(checkpoint_path, map_location=device)
    # Khôi phục trọng số mô hình
    model.load_state_dict(checkpoint_dict['model_state_dict'], strict=strict)
    # Khôi phục optimizer (nếu có)
    if optimizer is not None and 'optimizer_state_dict' in checkpoint_dict:
        optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
    # Trả về checkpoint để có thể dùng thêm các thông tin khác (epoch, loss, v.v.)
    return checkpoint_dict


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
