import numpy as np
import torch
import os

def load_checkpoint_chunk(checkpoint_path, model, device, optimizer = None):
    """Tải checkpoint từ file và khôi phục trạng thái model và optimizer."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    if isinstance(device, int):
        map_location = f'cuda:{device}'
    else:
        map_location = device
    checkpoint_dict = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint_dict['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint_dict:
        optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
    
    best_val_loss = checkpoint_dict.get('best_val_loss', None)
    epoch = checkpoint_dict.get('epoch', None)
    chunk_index = checkpoint_dict.get('chunk_index', None)
    patience_counter = checkpoint_dict.get('patience_counter', None)
    return best_val_loss, epoch, chunk_index, patience_counter
    

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
