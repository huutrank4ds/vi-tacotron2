from config import Hparams
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
    state_end_epoch = checkpoint_dict.get('end_epoch', None)
    return best_val_loss, epoch, chunk_index, patience_counter, state_end_epoch

def save_checkpoint_chunk(model, optimizer, best_val_loss, patience_counter, epoch, chunk_index, end_epoch, filepath, hparams: Hparams):
    model_state_dict = model.state_dict()
    checkpoint_dict = {
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'patience_counter': patience_counter,
        'epoch': epoch,
        'chunk_index': chunk_index,
        'end_epoch': end_epoch
    }
    if not os.path.exists(hparams.checkpoint_path):
        os.makedirs(hparams.checkpoint_path)
    torch.save(checkpoint_dict, os.path.join(hparams.checkpoint_path, filepath))
    print(f"Saved checkpoint at {os.path.join(hparams.checkpoint_path, filepath)}")
    

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

def parse_batch_gpu(batch, device, mel_transform, hparams):
    # 1. Load dữ liệu
    text_padded = to_gpu(batch['text_inputs'], device).long()
    input_lengths = to_gpu(batch['text_lengths'], device).long()
    speaker_embeddings = to_gpu(batch['speaker_embeddings'], device).float()
    wav_lengths = to_gpu(batch['wav_lengths'], device).long()
    audio_list = to_gpu(batch['audio_tensors'], device)
    
    # 2. Tính Mel Spectrograms
    mels = mel_transform(audio_list) # audio_list đã pad 0.0
    mels = torch.log(torch.clamp(mels, min=1e-5))
    
    # Tính độ dài thực tế (Dựa trên wav lengths gốc)
    r = hparams.n_frames_per_step
    remainder = mels.shape[2] % r
    if remainder != 0:
        pad_size = r - remainder
        # Pad chiều cuối (Time)
        mels = torch.nn.functional.pad(mels, (0, pad_size), value=hparams.mel_pad_value)

    # Tính độ dài thực tế
    mel_lengths = 1 + (wav_lengths // hparams.hop_length)
    
    # 3. Tạo Gate Target (Stop Token)
    max_mel_len = mels.shape[2]
    gate_padded = torch.zeros(mels.shape[0], max_mel_len, device=device)

    # Đánh dấu 1 ở vị trí kết thúc
    for i, l in enumerate(mel_lengths):
        end_idx = min(l - 1, max_mel_len - 1)
        gate_padded[i, end_idx:] = 1.0 

    return (
        (text_padded, input_lengths, mels, mel_lengths, speaker_embeddings),
        (mels, gate_padded)
    )