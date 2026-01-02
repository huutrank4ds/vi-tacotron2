import torch
import torch.nn as nn
from utils import get_mask_from_lengths
import math

class Tacotron2Loss(nn.Module):
    def __init__(self, n_frame_per_step, pos_weight=10.0, guided_sigma=0.2):
        super(Tacotron2Loss, self).__init__()
        self.r = n_frame_per_step
        # Dùng MSELoss (reduction='none' để tính từng phần tử)
        self.mel_loss_fn = nn.MSELoss(reduction='none') 
        # Dùng BCEWithLogitsLoss cho gate (đầu ra là logits)
        weight = torch.tensor([pos_weight])
        self.gate_loss_fn = nn.BCEWithLogitsLoss(reduction='none', pos_weight=weight)
        self.get_mask_from_lengths = get_mask_from_lengths
        self.sigma = guided_sigma

    def _compute_guided_attention_loss(self, alignments, input_lengths, output_lengths):
        B, T_out, T_in = alignments.shape
        device = alignments.device
        # Tạo lưới tọa độ cho Encoder và Decoder
        soft_mask = torch.zeros((B, T_out, T_in), device=device)
        for b in range(B):
            N = input_lengths[b].item()
            T_raw = output_lengths[b].item()
            T_target_steps = math.ceil(T_raw / self.r)
            T = min(T_target_steps, T_out)
            if N == 0 or T == 0: continue
            n_idx = torch.arange(N, device=device, dtype=torch.float32)
            t_idx = torch.arange(T, device=device, dtype=torch.float32)
            # Tạo lưới 2D [T, N]
            grid_n = n_idx.unsqueeze(0) / N  # [1, N]
            grid_t = t_idx.unsqueeze(1) / T  # [T, 1]
            # Tính ma trận hướng dẫn
            W = 1.0 - torch.exp(-((grid_t - grid_n) ** 2) / (2 * (self.sigma ** 2)))  # [T, N]
            soft_mask[b, :T, :N] = W
        guided_loss = torch.mean(alignments * soft_mask)
        return guided_loss


    def forward(self, model_outputs, ground_truth, output_lengths, input_lengths=None):
        mel_out, mel_out_postnet, gate_out, _ = model_outputs
        mel_target, gate_target = ground_truth

        # --- 1. Xử lý Mask (FIXED) ---
        # Lấy max_len từ Target (đã được pad chẵn r trong workers.py)
        max_len = mel_target.size(2) 
        device = mel_target.device
        
        # Tạo mask khớp size [B, T]
        ids = torch.arange(0, max_len, device=device, dtype=torch.long)
        mask = (ids < output_lengths.unsqueeze(1)) # True = Valid data
        
        # Mở rộng mask cho Mel [B, n_mels, T]
        # Expand mask theo mel_out (vì mel_out và mel_target cùng size)
        mel_mask = mask.unsqueeze(1).expand_as(mel_out)

        # --- 2. Tính Mel Loss ---
        mel_loss = self.mel_loss_fn(mel_out, mel_target)
        loss_mel = (mel_loss * mel_mask).sum() / mel_mask.sum()

        mel_postnet_loss = self.mel_loss_fn(mel_out_postnet, mel_target)
        loss_mel_postnet = (mel_postnet_loss * mel_mask).sum() / mel_mask.sum()

        # --- 3. Tính Gate Loss ---
        if gate_out.dim() > 2:
            gate_out = gate_out.squeeze(-1)
            
        gate_loss = self.gate_loss_fn(gate_out, gate_target)
        loss_gate = (gate_loss * mask).sum() / mask.sum()

        # --- 4. Tính Guided Attention Loss (nếu có input_lengths) ---
        guided_attn_loss = 0.0
        if input_lengths is not None:
            alignments = model_outputs[3]  # [B, T_out, T_in]
            guided_attn_loss = self._compute_guided_attention_loss(alignments, input_lengths, output_lengths)

        # --- 4. Tổng hợp ---
        total_loss = loss_mel + loss_mel_postnet + loss_gate + guided_attn_loss

        return total_loss, loss_mel, loss_mel_postnet, loss_gate