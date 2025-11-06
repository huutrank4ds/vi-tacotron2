import torch
import torch.nn as nn

class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()
        # Dùng MSELoss (reduction='none' để tính từng phần tử)
        self.mel_loss_fn = nn.MSELoss(reduction='none') 
        # Dùng BCEWithLogitsLoss cho gate (đầu ra là logits)
        self.gate_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def get_mask_from_lengths(self, lengths):
        """Tạo mask từ lengths - an toàn với cả CPU và GPU"""
        max_len = torch.max(lengths).item()
        device = lengths.device
        ids = torch.arange(0, max_len, dtype=torch.long, device=device)
        mask = (ids < lengths.unsqueeze(1)).bool()
        return mask

    def forward(self, model_outputs, ground_truth, output_lengths):
        """
        model_outputs: List từ forward [mel_out, mel_post_out, gate_out, align]
        ground_truth: Tuple từ parse_batch (mel_padded, gate_padded)
        """
        
        # --- Giải nén đầu vào ---
        mel_out = model_outputs[0]              # [B, n_mels, T]
        mel_out_postnet = model_outputs[1]      # [B, n_mels, T]
        gate_out = model_outputs[2]             # [B, T]
        
        mel_target = ground_truth[0]            # [B, n_mels, T]
        gate_target = ground_truth[1]           # [B, T]
        
        # --- Tạo mask ---
        # mask shape: [B, T]
        mask = self.get_mask_from_lengths(output_lengths)
        
        # Mở rộng mask cho mel: [B, T] -> [B, n_mels, T]
        mel_mask = mask.unsqueeze(1).expand_as(mel_out)
        
        # --- Tính toán 3 thành phần loss ---
        
        # 1. Mel loss (trước postnet)
        mel_loss_raw = self.mel_loss_fn(mel_out, mel_target)
        loss_mel = torch.sum(mel_loss_raw * mel_mask) / torch.sum(mel_mask)
        
        # 2. Mel loss (sau postnet)
        mel_postnet_loss_raw = self.mel_loss_fn(mel_out_postnet, mel_target)
        loss_mel_postnet = torch.sum(mel_postnet_loss_raw * mel_mask) / torch.sum(mel_mask)

        # 3. Gate loss
        gate_loss_raw = self.gate_loss_fn(gate_out, gate_target)
        loss_gate = torch.sum(gate_loss_raw * mask) / torch.sum(mask)
        
        # Tổng loss
        total_loss = loss_mel + loss_mel_postnet + loss_gate
        
        return total_loss, loss_mel, loss_mel_postnet, loss_gate