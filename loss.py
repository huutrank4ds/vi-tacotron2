import torch
import torch.nn as nn

class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()
        # Dùng MSELoss (hoặc L1Loss) cho spectrogram
        self.mel_loss = nn.MSELoss() 
        # Dùng BCEWithLogitsLoss cho gate (đầu ra là logits)
        self.gate_loss = nn.BCEWithLogitsLoss()

    def forward(self, model_outputs, ground_truth, output_lengths):
        """
        model_outputs: List từ forward [mel_out, mel_post_out, gate_out, align]
        ground_truth: Tuple từ parse_batch (mel_padded, gate_padded)
        """
        
        # --- Giải nén đầu vào ---
        mel_out = model_outputs[0]
        mel_out_postnet = model_outputs[1]
        gate_out = model_outputs[2]
        
        mel_target = ground_truth[0]  # mel_padded
        gate_target = ground_truth[1] # gate_padded
        
        # --- Tạo mask ---
        mel_mask = self.get_mask_from_lengths(output_lengths)
        gate_mask = self.get_mask_from_lengths(output_lengths)
        # --- Tính toán 3 thành phần loss ---
        
        # 1. Mel loss (trước postnet)
        loss_mel = self.mel_loss(mel_out * mel_mask, mel_target * mel_mask)
        
        # 2. Mel loss (sau postnet)
        loss_mel_postnet = self.mel_loss(mel_out_postnet * mel_mask, mel_target * mel_mask)

        # 3. Gate loss
        # Cần reshape gate_out (B, T_mel) -> (B*T_mel)
        # và gate_target (B, T_mel) -> (B*T_mel)
        loss_gate = self.gate_loss(
            gate_out.view(-1) * gate_mask.view(-1), 
            gate_target.view(-1) * gate_mask.view(-1)
        )
        
        # Tổng loss
        total_loss = loss_mel + loss_mel_postnet + loss_gate
        
        return total_loss, loss_mel, loss_mel_postnet, loss_gate