import torch
from torch import nn
from encoder import Encoder
from decoder import Decoder
from postnet import Postnet
from utils import to_gpu, get_mask_from_lengths
import torch.nn.functional as F

class Tacotron2(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        
        self.embedding = nn.Embedding(
            hparams.n_symbols, hparams.symbols_embedding_dim)
        
        with torch.no_grad():
            torch.nn.init.xavier_uniform_(self.embedding.weight)
            
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)

        self.speaker_projection = nn.Sequential(
            nn.Linear(hparams.speaker_embedding_dim, 256),
            nn.ReLU(),
            # nn.Dropout(0.1), # Dropout thường tắt khi inference, nhưng training cần
            nn.Linear(256, hparams.encoder_embedding_dim) 
        )
        self.init_speaker_projection_weights()

    def init_speaker_projection_weights(self):
        for module in self.speaker_projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def parse_batch(self, batch, rank):
        # Hàm này hiện tại có thể không dùng nếu bạn dùng parse_batch_gpu trong worker
        # Nhưng giữ lại cũng không sao
        text_padded = batch['text_inputs']
        input_lengths = batch['text_lengths']
        mel_padded = batch['mel_targets']
        gate_padded = batch['stop_tokens']
        speaker_embeddings = batch['speaker_embeddings']
        output_lengths = batch['mel_lengths']

        text_padded = to_gpu(text_padded, rank).long()
        input_lengths = to_gpu(input_lengths, rank).long()
        mel_padded = to_gpu(mel_padded, rank).float()
        gate_padded = to_gpu(gate_padded, rank).float()
        output_lengths = to_gpu(output_lengths, rank).long()
        speaker_embeddings = to_gpu(speaker_embeddings, rank).float()

        return (
            (text_padded, input_lengths, mel_padded, output_lengths, speaker_embeddings),
            (mel_padded, gate_padded))

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            # [FIX QUAN TRỌNG] Lấy max_len từ chính tensor output (đã pad r)
            max_len = outputs[0].size(2)
            device = outputs[0].device
            
            # Tạo mask [B, T_out]
            ids = torch.arange(0, max_len, device=device, dtype=torch.long)
            # True là dữ liệu Valid, False là Padding
            mask_valid = (ids < output_lengths.unsqueeze(1)) 
            
            # Đảo ngược mask: True là Padding (để fill)
            mask = ~mask_valid

            # Expand mask cho Mel [B, n_mels, T]
            mask_mel = mask.unsqueeze(1).expand_as(outputs[0])
            
            # [FIX] Bỏ .data để chuẩn PyTorch mới
            outputs[0].masked_fill_(mask_mel, 0.0)
            outputs[1].masked_fill_(mask_mel, 0.0)
            outputs[2].masked_fill_(mask, 1e3) # Gate

        return outputs
    
    def forward(self, inputs):
        # Giải nén input từ parse_batch_gpu
        text_inputs, text_lengths, mels, output_lengths, speaker_embeddings = inputs
        text_lengths, output_lengths = text_lengths.detach(), output_lengths.detach()

        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)

        encoder_outputs = self.encoder(embedded_inputs, text_lengths)
        
        # Xử lý Speaker
        speaker_projection = self.speaker_projection(speaker_embeddings)
        speaker_projection = F.normalize(speaker_projection) # Normalize L2
        
        # Cộng (Broadcast)
        encoder_outputs = encoder_outputs + speaker_projection.unsqueeze(1)

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, memory_lengths=text_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            output_lengths)

    def inference(self, text_inputs, speaker_embeddings):
        # text_inputs: [1, T]
        # speaker_embeddings: [1, Dim]

        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)
        
        # Xử lý Speaker
        speaker_projection = self.speaker_projection(speaker_embeddings)
        
        # [FIX QUAN TRỌNG] Phải Normalize giống hệt Forward
        speaker_projection = F.normalize(speaker_projection) 

        encoder_outputs = self.encoder.inference(embedded_inputs)
        encoder_outputs = encoder_outputs + speaker_projection.unsqueeze(1)

        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        # Inference không có output_lengths nên không mask (hoặc mask=None)
        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments]
        )

        return outputs