from model import Tacotron2
from module import SpeakerEncoder, Vocoder
from processing import PrepareTextMel
import os
from utils import load_checkpoint_chunk
import torch

class WrappedModel:
    def __init__(self, tacotron2_hparams, device='cuda'):
        self.device = device
        self.tacotron2 = Tacotron2(tacotron2_hparams).to(device)
        self.tacotron2.eval()
        
        self.speaker_encoder = SpeakerEncoder(device=device)
        self.vocoder = Vocoder(device=device)
        self.prepare = PrepareTextMel(hparams=tacotron2_hparams)
        checkpoint_path = os.path.join(tacotron2_hparams.checkpoint_path, tacotron2_hparams.name_file_checkpoint)
        checkpoint_param = load_checkpoint_chunk(checkpoint_path, self.tacotron2, device)

    def gen_wav(self, text, speaker_embedding=None, audio=None):
        # Chuẩn bị input text
        text_tensor = self.prepare.processing_text(text, self.device)  # [1, T] #type: ignore
        
        # Lấy speaker embedding
        if speaker_embedding is not None:
            spk_emb_tensor = speaker_embedding.unsqueeze(0).to(self.device)  # [1, Dim]
        elif audio is not None:
            with torch.no_grad():
                spk_emb_tensor = self.speaker_encoder.encode_signal(
                    audio.unsqueeze(0), fs=16000, target_device=self.device).unsqueeze(0)  # [1, Dim]
        else:
            raise ValueError("Either speaker_embedding or audio must be provided.")
        
        # Sinh mel spectrogram từ Tacotron2
        with torch.no_grad():
            mel_outputs, mel_outputs_postnet, _, _ = self.tacotron2.inference(
                text_tensor, spk_emb_tensor)
        
        # Chuyển mel spectrogram thành waveform với Vocoder
        with torch.no_grad():
            wav = self.vocoder.synthesize(mel_outputs_postnet.unsqueeze(0))  # [1, T]
        
        return wav.squeeze(0).cpu().numpy()