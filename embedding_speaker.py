import os
import torch
import torchaudio
import torchaudio.transforms as T
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.inference.vocoder import HIFIGAN
import IPython.display as ipd
import numpy as np
from tqdm import tqdm

class SpeakerEncoder:
    def __init__(self, device='cuda'):
        self.device = device
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": self.device}
        )
        self.classifier.eval()
        self.resamplers = {}

    def encode_file(self, file_path, target_device='cuda'):
        signal, fs = torchaudio.load(file_path)
        if fs != 16000:
            if fs not in self.resamplers:
                self.resamplers[fs] = T.Resample(orig_freq=fs, new_freq=16000)
            signal = self.resamplers[fs](signal)
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        with torch.no_grad():
            embedding = self.classifier.encode_batch(signal.to(self.device))
        return embedding.squeeze(0).to(target_device)

    def encode_directory(self, dir_path, endswith=['.wav', '.flac', '.mp3']):
        embeddings = {}
        for filename in tqdm(os.listdir(dir_path), desc="Encoding speaker embeddings"):
            if filename.endswith(tuple(endswith)):
                file_path = os.path.join(dir_path, filename)
                embedding = self.encode_file(file_path)
                embeddings[filename] = embedding
        return embeddings
    
    def encode_signal(self, signal, fs, target_device='cuda'):
        if fs != 16000:
            if fs not in self.resamplers:
                self.resamplers[fs] = T.Resample(orig_freq=fs, new_freq=16000)
            signal = self.resamplers[fs](signal)
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        with torch.no_grad():
            embedding = self.classifier.encode_batch(signal.to(self.device))
        return embedding.squeeze(0).to(target_device)
    
class Vocoder:
    def __init__(self, device='cuda'):
        self.device = device
        self.vocoder = HIFIGAN.from_hparams(
            source="speechbrain/tts-hifigan-libritts-16kHz",
            savedir="pretrained_models/tts-hifigan-libritts-16kHz",
            run_opts={"device": self.device}
        )
        self.vocoder.eval()

    def synthesize(self, mel_spectrogram):
        with torch.no_grad():
            waveform = self.vocoder.decode_batch(mel_spectrogram.to(self.device))
        return waveform.squeeze(1).cpu()
    
    def synthesize_to_file(self, mel_spectrogram, output_path):
        waveform = self.synthesize(mel_spectrogram)
        torchaudio.save(output_path, waveform, 16000)

    def _play_audio(self, waveform, ):
        waveform = waveform.detach().cpu().squeeze(0).numpy()
        if np.abs(waveform).max() > 1.0:
            waveform = waveform / np.abs(waveform).max()
        return ipd.Audio(waveform, rate=16000, normalize=False)
