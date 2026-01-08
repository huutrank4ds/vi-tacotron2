import os
import torch
import torchaudio
import torchaudio.transforms as T
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.inference.vocoders import HIFIGAN
from speechbrain.dataio.batch import PaddedBatch
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
        self.target_sample_rate = 16000

    def encode_file(self, file_path, target_device='cuda'):
        signal, fs = torchaudio.load(file_path)
        if fs != self.target_sample_rate:
            if fs not in self.resamplers:
                self.resamplers[fs] = T.Resample(orig_freq=fs, new_freq=self.target_sample_rate)
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
    
    def process_batch_audio(self, batch_data, device='cuda'):
        """Hàm xử lý danh sách audio raw thành list các tensors"""
        processed_tensors = []
        audios = batch_data['audio']
        for audio_info in audios:
            waveform = torch.tensor(audio_info['array'], dtype=torch.float32).to(device)
            original_sr = audio_info['sampling_rate']
            
            # Resample nếu cần
            if original_sr != self.target_sample_rate:
                if original_sr not in self.resamplers:
                    self.resamplers[original_sr] = T.Resample(orig_freq=original_sr, new_freq=self.target_sample_rate)
                waveform = self.resamplers[original_sr](waveform)
                
            # Đảm bảo mono (1 channel)
            if waveform.dim() > 1:
                waveform = waveform.mean(dim=0)
            processed_tensors.append({"signal": waveform})
        return processed_tensors
    
    def encoder_all_data(self, ds, batch_size=32, save_file=None):
        global_embeddings = []
        global_speaker_ids = []
        speaker_to_idx = {}
        idx_to_speaker = {} 
        next_speaker_id = 0

        for i in tqdm(range(0, len(ds), batch_size), desc="Encoding speaker embeddings for dataset"):
            batch_slice = ds[i:i+batch_size]
            batch_speakers = batch_slice['speaker']
            list_tensors = self.process_batch_audio(batch_slice, device=self.device)

            batch = PaddedBatch(list_tensors)
            signals = batch.signals.data
            lens = batch.signals.lengths
            with torch.no_grad():
                embeddings = self.classifier.encode_batch(signals, wave_lens=lens)
            embeddings = embeddings.squeeze(1).cpu().unbind(0)
            for idx, spk_name in enumerate(batch_speakers):
                # Mapping Speaker Name -> ID
                if spk_name not in speaker_to_idx:
                    speaker_to_idx[spk_name] = next_speaker_id
                    idx_to_speaker[next_speaker_id] = spk_name
                    next_speaker_id += 1
                
                # Lưu ID
                global_speaker_ids.append(speaker_to_idx[spk_name])
                # Lưu Embedding tương ứng
                global_embeddings.append(embeddings[idx])
        if save_file is not None:
            data_to_save = {
                'embeddings': global_embeddings,
                'speaker_ids': global_speaker_ids,
                'speaker_map': idx_to_speaker
            }
            torch.save(data_to_save, save_file)
            print(f"Saved speaker embeddings to {save_file}")
        return global_embeddings, global_speaker_ids, idx_to_speaker
    
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
