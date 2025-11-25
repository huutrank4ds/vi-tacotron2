import torch
import torchaudio
from config import Hparams
from torch.nn.utils.rnn import pad_sequence

class PrepareTextMel:
    """
    Một đối tượng callable duy nhất, vừa là processor vừa là hàm map.
    
    1. Khởi tạo với hparams để thiết lập các phép biến đổi (transforms) và 
       bộ ký tự (symbols).
    2. Có thể gọi được (callable) để xử lý các batch (chunks) từ 
       datasets.map, chuyển đổi text và audio.
    """
    def __init__(self, hparams: Hparams, speaker_embedding_dict: dict):
        """
        Khởi tạo processor với các siêu tham số.
        """
        self.hparams = hparams 
        self.speaker_embedding_dict = speaker_embedding_dict
        self._speaker_to_id = {s: i for i, s in self.speaker_embedding_dict['speaker_map'].items()}

        # --- Cấu hình Text ---
        self.symbols = hparams.symbols
        self._symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        self._id_to_symbol = {i: s for i, s in enumerate(self.symbols)}

        # --- Cấu hình Audio ---
        self.mel_transform = self.get_mel_transform(hparams)
        self.resampler_cache = {} 

    def get_mel_transform(self, hparams: Hparams):
        """Hàm helper để tạo đối tượng Mel transform."""
        return torchaudio.transforms.MelSpectrogram(
            sample_rate=hparams.target_sr,
            n_fft=hparams.n_fft,
            win_length=hparams.win_length,
            hop_length=hparams.hop_length,
            n_mels=hparams.n_mel_channels,
            f_min=hparams.f_min,
            f_max=hparams.f_max,
            power=1.0, 
            norm='slaney', 
            mel_scale='slaney'
        )

    # --- Các Method xử lý Text ---
    
    def text_to_sequence(self, text):
        """Chuyển đổi văn bản thành chuỗi ID"""
        sequence = []
        for char in text.lower():
            char_id = self._symbol_to_id.get(char) 
            if char_id is not None:
                sequence.append(char_id)
        return sequence

    def sequence_to_text(self, sequence):
        """Chuyển đổi chuỗi ID thành văn bản."""
        return "".join([self._id_to_symbol.get(i, '') for i in sequence])

    # --- Các Method xử lý Audio ---

    def resample_audio(self, audio_array, original_sr):
        """
        Resample audio từ original_sr về target_sr.
        """
        audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
        target_sr = self.hparams.target_sr
        if original_sr != target_sr:
            if original_sr not in self.resampler_cache:
                self.resampler_cache[original_sr] = torchaudio.transforms.Resample(
                    original_sr, target_sr
                )
            audio_tensor = self.resampler_cache[original_sr](audio_tensor)
        return audio_tensor

    def audio_to_mel(self, audio_array, original_sr):
        """
        Chuyển đổi waveform thành log-mel spectrogram.
        """
        # 1. Resample nếu cần
        audio_tensor = self.resample_audio(audio_array, original_sr)
        
        # 3. Tính Mel Spectrogram
        mel = self.mel_transform(audio_tensor.unsqueeze(0)) 
        
        # 4. Chuyển sang thang đo log
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        
        # 5. Bỏ batch dim và trả về
        return log_mel.squeeze(0) # Shape: [n_mels, n_frames]

    # --- Phương thức __call__ để dùng với .map() ---

    def __call__(self, batch):
        """
        Xử lý một batch (chunk) dữ liệu khi được gọi bởi datasets.map.
        
        Input: `batch` là một dict, ví dụ: 
               {
                'text': [str_1, ..., str_N], 
                'audio': {'path': [...], 'array': [...], 'sampling_rate': [...]},
                'speaker': [...]
                }
        Output: Một dict mới với các cột đã xử lý.
        """
        # Danh sách lưu trữ kết quả
        text_inputs_list = []
        text_lengths_list = []
        speaker_embeddings_list = []
        audio_tensors_list = []
        wav_lengths_list = []

        # Lặp qua từng mẫu trong batch (chunk)
        for i in range(len(batch['text'])):
            # --- 1. Xử lý Text ---
            text = batch['text'][i]
            text_seq = torch.IntTensor(self.text_to_sequence(text))
            text_len = text_seq.shape[0]  # Độ dài chuỗi text
            
            # --- 2. Xử lý Audio ---
            audio_data = batch['audio'][i]
            audio_array = audio_data['array']
            original_sr = audio_data['sampling_rate']
            
            speaker_name = batch['speaker'][i]
            speaker_id = self._speaker_to_id.get(speaker_name, None)
            if speaker_id is None:
                raise ValueError(f"Speaker '{speaker_name}' not found in speaker embedding dictionary.")
            speaker_embedding = self.speaker_embedding_dict['mean_embeddings'][speaker_id]

            # log_mel có shape: [n_mels, n_frames]
            audio_tensor = self.resample_audio(audio_array, original_sr)
            wav_len = audio_tensor.shape[0]

            # --- 4. Thêm vào danh sách ---
            text_inputs_list.append(text_seq)
            text_lengths_list.append(text_len)
            speaker_embeddings_list.append(speaker_embedding)
            audio_tensors_list.append(audio_tensor)
            wav_lengths_list.append(wav_len)
        
        # Trả về dict các danh sách
        return {
            'text_inputs': text_inputs_list,
            'text_lengths': text_lengths_list,
            'audio_tensors': audio_tensors_list,
            'wav_lengths': wav_lengths_list,
            'speaker_embeddings': speaker_embeddings_list,
        }
            
    # def __call__(self, batch):
    #     """
    #     Xử lý một batch (chunk) dữ liệu khi được gọi bởi datasets.map.
        
    #     Input: `batch` là một dict, ví dụ: 
    #            {
    #             'text': [str_1, ..., str_N], 
    #             'audio': {'path': [...], 'array': [...], 'sampling_rate': [...]},
    #             'speaker': [...]
    #             }
    #     Output: Một dict mới với các cột đã xử lý.
    #     """
    #     # Danh sách lưu trữ kết quả
    #     text_inputs_list = []
    #     text_lengths_list = []
    #     mel_targets_list = []
    #     mel_lengths_list = []
    #     speaker_embeddings_list = []
    #     stop_tokens_list = []

    #     # Lặp qua từng mẫu trong batch (chunk)
    #     for i in range(len(batch['text'])):
    #         # --- 1. Xử lý Text ---
    #         text = batch['text'][i]
    #         text_seq = torch.IntTensor(self.text_to_sequence(text))
    #         text_len = text_seq.shape[0]  # Độ dài chuỗi text
            
    #         # --- 2. Xử lý Audio ---
    #         audio_data = batch['audio'][i]
    #         audio_array = audio_data['array']
    #         original_sr = audio_data['sampling_rate']
            
    #         speaker_name = batch['speaker'][i]
    #         speaker_id = self._speaker_to_id.get(speaker_name, None)
    #         if speaker_id is None:
    #             raise ValueError(f"Speaker '{speaker_name}' not found in speaker embedding dictionary.")
    #         speaker_embedding = self.speaker_embedding_dict['mean_embeddings'][speaker_id]

    #         # log_mel có shape: [n_mels, n_frames]
    #         log_mel = self.audio_to_mel(audio_array, original_sr)

    #         # Reverse về shape [n_frames, n_mels]
    #         mel_target = log_mel.T 
    #         mel_len = mel_target.shape[0] # Số lượng frame (n_frames)

    #         # Bỏ qua các mẫu bị lỗi
    #         if mel_len == 0 or text_len == 0:
    #             continue 

    #         # --- 3. Tạo Stop Tokens ---
    #         stop_token = [0] * mel_len
    #         stop_token[-1] = 1
    #         stop_token = torch.FloatTensor(stop_token)

    #         # --- 4. Thêm vào danh sách ---
    #         text_inputs_list.append(text_seq)
    #         text_lengths_list.append(text_len)
    #         mel_targets_list.append(mel_target)
    #         mel_lengths_list.append(mel_len)
    #         speaker_embeddings_list.append(speaker_embedding)
    #         stop_tokens_list.append(stop_token)

    #     # Trả về dict các danh sách
    #     return {
    #         'text_inputs': text_inputs_list,
    #         'text_lengths': text_lengths_list,
    #         'mel_targets': mel_targets_list,
    #         'mel_lengths': mel_lengths_list,
    #         'speaker_embeddings': speaker_embeddings_list,
    #         'stop_tokens': stop_tokens_list
    #     }

class CollateTextMel:
    """
    Một đối tượng callable để đệm (pad) một batch dữ liệu 
    cho DataLoader của Tacotron 2.
    """
    def __init__(self, hparams: Hparams):
        self.text_pad_value = hparams.text_pad_value
        self.mel_pad_value = hparams.mel_pad_value
        self.stop_pad_value = hparams.stop_pad_value

    def __call__(self, batch):
        """
        Xử lý một batch (list các dict) khi được gọi.
        
        Input: `batch` là một list các dict, 
                mỗi dict có các keys: 
                'text_inputs', 'text_lengths', 
                'audio_tensors', 'speaker_embeddings'.
        """
        
        # --- 1. Đệm (Pad) Text Inputs ---
        all_text_inputs = [torch.as_tensor(item['text_inputs'], dtype=torch.long) for item in batch]
        text_padded = pad_sequence(
            all_text_inputs, 
            batch_first=True, 
            padding_value=self.text_pad_value
        )
        
        # --- 2. Đệm (Pad) Audio Tensors ---
        all_audio_tensors = [torch.as_tensor(item['audio_tensors'], dtype=torch.float) for item in batch]
        audio_padded = pad_sequence(
            all_audio_tensors, 
            batch_first=True, 
            padding_value=0.0
        )
        
        # --- 3. Lấy độ dài (Lengths) ---
        text_lengths = torch.tensor([item['text_lengths'] for item in batch], dtype=torch.long)
        wav_lengths = torch.tensor([item['wav_lengths'] for item in batch], dtype=torch.long)
        speaker_embeddings = torch.stack([torch.as_tensor(item['speaker_embeddings'], dtype=torch.float) for item in batch])
        
        return {
            'text_inputs': text_padded,      # [B, max_text_len]
            'text_lengths': text_lengths,    # [B]
            'audio_tensors': audio_padded,   # [B, max_audio_len]
            'wav_lengths': wav_lengths,      # [B]
            'speaker_embeddings': speaker_embeddings,  # [B, embedding_dim]
        }

    # def __call__(self, batch):
    #     """
    #     Xử lý một batch (list các dict) khi được gọi.
        
    #     Input: `batch` là một list các dict, 
    #             mỗi dict có các keys: 
    #             'text_inputs', 'text_lengths', 
    #             'mel_targets', 'mel_lengths', 'stop_tokens', 'speaker_embeddings'.
    #     """
        
    #     # --- 1. Đệm (Pad) Text Inputs ---
    #     all_text_inputs = [torch.as_tensor(item['text_inputs'], dtype=torch.long) for item in batch]
    #     text_padded = pad_sequence(
    #         all_text_inputs, 
    #         batch_first=True, 
    #         padding_value=self.text_pad_value
    #     )
        
    #     # --- 2. Đệm (Pad) Mel Targets ---
    #     all_mel_targets = [torch.as_tensor(item['mel_targets'], dtype=torch.float) for item in batch]
    #     mel_padded = pad_sequence(
    #         all_mel_targets, 
    #         batch_first=True, 
    #         padding_value=self.mel_pad_value
    #     )
        
    #     # Chuyển [B, max_mel_len, n_mels] -> [B, n_mels, max_mel_len]
    #     mel_padded = mel_padded.transpose(1, 2) 
        
    #     # --- 3. Đệm (Pad) Stop Tokens ---
    #     all_stop_tokens = [torch.as_tensor(item['stop_tokens'], dtype=torch.float) for item in batch]
    #     stop_padded = pad_sequence(
    #         all_stop_tokens, 
    #         batch_first=True, 
    #         padding_value=self.stop_pad_value
    #     )
        
    #     # --- 4. Lấy độ dài (Lengths) ---
    #     text_lengths = torch.tensor([item['text_lengths'] for item in batch], dtype=torch.long)
    #     mel_lengths = torch.tensor([item['mel_lengths'] for item in batch], dtype=torch.long)
    #     speaker_embeddings = torch.stack([torch.as_tensor(item['speaker_embeddings'], dtype=torch.float) for item in batch])
        
    #     return {
    #         'text_inputs': text_padded,      # [B, max_text_len]
    #         'text_lengths': text_lengths,    # [B]
    #         'mel_targets': mel_padded,       # [B, n_mels, max_mel_len]
    #         'mel_lengths': mel_lengths,      # [B],
    #         'speaker_embeddings': speaker_embeddings,  # [B, embedding_dim]
    #         'stop_tokens': stop_padded,      # [B, max_mel_len]
    #     }