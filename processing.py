import torch
import torchaudio
from config import Hparams
from torch.nn.utils.rnn import pad_sequence

class AudioTextProcessor:
    def __init__(self, hparams: Hparams):
        self.hparams = hparams 

        self.symbols = hparams.symbols
        self._symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        self._id_to_symbol = {i: s for i, s in enumerate(self.symbols)}

        self.mel_transform = self.get_mel_transform(hparams)
        self.resampler_cache = {} 

    def get_mel_transform(self, hparams: Hparams):
        """Hàm helper để tạo đối tượng Mel transform."""
        return torchaudio.transforms.MelSpectrogram(
            sample_rate=hparams.target_sr,
            n_fft=hparams.n_fft,
            win_length=hparams.win_length,
            hop_length=hparams.hop_length,
            n_mels=hparams.n_mels,
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

    def audio_to_mel(self, audio_array, original_sr):
        """
        Chuyển đổi waveform thành log-mel spectrogram.
        """
        
        # 1. Chuyển sang tensor
        audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
        
        # 2. Resample nếu cần
        target_sr = self.hparams.target_sr 
        if original_sr != target_sr:
            # Chỉ tạo resampler một lần cho mỗi original_sr
            if original_sr not in self.resampler_cache:
                self.resampler_cache[original_sr] = torchaudio.transforms.Resample(
                    original_sr, target_sr
                )
            audio_tensor = self.resampler_cache[original_sr](audio_tensor)
        
        # 3. Tính Mel Spectrogram
        mel = self.mel_transform(audio_tensor.unsqueeze(0)) 
        
        # 4. Chuyển sang thang đo log
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        
        # 5. Bỏ batch dim và trả về
        return log_mel.squeeze(0)

class PrepareTextMel:
    """
    Một đối tượng callable để xử lý batch (chunk) từ datasets.map.
    Chuyển đổi text thành chuỗi ID và audio thành mel-spectrogram.
    """
    def __init__(self, processor: AudioTextProcessor):
        self.processor = processor

    def __call__(self, batch):
        """
        Xử lý một batch (chunk) dữ liệu khi được gọi.
        
        Input: `batch` là một dict, ví dụ: 
               {'text': [str_1, ..., str_N], 
                'audio': {'path': [...], 'array': [...], 'sampling_rate': [...]}}
        Output: Một dict mới với các cột đã xử lý.
        """
        # Danh sách lưu trữ kết quả
        text_inputs_list = []
        text_lengths_list = []
        mel_targets_list = []
        mel_lengths_list = []
        stop_tokens_list = []

        # Lặp qua từng mẫu trong batch (chunk)
        for i in range(len(batch['text'])):
            # --- 1. Xử lý Text ---
            text = batch['text'][i]
            # Sử dụng processor từ self
            text_seq = self.processor.text_to_sequence(text) 
            
            # --- 2. Xử lý Audio ---
            audio_data = batch['audio'][i]
            audio_array = audio_data['array']
            original_sr = audio_data['sampling_rate']
            
            # Sử dụng processor từ self
            # log_mel có shape: [n_mels, n_frames]
            log_mel = self.processor.audio_to_mel(audio_array, original_sr)

            # Reverse về shape [n_frames, n_mels]
            mel_target = log_mel.T 
            
            mel_len = mel_target.shape[0] # Số lượng frame (n_frames)
            text_len = len(text_seq) # Độ dài chuỗi text

            # Bỏ qua các mẫu bị lỗi
            if mel_len == 0 or text_len == 0:
                continue 

            # --- 3. Tạo Stop Tokens ---
            stop_token = [0] * mel_len
            stop_token[-1] = 1

            # --- 4. Thêm vào danh sách ---
            text_inputs_list.append(text_seq)
            text_lengths_list.append(text_len)
            mel_targets_list.append(mel_target)
            mel_lengths_list.append(mel_len)
            stop_tokens_list.append(stop_token)

        # Trả về dict các danh sách
        return {
            'text_inputs': text_inputs_list,
            'text_lengths': text_lengths_list,
            'mel_targets': mel_targets_list,
            'mel_lengths': mel_lengths_list,
            'stop_tokens': stop_tokens_list
        }
    
import torch
from torch.nn.utils.rnn import pad_sequence

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
               ví dụ: [{'text_inputs': [1,2,3], 'mel_targets': tensor_A, ...}, 
                       {'text_inputs': [4,5], 'mel_targets': tensor_B, ...}]
        """
        
        # --- Chuyển đổi dữ liệu đầu vào ---
        # Chuyển đổi text và stop_tokens (dạng list) sang tensor
        # mel_targets đã là tensor từ lớp PrepareTextMel
        for i in range(len(batch)):
            batch[i]['text_inputs'] = torch.tensor(
                batch[i]['text_inputs'], dtype=torch.long
            )
            batch[i]['stop_tokens'] = torch.tensor(
                batch[i]['stop_tokens'], dtype=torch.float32
            )
        
        # --- 1. Đệm (Pad) Text Inputs ---
        all_text_inputs = [item['text_inputs'] for item in batch]
        text_padded = pad_sequence(
            all_text_inputs, 
            batch_first=True, 
            padding_value=self.text_pad_value
        )
        
        # --- 2. Đệm (Pad) Mel Targets ---
        all_mel_targets = [item['mel_targets'] for item in batch]
        mel_padded = pad_sequence(
            all_mel_targets, 
            batch_first=True, 
            padding_value=self.mel_pad_value
        )
        
        # Chuyển [B, max_mel_len, n_mels] -> [B, n_mels, max_mel_len]
        mel_padded = mel_padded.transpose(1, 2) 
        
        # --- 3. Đệm (Pad) Stop Tokens ---
        all_stop_tokens = [item['stop_tokens'] for item in batch]
        stop_padded = pad_sequence(
            all_stop_tokens, 
            batch_first=True, 
            padding_value=self.stop_pad_value
        )
        
        # --- 4. Lấy độ dài (Lengths) ---
        text_lengths = torch.tensor([item['text_lengths'] for item in batch], dtype=torch.long)
        mel_lengths = torch.tensor([item['mel_lengths'] for item in batch], dtype=torch.long)
        
        return {
            'text_inputs': text_padded,      # [B, max_text_len]
            'text_lengths': text_lengths,    # [B]
            'mel_targets': mel_padded,       # [B, n_mels, max_mel_len]
            'mel_lengths': mel_lengths,      # [B]
            'stop_tokens': stop_padded       # [B, max_mel_len]
        }

# def preprocess_function(batch):
#     """
#     Xử lý một batch (chunk) dữ liệu từ `datasets.map`.
#     Input: `batch` là một dict, ví dụ: {'text': [str_1, ..., str_N], 'audio': {'path': [path_1, ..., path_N], 'array': [arr_1, ..., arr_N], 'sampling_rate': [sr_1, ..., sr_N]}}
#     Output: Một dict mới với các cột đã xử lý.
#     """
#     # Danh sách lưu trữ kết quả
#     text_inputs_list = []
#     text_lengths_list = []
#     mel_targets_list = []
#     mel_lengths_list = []
#     stop_tokens_list = []

#     # Lặp qua từng mẫu trong batch (chunk)
#     for i in range(len(batch['text'])):
#         # --- 1. Xử lý Text ---
#         text = batch['text'][i]
#         text_seq = processor.text_to_sequence(text)
        
#         # --- 2. Xử lý Audio ---
#         audio_data = batch['audio'][i]
#         audio_array = audio_data['array']
#         original_sr = audio_data['sampling_rate']
        
#         # log_mel có shape: [n_mels, n_frames] (Đã loại bỏ batch dim)
#         log_mel = processor.audio_to_mel(audio_array, original_sr)

#         # Reverse về shape [n_frames, n_mels]
#         # Tacotron 2 thường làm việc với shape: [n_frames, n_mels]
#         mel_target = log_mel.T 
        
#         mel_len = mel_target.shape[0] # Số lượng frame (n_frames)
#         text_len = len(text_seq) # Độ dài chuỗi text

#         # Bỏ qua các mẫu bị lỗi (ví dụ: audio quá ngắn, text rỗng)
#         if mel_len == 0 or text_len == 0:
#             continue 

#         # --- 3. Tạo Stop Tokens ---
#         # Stop token là một chuỗi 0, và số 1 ở frame cuối cùng
#         stop_token = [0] * mel_len
#         stop_token[-1] = 1

#         # --- 4. Thêm vào danh sách ---
#         text_inputs_list.append(text_seq)
#         text_lengths_list.append(text_len)
#         mel_targets_list.append(mel_target)
#         mel_lengths_list.append(mel_len)
#         stop_tokens_list.append(stop_token)

#     # Trả về dict các danh sách
#     return {
#         'text_inputs': text_inputs_list,
#         'text_lengths': text_lengths_list,
#         'mel_targets': mel_targets_list,
#         'mel_lengths': mel_lengths_list,
#         'stop_tokens': stop_tokens_list
#     }

    
# def tacotron2_collate(batch):
#     """
#     Hàm collate để đệm (pad) các batch dữ liệu cho Tacotron 2.
#     Input: `batch` là một list các dict, 
#            ví dụ: [{'text_inputs': [1,2,3], 'mel_targets': tensor_A, ...}, 
#                    {'text_inputs': [4,5], 'mel_targets': tensor_B, ...}]
#     """
    
#     # --- Chuyển đổi dữ liệu đầu vào ---
#     # Chuyển đổi text và stop_tokens (dạng list) sang tensor
#     # mel_targets đã là tensor từ preprocess_function
#     for i in range(len(batch)):
#         batch[i]['text_inputs'] = torch.tensor(
#             batch[i]['text_inputs'], dtype=torch.long
#         )
#         batch[i]['stop_tokens'] = torch.tensor(
#             batch[i]['stop_tokens'], dtype=torch.float32
#         )
#         # Không cần chuyển mel_targets vì nó đã là tensor
    
#     # --- 1. Đệm (Pad) Text Inputs ---
#     all_text_inputs = [item['text_inputs'] for item in batch]
#     text_padded = pad_sequence(all_text_inputs, batch_first=True, padding_value=0)
    
#     # --- 2. Đệm (Pad) Mel Targets ---
#     all_mel_targets = [item['mel_targets'] for item in batch]
#     mel_pad_value = torch.log(torch.tensor(1e-5)) 
#     mel_padded = pad_sequence(all_mel_targets, batch_first=True, padding_value=mel_pad_value)
    
#     # Chuyển [B, max_mel_len, n_mels] -> [B, n_mels, max_mel_len]
#     mel_padded = mel_padded.transpose(1, 2) 
    
#     # --- 3. Đệm (Pad) Stop Tokens ---
#     all_stop_tokens = [item['stop_tokens'] for item in batch]
#     stop_padded = pad_sequence(all_stop_tokens, batch_first=True, padding_value=1.0)
    
#     # --- 4. Lấy độ dài (Lengths) ---
#     text_lengths = torch.tensor([item['text_lengths'] for item in batch], dtype=torch.long)
#     mel_lengths = torch.tensor([item['mel_lengths'] for item in batch], dtype=torch.long)
    
#     return {
#         'text_inputs': text_padded,       # [B, max_text_len]
#         'text_lengths': text_lengths,     # [B]
#         'mel_targets': mel_padded,        # [B, n_mels, max_mel_len]
#         'mel_lengths': mel_lengths,       # [B]
#         'stop_tokens': stop_padded        # [B, max_mel_len]
#     }