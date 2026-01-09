import torch
import torchaudio
from config import Hparams
from torch.nn.utils.rnn import pad_sequence
import re
import librosa #type: ignore

class PrepareTextMel:
    """
    Một đối tượng callable duy nhất, vừa là processor vừa là hàm map.
    
    1. Khởi tạo với hparams để thiết lập các phép biến đổi (transforms) và 
       bộ ký tự (symbols).
    2. Có thể gọi được (callable) để xử lý các batch (chunks) từ 
       datasets.map, chuyển đổi text và audio.
    """
    def __init__(self, hparams: Hparams, speaker_embedding_dict: dict = None): # type: ignore
        """
        Khởi tạo processor với các siêu tham số.
        """
        self.hparams = hparams 
        if speaker_embedding_dict is not None:
            self.speaker_embedding_dict = speaker_embedding_dict
            self._speaker_to_id = {s: i for i, s in self.speaker_embedding_dict['speaker_map'].items()}
        else:
            self.speaker_embedding_dict = None
            self._speaker_to_id = {}

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
    def clean_text(self, text):
        if not text: return ""
        text = text.lower()
        # Token tạm để bảo vệ những chữ "chấm" cần giữ lại
        TEMP_TOKEN = " _SPECIAL_DOT_ "
        # --- BƯỚC 0: BẢO VỆ SỐ THẬP PHÂN (5 chấm 5 -> giữ nguyên) ---
        text = re.sub(r'(\d+)\s+chấm\s+(\d+)', rf'\1{TEMP_TOKEN}\2', text)
        # --- BƯỚC 1: BẢO VỆ TỪ GHÉP PHÍA TRƯỚC (Whitelist Before) ---
        # Những từ này đứng trước "chấm" thì "chấm" là động từ/danh từ -> GIỮ NGUYÊN
        whitelist_before = [
            "dấu", "nước", "đồ", "bún", "rau", "muối", 
            "bát", "chén", "đĩa", "tự", "người", "ban", 
            "ba", "vừa", "đang"
        ]
        for word in whitelist_before:
            text = re.sub(rf'\b{word}\s+chấm\b', f'{word}{TEMP_TOKEN}', text)
        # --- BƯỚC 2: BẢO VỆ TỪ GHÉP PHÍA SAU (Whitelist After) ---
        # [QUAN TRỌNG] Thêm "bài", "thi", "hết" vào đây để GIỮ NGUYÊN
        common_words = "dứt|điểm|thi|công|bài|phá|mút|bi|tử|lượng|phạt|đầu|nương|hết|hỏi|than|lửng|phẩy"
        domains = "com|net|vn|org|edu|gov|io|info|biz"
        # Kết hợp tất cả từ cần bảo vệ
        whitelist_after = f"{common_words}|{domains}"
        # Logic Regex:
        # Tìm chữ "chấm" (và khoảng trắng trước nó \s*)
        # NHƯNG chỉ thay thế nếu phía sau nó KHÔNG PHẢI (?!) là các từ trong whitelist
        pattern = r'\s*\bchấm\b(?!\s*(' + whitelist_after + '))'
        # Thay thế các trường hợp còn lại thành dấu "."
        text = re.sub(pattern, '.', text)
        # --- BƯỚC 3: KHÔI PHỤC ---
        # Trả lại chữ "chấm" cho các trường hợp đã bảo vệ
        text = text.replace(TEMP_TOKEN, " chấm ")
        # --- BƯỚC 4: DỌN DẸP DẤU CÂU ---
        text = re.sub(r'\s+\.', '.', text)   # Xóa space thừa trước dấu chấm
        text = re.sub(r'\.\s*', '. ', text)  # Thêm space sau dấu chấm
        text = re.sub(r'\.\s*\.', '.', text) # Fix lỗi 2 dấu chấm
        text = re.sub(r'\s+', ' ', text)     # Fix nhiều space
        return text.strip()

    def text_to_sequence(self, text):
        """Chuyển đổi văn bản thành chuỗi ID"""
        sequence = []
        text = self.clean_text(text)
        for char in text:
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

    # Phương thức xử lý mẫu đơn lẻ
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

    def prosessing_text(self, text, device):
        return torch.IntTensor(self.text_to_sequence(text)).unsqueeze(0).to(device)

    # --- Phương thức __call__ để dùng với .map() ---
    def __call__(self, batch):
        """
        Xử lý một batch (chunk) dữ liệu khi được gọi bởi datasets.map.
        """
        # Danh sách lưu trữ kết quả
        text_inputs_list = []
        text_lengths_list = []
        speaker_embeddings_list = []
        audio_tensors_list = []
        wav_lengths_list = []

        # Lặp qua từng mẫu trong batch
        for i in range(len(batch['text'])):
            # --- 1. Xử lý Text ---
            text = batch['text'][i]
            # text_to_sequence nên đã bao gồm chuẩn hóa Unicode NFC
            text_seq = torch.IntTensor(self.text_to_sequence(text))
            text_len = text_seq.shape[0]
            
            # --- 2. Xử lý Audio ---
            audio_data = batch['audio'][i]
            audio_array = audio_data['array']
            original_sr = audio_data['sampling_rate']
            audio_array, _ = librosa.effects.trim(audio_array, top_db=20)
            
            # Resample về target_sr
            audio_tensor = self.resample_audio(audio_array, original_sr)
            
            # [QUAN TRỌNG] Đảm bảo audio là 1D [Time]
            if audio_tensor.dim() > 1:
                audio_tensor = audio_tensor.squeeze() 
            
            wav_len = audio_tensor.shape[0]

            # --- 3. Bộ lọc dữ liệu (Data Filtering) ---
            # Tính độ dài giây
            duration_sec = wav_len / self.hparams.target_sr
            
            # Điều kiện lọc:
            # - Text rỗng hoặc Audio rỗng
            # - Audio quá ngắn (< 0.5s): Thường là lỗi cắt hoặc khoảng lặng
            # - Audio quá dài (> 20s): Gây tràn bộ nhớ GPU (OOM) với batch size lớn
            # - Audio quá nhanh (CPS > 22) hoặc quá chậm (CPS < 8)
            if text_len < self.hparams.text_len_threshold or wav_len == 0:
                continue
            if duration_sec < self.hparams.duration_min_threshold or duration_sec > self.hparams.duration_max_threshold:
                continue
            cps = text_len / duration_sec
            if cps < self.hparams.cps_min_threshold or cps > self.hparams.cps_max_threshold:
                # print(f"Skipped abnormal CPS: {cps:.2f} (Text len: {text_len}, Duration: {duration_sec:.2f}s)")
                continue

            # --- 4. Xử lý Speaker Embedding ---
            speaker_name = batch['speaker'][i]
            
            # Logic lấy embedding an toàn (Tránh crash)
            speaker_embedding = None
            if self.speaker_embedding_dict is not None:
                speaker_id = self._speaker_to_id.get(speaker_name)
                if speaker_id is not None:
                    try:
                        speaker_embedding = self.speaker_embedding_dict['mean_embeddings'][speaker_id]
                    except (KeyError, IndexError):
                        pass
            
            # Fallback: Nếu không tìm thấy hoặc lỗi, dùng vector 0
            if speaker_embedding is None:
                print(f"Warning: Speaker '{speaker_name}' not found. Remove this sample!")
                # speaker_embedding = torch.zeros(self.hparams.speaker_embedding_dim, dtype=torch.float32)
                continue


            # --- 5. Thêm vào danh sách ---
            text_inputs_list.append(text_seq)
            text_lengths_list.append(text_len)
            speaker_embeddings_list.append(speaker_embedding)
            audio_tensors_list.append(audio_tensor)
            wav_lengths_list.append(wav_len)
        
        # Trả về dict các danh sách (Key khớp với CollateTextMel)
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
        
        # Gom Text
        text_inputs = [torch.as_tensor(x['text_inputs'], dtype=torch.long) for x in batch]
        text_padded = pad_sequence(text_inputs, batch_first=True, padding_value=self.text_pad_value)
        text_lengths = torch.tensor([x['text_lengths'] for x in batch], dtype=torch.long)
        
        # Gom Audio (Quan trọng)
        audio_tensors = [torch.as_tensor(x['audio_tensors'], dtype=torch.float) for x in batch]
        
        # Pad Audio với giá trị 0 (Silence)
        # Output: [Batch, Max_Time]
        audio_padded = pad_sequence(audio_tensors, batch_first=True, padding_value=0.0)
        wav_lengths = torch.tensor([x['wav_lengths'] for x in batch], dtype=torch.long)
        
        # Gom Speaker
        speaker_embeddings = torch.stack([torch.as_tensor(x['speaker_embeddings'], dtype=torch.float) for x in batch])
        
        return {
            'text_inputs': text_padded,      # [B, max_text_len]
            'text_lengths': text_lengths,    # [B]
            'audio_tensors': audio_padded,   # [B, max_audio_len]
            'wav_lengths': wav_lengths,      # [B]
            'speaker_embeddings': speaker_embeddings, # [B, Dim]
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