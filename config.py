from dataclasses import dataclass, field
from typing import Dict, List, Optional

_pad = '_'
_punctuation = '!,.:;? '
_special = '-'
_letters = 'aàáảãạăằắẳẵặâầấẩẫậbcdđeèéẻẽẹêềếểễệghiìíỉĩịklmnoòóỏõọôồốổỗộơớờớởỡợpqrstuùúủũụưừứửữựvxyỳýỷỹỵ'
SYMBOLS_LIST = [_pad] + list(_special) + list(_punctuation) + list(_letters)
DATASET_CHUNKS_DEFAULT = [f'/kaggle/input/phoaudiobook-{i}' for i in range(1, 13)]

@dataclass
class Hparams:
    # --- Tham số cho dataset ---
    dataset_name: str = "thivux/phoaudiobook"
    dataset_config: str = 'default'
    hf_parquets_folder: str = 'data'
    parquet_valid_file: Optional[str] = None  # Đường dẫn tới file validation nếu có
    speaker_embeddings_file: Optional[str] = None  # Đường dẫn tới file embeddings nếu có
    validation_speaker_embeddings_file: Optional[str] = None  # Đường dẫn tới file embeddings validation nếu có
    num_train_samples: int = 1043443
    text_len_threshold: int = 5  # Giới hạn độ dài text tối thiểu
    duration_min_threshold: float = 0.5  # Giới hạn độ dài audio tối thiểu (giây)
    duration_max_threshold: float = 20.0  # Giới hạn độ dài audio tối đa (giây)


    # --- Tham số cho DDP ---
    ddp_run: bool = True
    ddp_backend: str = 'nccl'  # Hoặc 'gloo' tùy vào môi trường của bạn
    ddp_url: str = 'tcp://localhost:12355'

    # --- Các tham số khác ---
    mask_padding: bool = True
    fp16_run: bool = False
    n_frames_per_step: int = 2 # Rất quan trọng cho Decoder
    num_cpu : int = 4  # Số worker cho DataLoader
    
    # Text
    symbols: List[str] = field(default_factory=lambda: SYMBOLS_LIST)
    n_symbols: int = len(SYMBOLS_LIST)
    symbols_embedding_dim: int = 512
    text_pad_value: int = 0
    
    # Audio
    target_sr: int = 16000
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    f_min: float = 0.0
    f_max: float = 8000.0
    mel_pad_value: float = -11.5129  # log(1e-5)
    n_mel_channels: int = 80
    stop_pad_value: float = 1.0

    # --- Các tham số BẮT BUỘC cho Encoder/Decoder/Postnet ---
    # (Đây là các giá trị chuẩn của Tacotron 2)
        
    # --- Encoder ---
    speaker_embedding_dim: int = 192
    encoder_n_convolutions: int = 3
    encoder_kernel_size: int = 5
    encoder_embedding_dim: int = 512
    speaker_projection_dropout: float = 0.1

    # --- Prenet ---
    prenet_dim: int = 256

    # --- Attention ---
    attention_rnn_dim: int = 1024
    attention_dim: int = 128
    attention_location_n_filters: int = 32
    attention_location_kernel_size: int = 31

    # --- Decoder ---
    decoder_rnn_dim: int = 1024
    max_decoder_steps: int = 1000
    gate_threshold: float = 0.5
    p_attention_dropout: float = 0.1
    p_decoder_dropout: float = 0.1

    # --- Postnet ---
    postnet_n_convolutions: int = 5
    postnet_kernel_size: int = 5
    postnet_embedding_dim: int = 512

    # --- Cấu hình Training ---
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    batch_size: int = 32
    checkpoint_path: str = "checkpoints/"
    name_file_checkpoint: str = "checkpoint_step_1000.pt"
    val_interval: int = 100
    log_interval: int = 100
    seed: int = 42
    shuffle: bool = True
    shuffle_buffer_size: int = 10_000
    max_step_training: int = 5000
    early_stopping_patience: int = 48
    max_epochs: int = 50

    # --- Các tham số cho dữ liệu lớn ---
    dataset_chunks: List[str] = field(default_factory=lambda: DATASET_CHUNKS_DEFAULT)
    cache_chunk_dir: str = '/kaggle/working/chunk_cache'
    metadata: Dict[int, int] = field(default_factory=dict)  # Lưu trữ metadata cho từng chunk
    
