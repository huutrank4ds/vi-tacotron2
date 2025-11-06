from dataclasses import dataclass

@dataclass
class Hparams:
    # --- Tham số cho dataset ---
    dataset_name: str = "thivux/phoaudiobook"
    dataset_config: str = None
    seed: int = 42
    shuffle: bool = True
    shuffle_buffer_size: int = 10_000

    # --- Tham số cho DDP ---
    ddp_run: bool = True
    ddp_backend: str = 'nccl'  # Hoặc 'gloo' tùy vào môi trường của bạn
    ddp_url: str = 'tcp://localhost:12355'

    # --- Các tham số khác ---
    mask_padding: bool = True
    fp16_run: bool = False
    n_frames_per_step: int = 1 # Rất quan trọng cho Decoder
    
    
    # Text
    n_symbols: int = 67 # Phải khớp với len(symbols)
    symbols: str = ' abcdefghijklmnopqrstuvwxyzáàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ.,!?'
    symbols_embedding_dim: int = 512
    text_pad_value: int = 0
    
    # Audio
    target_sr: int = 22050
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
    encoder_n_convolutions: int = 3
    encoder_kernel_size: int = 5
    encoder_embedding_dim: int = 512

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
    epochs: int = 500
