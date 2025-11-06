import torch
from torch.utils.data import DataLoader
# Import thêm Features và các kiểu dữ liệu từ datasets
from datasets import load_dataset, Features, Value, Sequence, Array2D
from processing import PrepareTextMel, CollateTextMel
from config import Hparams

# === HÀM TẠO DATALOADER ===
def get_trainloader_valset(rank, world_size, hparams: Hparams):
    """
    Tải và chuẩn bị DataLoader cho training (streaming, sharded)
    và Validation set (chỉ rank 0, non-streaming).
    """
    
    # --- 0. Khởi tạo processors ---
    # Sử dụng đối tượng hparams được truyền vào hàm
    prepare_text_mel = PrepareTextMel(hparams)
    collate_fn = CollateTextMel() # Giả định CollateTextMel không cần hparams

    DATASET_NAME = hparams.dataset_name
    DATASET_CONFIG = hparams.dataset_config

    # --- 1. XỬ LÝ TRAINING (Tất cả worker) ---
    iterable_train_ds = load_dataset(
        DATASET_NAME, 
        DATASET_CONFIG, 
        split='train', 
        streaming=True,
        trust_remote_code=True # An toàn hơn nên thêm
    )
    
    # Xóa dòng .cast_column("audio", None) - chúng ta sẽ dùng remove_columns

    if hparams.shuffle:
        shuffled_ds = iterable_train_ds.shuffle(seed=hparams.seed, buffer_size=hparams.shuffle_buffer_size)
        sharded_ds = shuffled_ds.shard(num_shards=world_size, index=rank)
    else:
        sharded_ds = iterable_train_ds.shard(num_shards=world_size, index=rank)

    # === SỬA LỖI: Định nghĩa schema đầu ra rõ ràng ===
    new_features = Features({
        'text_inputs': Sequence(Value('int64')),
        'text_lengths': Value('int64'),
        # Giả sử hparams.n_mels là 80
        'mel_targets': Array2D(shape=(None, hparams.n_mels), dtype='float32'),
        'mel_lengths': Value('int64'),
        'stop_tokens': Sequence(Value('float32'))
    })

    processed_ds = sharded_ds.map(
        prepare_text_mel,
        batched=True,
        batch_size=1000,
        remove_columns=sharded_ds.column_names, # <-- SỬA: Xóa cột cũ
        features=new_features                  # <-- SỬA: Cung cấp schema mới
    )
    
    trainloader = DataLoader(
        processed_ds,
        batch_size=hparams.batch_size,
        num_workers=0, # Chính xác, phải là 0 cho streaming
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # --- 2. XỬ LÝ VALIDATION (Chỉ Rank 0) ---
    valset = None
    if rank == 0:
        print("[Rank 0] Đang tải và xử lý validation set...")
        valset = load_dataset(
            DATASET_NAME, 
            DATASET_CONFIG, 
            split='validation', 
            streaming=False, # Tải 1 lần
            trust_remote_code=True # An toàn hơn nên thêm
        )

        valset = valset.map(
            prepare_text_mel,
            batched=True,
            batch_size=1000,
            # Có thể dùng nhiều worker vì không streaming
            num_proc=hparams.num_workers, 
            remove_columns=valset.column_names, # <-- SỬA: Xóa cột cũ
            features=new_features                 # <-- SỬA: Cung cấp schema mới
        )
        print(f"[Rank 0] Đã tạo tập Validation thành công.")

    print(f"[Rank {rank}] Đã tạo DataLoader thành công.")
    return trainloader, valset