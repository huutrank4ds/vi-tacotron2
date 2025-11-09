import torch
import os
from torch.utils.data import DataLoader
# Import thêm Features và các kiểu dữ liệu từ datasets
from datasets import load_dataset, load_from_disk
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
    collate_fn = CollateTextMel(hparams)

    DATASET_NAME = hparams.dataset_name
    DATASET_CONFIG = hparams.dataset_config

    # --- 1. XỬ LÝ TRAINING (Tất cả worker) ---
    iterable_train_ds = load_dataset(
        DATASET_NAME, 
        DATASET_CONFIG, 
        split='train', 
        streaming=True
    )

    if hparams.shuffle:
        shuffled_ds = iterable_train_ds.shuffle(seed=hparams.seed, buffer_size=hparams.shuffle_buffer_size)
        sharded_ds = shuffled_ds.shard(num_shards=world_size, index=rank)
    else:
        sharded_ds = iterable_train_ds.shard(num_shards=world_size, index=rank)

    processed_ds = sharded_ds.map(
        prepare_text_mel,
        batched=True,
        batch_size=1000
    )
    
    trainloader = DataLoader(
        processed_ds,
        batch_size=hparams.batch_size,
        num_workers=0, # Chính xác, phải là 0 cho streaming
        pin_memory=False,    
        collate_fn=collate_fn
    )
    
    # --- 2. XỬ LÝ VALIDATION (Chỉ Rank 0) ---
    valset = None
    if rank == 0:
        processed_dataset_path = "./my_processed_phoaudiobook"
        if not os.path.exists(processed_dataset_path):
            print("[Rank 0] Đang tải và xử lý validation set...")
            valset = load_dataset(
                DATASET_NAME, 
                DATASET_CONFIG, 
                split='validation[:5%]', 
                streaming=False
            )

            valset = valset.map(
                prepare_text_mel,
                batched=True,
                batch_size=1000
            )

            valset.save_to_disk(processed_dataset_path)
            print(f"[Rank 0] Đã lưu tập validation đã xử lý tại {processed_dataset_path}.")
        else:
            print("[Rank 0] Đang tải tập validation đã xử lý từ đĩa...")
            valset = load_from_disk(processed_dataset_path) 
        print(f"[Rank 0] Đã tạo tập Validation thành công.")

    print(f"[Rank {rank}] Đã tạo DataLoader thành công.")
    return trainloader, valset