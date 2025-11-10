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
        shuffled_ds = iterable_train_ds.shuffle(seed=hparams.seed, buffer_size=hparams.shuffle_buffer_size) # type: ignore
        sharded_ds = shuffled_ds.shard(num_shards=world_size, index=rank) # type: ignore
    else:
        sharded_ds = iterable_train_ds.shard(num_shards=world_size, index=rank) # type: ignore

    processed_ds = sharded_ds.map(
        prepare_text_mel,
        batched=True,
        batch_size=1000
    )
    
    trainloader = DataLoader(
        processed_ds, # type: ignore
        batch_size=hparams.batch_size,
        num_workers=0, # Chính xác, phải là 0 cho streaming
        pin_memory=False,    
        collate_fn=collate_fn
    )
    
    # --- 2. XỬ LÝ VALIDATION (Chỉ Rank 0) ---
    valset = None
    if rank == 0:
        if hparams.parquet_valid_file is not None:
            # Tải từ file parquet đã lưu sẵn
            val_dict = load_dataset(
                'parquet', 
                data_files={"validation": hparams.parquet_valid_file},
            )
            val_ds = val_dict["validation"] # type: ignore
        else:
            raise ValueError("Parquet valid file must be not be None for validation dataset.")
        
        processed_val_ds = val_ds.map( # type: ignore
            prepare_text_mel, 
            batched=True,
            batch_size=1000
        )
        
        valset = DataLoader(
            processed_val_ds, # type: ignore
            batch_size=hparams.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )

    print(f"[Rank {rank}] Đã tạo DataLoader thành công.")
    return trainloader, valset