import os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from processing import preprocess_function, tacotron2_collate

DATASET_NAME = "thivux/phoaudiobook" 
DATASET_CONFIG = None

# === HÀM TẠO DATALOADER ===
def create_dataloader(rank, world_size, batch_size_per_gpu, num_workers_per_gpu):
    iterable_ds = load_dataset(
        DATASET_NAME, 
        DATASET_CONFIG, 
        split='train', 
        streaming=True,
        trust_remote_code=True
    )

    # Xáo trộn trước khi chia các batch nhỏ (có thể tái tạo kết quả)
    shuffled_ds = iterable_ds.shuffle(seed=42, buffer_size=10_000)
    # Phân mảnh dataset
    sharded_ds = shuffled_ds.shard(num_shards=world_size, index=rank)

    processed_ds = sharded_ds.map(
        preprocess_function,
        batched=True,
        batch_size=1000
    )

    processed_ds = processed_ds.with_format("torch")
    
    loader = DataLoader(
        processed_ds,
        batch_size=batch_size_per_gpu,
        num_workers=num_workers_per_gpu,
        pin_memory=True,
        collate_fn=tacotron2_collate
    )
    
    print(f"[Rank {rank}] Đã tạo DataLoader thành công.")
    return loader