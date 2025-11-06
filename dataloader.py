from torch.utils.data import DataLoader
from datasets import load_dataset
from processing import PrepareTextMel, CollateTextMel
from config import Hparams

DATASET_NAME = Hparams.dataset_name
DATASET_CONFIG = Hparams.dataset_config

prepare_text_mel = PrepareTextMel(Hparams())
collate_fn = CollateTextMel(Hparams())

# === HÀM TẠO DATALOADER ===
def get_trainloader_valset(rank, world_size, hparams: Hparams):
    # Tải dataset dạng streamable cho tập train
    iterable_train_ds = load_dataset(
        DATASET_NAME, 
        DATASET_CONFIG, 
        split='train', 
        streaming=True,
        trust_remote_code=True
    )

    # Xáo trộn trước khi chia các batch nhỏ (có thể tái tạo kết quả)
    if hparams.shuffle:
        shuffled_ds = iterable_train_ds.shuffle(seed=hparams.seed, buffer_size=hparams.shuffle_buffer_size)
        # Phân mảnh (shard) dataset cho DDP
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
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    valset = None
    if rank == 0:
        valset = load_dataset(
            DATASET_NAME, 
            DATASET_CONFIG, 
            split='validation', 
            streaming=False,
            trust_remote_code=True
        )

        valset = valset.map(
            prepare_text_mel,
            batched=True,
            batch_size=1000
        )
        print(f"[Rank {rank}] Đã tạo tập Validation thành công.")

    print(f"[Rank {rank}] Đã tạo DataLoader thành công.")
    return trainloader, valset