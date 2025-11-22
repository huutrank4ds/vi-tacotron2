from torch.utils.data import DataLoader
from datasets import load_dataset
from processing import PrepareTextMel, CollateTextMel
from config import Hparams
from huggingface_hub import list_repo_files, hf_hub_url
import torch


def get_parquet_file_list(hparams: Hparams):
    """Lấy danh sách URL đầy đủ của các file parquet trong split 'train'."""
    all_files = list_repo_files(
        repo_id=hparams.dataset_name,
        repo_type='dataset',
    )
    train_files = [f for f in all_files if f.startswith(f'{hparams.hf_parquets_folder}/train-') and f.endswith('.parquet')]
    file_urls = [
        hf_hub_url(
            repo_id=hparams.dataset_name,
            filename=f,
            repo_type='dataset'
        ) for f in train_files
    ]
    print(f"Found {len(file_urls)} parquet files for training dataset.")
    return file_urls

def get_valset(hparams: Hparams):
    """Lấy URL đầy đủ của file parquet validation."""
    if hparams.parquet_valid_file is None:
        raise ValueError("Parquet valid file must be not be None for validation dataset.")
    
    val_dict = load_dataset(
            'parquet', 
            data_files={"validation": hparams.parquet_valid_file},
        )
    val_ds = val_dict["validation"] # type: ignore
    print("Validation dataset loaded from parquet file.")
    return val_ds


# === HÀM TẠO DATALOADER ===
def get_trainloader_valset(rank, world_size, hparams: Hparams, seed=None):
    """
    Tải và chuẩn bị DataLoader cho training (streaming, sharded)
    và Validation set (chỉ rank 0, non-streaming).
    """
    
    seed = hparams.seed if seed is None else seed
    
    # --- 0. Khởi tạo processors ---
    # Sử dụng đối tượng hparams được truyền vào hàm
    try: 
        if hparams.speaker_embeddings_file is None or hparams.validation_speaker_embeddings_file is None:
            raise ValueError("Speaker embeddings file path must be provided in hparams.")
        speaker_embedding_dict = torch.load(hparams.speaker_embeddings_file)
        speaker_embedding_dict_val = torch.load(hparams.validation_speaker_embeddings_file)
    except Exception as e:
        raise RuntimeError(f"Error loading speaker embeddings: {e}")
    prepare_text_mel_train = PrepareTextMel(hparams, speaker_embedding_dict)
    prepare_text_mel_val = PrepareTextMel(hparams, speaker_embedding_dict_val) 
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
        shuffled_ds = iterable_train_ds.shuffle(seed=seed, buffer_size=hparams.shuffle_buffer_size) # type: ignore
        sharded_ds = shuffled_ds.shard(num_shards=world_size, index=rank) # type: ignore
    else:
        sharded_ds = iterable_train_ds.shard(num_shards=world_size, index=rank) # type: ignore

    processed_ds = sharded_ds.map(
        prepare_text_mel_train,
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
            prepare_text_mel_val, 
            batched=True,
            batch_size=1000
        )
        
        valset = DataLoader(
            processed_val_ds, # type: ignore
            batch_size=hparams.batch_size,
            collate_fn=collate_fn,
            num_workers=0
        )

    print(f"[Rank {rank}] DataLoader created successfully.")
    return trainloader, valset