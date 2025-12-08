from torch.utils.data import DataLoader
from datasets import load_dataset
from processing import PrepareTextMel, CollateTextMel
from config import Hparams
from huggingface_hub import list_repo_files, hf_hub_url
import torch
from pathlib import Path
import torch.distributed as dist

def get_valset(hparams: Hparams):
    """Tải validation set từ file parquet đã lưu sẵn."""
    if hparams.parquet_valid_file is not None:
        # Tải từ file parquet đã lưu sẵn
        val_dict = load_dataset(
            'parquet', 
            data_files={"validation": hparams.parquet_valid_file},
        )
        val_ds = val_dict["validation"] # type: ignore
    else:
        raise ValueError("Parquet valid file must be not be None for validation dataset.")
    return val_ds

def get_valloader(hparams: Hparams, prepare_text_mel_val, collate_fn):
    """Tải DataLoader cho validation set."""
    val_ds = get_valset(hparams)
    
    processed_val_ds = val_ds.map( # type: ignore
        prepare_text_mel_val, 
        batched=True,
        batch_size=1000
    )
    
    valloader = DataLoader(
        processed_val_ds, # type: ignore
        batch_size=hparams.batch_size,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=False,
        shuffle=False # Validation không cần shuffle
    )
    return valloader

def load_speaker_embeddings(hparams: Hparams):
    """Tải speaker embeddings từ file đã lưu sẵn."""
    try: 
        if hparams.speaker_embeddings_file is None or hparams.validation_speaker_embeddings_file is None:
            raise ValueError("Speaker embeddings file path must be provided in hparams.")
        speaker_embedding_dict = torch.load(hparams.speaker_embeddings_file)
        speaker_embedding_dict_val = torch.load(hparams.validation_speaker_embeddings_file)
    except Exception as e:
        raise RuntimeError(f"Error loading speaker embeddings: {e}")
    return speaker_embedding_dict, speaker_embedding_dict_val

def load_dataset_chunks(rank, hparams: Hparams, index: int):
    """
    Tải một chunk dataset.
    Tối ưu DDP: Rank 0 tải và ghi cache trước, các Rank khác đợi và dùng lại cache.
    """
    if index < 0 or index >= len(hparams.dataset_chunks):
        raise IndexError("Index out of range for dataset chunks.")
    
    file_pattern = str(Path(hparams.dataset_chunks[index]) / '*.parquet')

    print(f"[Rank {rank}] Generating cache for chunk {index}...")
    dataset = load_dataset(
        'parquet', 
        data_files={'train': file_pattern},
        split='train',
        streaming=True
    )
    print(f"[Rank {rank}] Dataset chunk {index} ready.") # type: ignore
    return dataset

def remove_chunk_cache(hparams: Hparams, index: int):
    """Xóa cache của chunk dataset đã load."""
    chunk_cache_dir = Path(hparams.cache_chunk_dir) / f"chunk_{index}"
    if chunk_cache_dir.exists() and chunk_cache_dir.is_dir():
        for item in chunk_cache_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                import shutil
                shutil.rmtree(item)
        chunk_cache_dir.rmdir()
        print(f"Removed cache for chunk {index}.")
    else:
        print(f"No cache found for chunk {index} to remove.")

def get_trainloader_chunk(
    rank, 
    world_size, 
    hparams: Hparams, 
    index: int, 
    prepare_text_mel_train,
    collate_fn,
    seed=None
):
    seed = hparams.seed if seed is None else seed

    # --- 1. XỬ LÝ TRAINING CHUNK ---
    # Load dataset (Map-style Arrow)
    dataset_chunk = load_dataset_chunks(rank, hparams, index)
    
    # Thực hiện global shuffle (tùy chọn) và sharding
    if hparams.shuffle:
        shuffled_ds = dataset_chunk.shuffle(seed=seed, buffer_size=hparams.shuffle_buffer_size) # type: ignore
        sharded_ds = shuffled_ds.shard(num_shards=world_size, index=rank) # type: ignore
    else:
        sharded_ds = dataset_chunk.shard(num_shards=world_size, index=rank) # type: ignore

    try:
        remove_cols = list(dataset_chunk.features.keys()) # type: ignore
    except:
        remove_cols = ['audio', 'text', 'speaker']

    # Map xử lý
    processed_ds = sharded_ds.map(
        prepare_text_mel_train,
        batched=True,
        batch_size=1000,
        remove_columns=remove_cols
    )
    
    # Tạo DataLoader
    trainloader = DataLoader(
        processed_ds, # type: ignore
        batch_size=hparams.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,     
        collate_fn=collate_fn,
    )
    print(f"[Rank {rank}] DataLoader for chunk {index} created.")
    return trainloader


def get_trainloader_valset(rank, world_size, hparams: Hparams, seed=None):
    """
    Tải và chuẩn bị DataLoader cho training (streaming, sharded)
    và Validation set (chỉ rank 0, non-streaming).
    """
    
    seed = hparams.seed if seed is None else seed
    
    # --- 0. Khởi tạo processors ---
    # Sử dụng đối tượng hparams được truyền vào hàm
    speaker_embedding_dict, speaker_embedding_dict_val = load_speaker_embeddings(hparams)
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
        valset = get_valloader(hparams, prepare_text_mel_val, collate_fn)

    print(f"[Rank {rank}] DataLoader created successfully.")
    return trainloader, valset