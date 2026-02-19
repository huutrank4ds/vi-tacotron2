import gc
import random
import torch
from config import Hparams
from torch.optim import AdamW
import torch.distributed as dist
from dataloader import get_trainloader_chunk, load_speaker_embeddings, get_valloader
from model import Tacotron2
from loss import Tacotron2Loss
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm
import os
from datetime import timedelta
import builtins
from processing import PrepareTextMel, CollateTextMel
from utils import load_checkpoint_chunk, save_checkpoint_chunk, parse_batch_gpu
import torch.amp
import csv

# Override print để luôn flush output
if not hasattr(builtins, "original_print_safe"):
    builtins.original_print_safe = builtins.print #type: ignore
def print_flush(*args, **kwargs):
    kwargs['flush'] = True
    builtins.original_print_safe(*args, **kwargs) #type: ignore
builtins.print = print_flush


def init_distributed_training(rank, world_size, hparams: Hparams):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for DDP.")
    print(f"[Rank {rank}] Init DDP...")
    # Thiết lập GPU cho mỗi tiến trình
    torch.cuda.set_device(rank)

    dist.init_process_group(
        backend=hparams.ddp_backend,
        init_method=hparams.ddp_url,
        world_size=world_size,
        rank=rank,
        timeout=timedelta(minutes=60)
    )
    print(f"[Rank {rank}] DDP initialized on GPU {rank % torch.cuda.device_count()}.")

def train_worker_chunk_by_chunk(rank, world_size, hparams):
    # --- 1. Setup Device & DDP ---
    if hparams.ddp_run:
        device_id = rank % torch.cuda.device_count()
        init_distributed_training(device_id, world_size, hparams)
    else:
        device_id = 0 
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)

    # --- 2. Model Setup ---
    model = Tacotron2(hparams).to(device_id)
    if hparams.ddp_run:
        model = DDP(model, device_ids=[device_id])
    raw_model = model.module if hparams.ddp_run else model
    if hparams.guided_attention:
        print(f"[Rank {rank}] Guided Attention is ENABLED with sigma={hparams.guided_attention_sigma}.")
        criterion = Tacotron2Loss(hparams.n_frames_per_step, guided_sigma=hparams.guided_attention_sigma).to(device_id)
    else:
        print(f"[Rank {rank}] Guided Attention is DISABLED.")
        criterion = Tacotron2Loss(hparams.n_frames_per_step).to(device_id)
    optimizer = AdamW(model.parameters(), lr=hparams.learning_rate, weight_decay=hparams.weight_decay)

    # --- Scheduler (Optional) ---
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

    use_scaler = hparams.fp16_run and torch.cuda.get_device_capability()[0] < 8
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler) #type: ignore
    dtype_run = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    if use_scaler:
        if dtype_run == torch.bfloat16:
            print(f"[Rank {rank}] Using GradScaler with bfloat16.")
        else:
            print(f"[Rank {rank}] Using GradScaler with float16.")
    else:
        print(f"[Rank {rank}] Not using GradScaler.")
    
 
    # --- 3. Load Checkpoint ---
    global_epoch = 0 
    start_chunk_index = 0 
    best_val_loss = float('inf')
    patience_counter = 0  
    
    if not os.path.exists(hparams.checkpoint_path):
        os.makedirs(hparams.checkpoint_path)

    path_to_checkpoint = os.path.join(hparams.checkpoint_path, hparams.name_file_checkpoint)
    if os.path.exists(path_to_checkpoint):
        best_val_loss, epoch, chunk_index, patience_counter, end_epoch = load_checkpoint_chunk(path_to_checkpoint, raw_model, device_id, optimizer)
        if chunk_index + 1 >= len(hparams.dataset_chunks) or end_epoch:
            global_epoch = epoch + 1
            start_chunk_index = 0
        else:
            global_epoch = epoch
            start_chunk_index = chunk_index + 1
        print(f"[Rank {rank}] Resumed: Epoch {global_epoch}, Chunk {start_chunk_index}")

    # --- Init CSV Logger (Chỉ Rank 0) ---
    log_file = os.path.join(hparams.checkpoint_path, "training_log.csv")
    if rank == 0:
        if not os.path.exists(log_file):
            with open(log_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "chunk_idx", "chunk_dataset", 
                                 "train_loss", "train_mel", "train_postnet", "train_gate",
                                 "val_loss", "val_mel", "val_postnet", "val_gate"])

    # --- 4. Prepare Global Resources ---
    speaker_embedding_dict, speaker_embedding_dict_val = load_speaker_embeddings(hparams)
    prepare_text_mel_train = PrepareTextMel(hparams, speaker_embedding_dict)
    collate_fn = CollateTextMel(hparams)
    mel_transform = prepare_text_mel_train.get_mel_transform(hparams).to(device_id)

    val_set = None
    if rank == 0:
        print(f"[Rank {rank}] Preparing Validation Set...")
        prepare_text_mel_val = PrepareTextMel(hparams, speaker_embedding_dict_val)
        val_set = get_valloader(hparams, prepare_text_mel_val, collate_fn)

    # --- 5. Training Loop ---
    should_stop_global = False
    all_chunk_indices = list(range(len(hparams.dataset_chunks)))

    for epoch in range(global_epoch, hparams.max_epochs):
        if should_stop_global: break 

        if rank == 0:
            print(f"\n{'='*30} Epoch {epoch} {'='*30}")

        epoch_seed = hparams.seed + epoch
        random.seed(epoch_seed)
        shuffled_chunk_indices = all_chunk_indices.copy()
        if hparams.shuffle:
            random.shuffle(shuffled_chunk_indices)

        current_list_index = start_chunk_index if epoch == global_epoch else 0
        
        for list_idx in range(current_list_index, len(shuffled_chunk_indices)):
            chunk_idx = shuffled_chunk_indices[list_idx]
            chunk_name = os.path.basename(hparams.dataset_chunks[chunk_idx])
            
            # --- Tạo DataLoader ---
            train_loader = get_trainloader_chunk(
                rank, world_size, hparams, chunk_idx,
                prepare_text_mel_train, collate_fn,
                seed=hparams.seed + epoch
            )

            # --- Training Loop ---
            model.train()
            train_loader_iter = iter(train_loader)
            
            # Biến tích lũy loss cho chunk này
            total_train_loss = 0.0
            total_train_mel = 0.0
            total_train_post = 0.0
            total_train_gate = 0.0
            num_train_batches = 0

            if rank == 0:
                global_batch_size = hparams.batch_size * world_size
                total_samples = hparams.metadata.get(chunk_idx + 1, 0)
                total_steps = (total_samples + global_batch_size - 1) // global_batch_size
                pbar = tqdm(train_loader, 
                            desc=f"Epoch {epoch}| Chunk {list_idx} (Real: {chunk_name})", 
                            total=total_steps, 
                            unit="batch", 
                            position=0)
            else:
                pbar = train_loader_iter

            for i, batch in enumerate(pbar):
                optimizer.zero_grad()
                model_inputs, ground_truth = parse_batch_gpu(batch, device_id, mel_transform, hparams)
                
                with torch.amp.autocast(device_type='cuda', dtype=dtype_run, enabled=hparams.fp16_run): #type: ignore
                    model_outputs = model(model_inputs)
                    output_length = model_inputs[3]
                    if hparams.guided_attention:
                        loss, loss_mel, loss_mel_postnet, loss_gate = criterion(model_outputs, ground_truth, output_length, model_inputs[1])
                    else:
                        # Không tham số input_lengths thì hàm loss sẽ không tính guided attention loss
                        loss, loss_mel, loss_mel_postnet, loss_gate = criterion(model_outputs, ground_truth, output_length)

                if use_scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                # Tích lũy loss (chỉ cần lấy ở rank 0)
                # Ở đây lấy local loss của rank 0 làm đại diện cho nhanh
                if rank == 0:
                    total_train_loss += loss.item()
                    total_train_mel += loss_mel.item()
                    total_train_post += loss_mel_postnet.item()
                    total_train_gate += loss_gate.item()
                    num_train_batches += 1
                    
                    pbar.set_postfix({'Loss': f"{loss.item():.4f}",  #type: ignore
                                      'Mel': f"{loss_mel.item():.4f}",
                                      'Postnet': f"{loss_mel_postnet.item():.4f}",
                                      'Gate': f"{loss_gate.item():.4f}"})

            # --- VALIDATION ---
            if hparams.ddp_run:
                dist.barrier()

            stop_signal = torch.tensor(0).to(device_id) 

            if rank == 0 and val_set is not None:
                pbar.close() #type: ignore
                model.eval() 
                
                # Biến tích lũy Val Loss
                total_val_loss = 0.0
                total_val_mel = 0.0
                total_val_post = 0.0
                total_val_gate = 0.0
                
                with torch.no_grad():
                    val_progress = tqdm(val_set, desc="Validation", unit="batch", leave=False, position=1)
                    for val_batch in val_progress:
                        v_inputs, v_truth = parse_batch_gpu(val_batch, device_id, mel_transform, hparams)
                        with torch.amp.autocast(device_type='cuda', dtype=dtype_run, enabled=hparams.fp16_run): #type: ignore
                            v_outputs = model(v_inputs)
                            v_out_len = v_inputs[3]
                            v_loss, v_mel, v_mel_postnet, v_gate = criterion(v_outputs, v_truth, v_out_len)
                        
                        total_val_loss += v_loss.item()
                        total_val_mel += v_mel.item()
                        total_val_post += v_mel_postnet.item()
                        total_val_gate += v_gate.item()
                    val_progress.close()
                
                # Tính trung bình
                avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0
                avg_train_mel = total_train_mel / num_train_batches if num_train_batches > 0 else 0
                avg_train_post = total_train_post / num_train_batches if num_train_batches > 0 else 0
                avg_train_gate = total_train_gate / num_train_batches if num_train_batches > 0 else 0

                avg_val_loss = total_val_loss / len(val_set)
                avg_val_mel = total_val_mel / len(val_set)
                avg_val_post = total_val_post / len(val_set)
                avg_val_gate = total_val_gate / len(val_set)

                print(f"\n[Rank {rank}] Epoch {epoch} | Chunk {list_idx} | Train Loss: {avg_train_loss:.5f}, Train_Mel: {avg_train_mel:.5f}, Train_Post: {avg_train_post:.5f}, Train_Gate: {avg_train_gate:.5f} \n"
                      f"| Val Loss: {avg_val_loss:.5f} | Mel: {avg_val_mel:.5f} | Post: {avg_val_post:.5f} | Gate: {avg_val_gate:.5f}")

                # Ghi vào CSV
                with open(log_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, list_idx, chunk_name, 
                                     avg_train_loss, avg_train_mel, avg_train_post, avg_train_gate,
                                     avg_val_loss, avg_val_mel, avg_val_post, avg_val_gate])
                
                if list_idx + 1 >= len(hparams.dataset_chunks):
                    end_epoch = True
                else:
                    end_epoch = False
                # Checkpoint & Early Stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    save_name = f"checkpoint_vi_tacotron2_best.pt"
                    # Kiểm tra hết chunk list chưa
                    save_checkpoint_chunk(raw_model, optimizer, best_val_loss, patience_counter, epoch, list_idx, end_epoch, save_name, hparams)
                    print(f"Saved NEW BEST model: {save_name}")
                else:
                    patience_counter += 1
                    save_name = f"checkpoint_vi_tacotron2_last.pt"
                    save_checkpoint_chunk(raw_model, optimizer, best_val_loss, patience_counter, epoch, list_idx, end_epoch, save_name, hparams)
                    print(f"Saved Last Checkpoint: {save_name}")

                    if patience_counter >= hparams.early_stopping_patience:
                        print(f"==> EARLY STOPPING TRIGGERED!")
                        stop_signal = torch.tensor(1).to(device_id)

                model.train()
                # scheduler.step() # Nếu dùng scheduler

            if hparams.ddp_run:
                dist.broadcast(stop_signal, src=0)
            
            if stop_signal.item() == 1:
                should_stop_global = True
                if rank == 0: print("Master requested stop.")
            
            # Clean up memory
            del train_loader
            if 'pbar' in locals(): del pbar
            if 'train_loader_iter' in locals(): del train_loader_iter
            gc.collect()
            torch.cuda.empty_cache()
            
            if hparams.ddp_run: dist.barrier()

            if should_stop_global: break
            
        # --- [NEW] KẾT THÚC EPOCH ---
        if rank == 0 and not should_stop_global:
            # Lưu Checkpoint Epoch riêng (để backup)
            epoch_save_name = f"checkpoint_vi_tacotron2_epoch_{epoch}.pt"
            # Lưu chunk_index = 0 để lần sau load lên biết là bắt đầu epoch mới
            save_checkpoint_chunk(raw_model, optimizer, best_val_loss, patience_counter, epoch, len(hparams.dataset_chunks) - 1,  True, epoch_save_name, hparams)
            print(f"[Epoch Checkpoint] Saved: {epoch_save_name}")

        start_chunk_index = 0

    if hparams.ddp_run:
        dist.destroy_process_group()