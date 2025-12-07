import gc
import random
import torch
from config import Hparams
from torch.optim import AdamW
import torch.distributed as dist
from dataloader import get_trainloader_valset, get_trainloader_chunk, load_speaker_embeddings, get_valloader, remove_chunk_cache
from model import Tacotron2
from loss import Tacotron2Loss
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm
import os
from datetime import timedelta
import builtins
from processing import PrepareTextMel, CollateTextMel
from utils import to_gpu, load_checkpoint_chunk


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

def parse_batch_gpu(batch, device, mel_transform, hparams):
    # Helper chuyển GPU (dùng hàm của bạn)
    # Lưu ý: device ở đây chính là rank (int) hoặc device object đều được
    
    # 1. Load dữ liệu
    text_padded = to_gpu(batch['text_inputs'], device).long()
    input_lengths = to_gpu(batch['text_lengths'], device).long()
    speaker_embeddings = to_gpu(batch['speaker_embeddings'], device).float()
    wav_lengths = to_gpu(batch['wav_lengths'], device).long()
    audio_list = to_gpu(batch['audio_tensors'], device)
    
    # 2. Tính Mel Spectrograms
    mels = mel_transform(audio_list) # audio_list đã pad 0.0
    mels = torch.log(torch.clamp(mels, min=1e-5))
    
    # Tính độ dài thực tế (Dựa trên wav lengths gốc)
    r = hparams.n_frames_per_step
    remainder = mels.shape[2] % r
    if remainder != 0:
        pad_size = r - remainder
        # Pad chiều cuối (Time)
        mels = torch.nn.functional.pad(mels, (0, pad_size), value=hparams.mel_pad_value)

    # Tính độ dài thực tế
    mel_lengths = 1 + (wav_lengths // hparams.hop_length)
    
    # 3. Tạo Gate Target (Stop Token)
    max_mel_len = mels.shape[2]
    gate_padded = torch.zeros(mels.shape[0], max_mel_len, device=device)

    # Đánh dấu 1 ở vị trí kết thúc
    for i, l in enumerate(mel_lengths):
        end_idx = min(l - 1, max_mel_len - 1)
        gate_padded[i, end_idx:] = 1.0 

    return (
        (text_padded, input_lengths, mels, mel_lengths, speaker_embeddings),
        (mels, gate_padded)
    )

def save_checkpoint_chunk(model, optimizer, best_val_loss, patience_counter, epoch, chunk_index, filepath, hparams: Hparams):
    model_state_dict = model.state_dict()
    checkpoint_dict = {
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'patience_counter': patience_counter,
        'epoch': epoch,
        'chunk_index': chunk_index
    }
    if not os.path.exists(hparams.checkpoint_path):
        os.makedirs(hparams.checkpoint_path)
    torch.save(checkpoint_dict, os.path.join(hparams.checkpoint_path, filepath))
    print(f"Saved checkpoint at {os.path.join(hparams.checkpoint_path, filepath)}")

def save_checkpoint_step(model, optimizer, best_val_loss, epoch, step, filepath, hparams: Hparams):
    model_state_dict = model.state_dict()
    checkpoint_dict = {
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'best_val_loss': best_val_loss,
        'epoch': epoch
    }
    if not os.path.exists(hparams.checkpoint_path):
        os.makedirs(hparams.checkpoint_path)
    torch.save(checkpoint_dict, os.path.join(hparams.checkpoint_path, filepath))
    print(f"Saved checkpoint at {os.path.join(hparams.checkpoint_path, filepath)}")

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

    criterion = Tacotron2Loss().to(device_id)
    optimizer = AdamW(model.parameters(), lr=hparams.learning_rate, weight_decay=hparams.weight_decay)

    # --- 3. Load Checkpoint ---
    global_epoch = 0 
    start_chunk_index = 0 
    best_val_loss = float('inf')
    patience_counter = 0  

    path_to_checkpoint = os.path.join(hparams.checkpoint_path, hparams.name_file_checkpoint)
    if os.path.exists(path_to_checkpoint):
        best_val_loss, epoch, chunk_index, patience_counter = load_checkpoint_chunk(path_to_checkpoint, raw_model, device_id, optimizer)
        if chunk_index + 1 >= len(hparams.dataset_chunks):
            global_epoch = epoch + 1
            start_chunk_index = 0
        else:
            global_epoch = epoch
            start_chunk_index = chunk_index + 1
        print(f"[Rank {rank}] Resumed: Epoch {global_epoch}, Chunk {start_chunk_index}")

    # --- 4. Prepare Global Resources ---
    # Load Embeddings
    speaker_embedding_dict, speaker_embedding_dict_val = load_speaker_embeddings(hparams)
    
    # Processors & Collate
    prepare_text_mel_train = PrepareTextMel(hparams, speaker_embedding_dict)
    collate_fn = CollateTextMel(hparams)

    # mel_transform dùng chung cho cả Train và Val, và cho tất cả các Batch
    mel_transform = prepare_text_mel_train.get_mel_transform(hparams).to(device_id)

    # Validation Set (Chỉ Rank 0)
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
        random.shuffle(shuffled_chunk_indices)

        current_list_index = start_chunk_index if epoch == global_epoch else 0
        
        for list_idx in range(current_list_index, len(shuffled_chunk_indices)):
            chunk_idx = shuffled_chunk_indices[list_idx]
            
            # --- Tạo DataLoader ---
            train_loader = get_trainloader_chunk(
                rank, world_size, hparams, chunk_idx,
                prepare_text_mel_train, # Truyền processor đã init
                collate_fn,
                seed=hparams.seed + epoch
            )

            # --- Training Loop ---
            model.train()
            train_loader_iter = iter(train_loader)
            if rank == 0:
                global_batch_size = hparams.batch_size * world_size
                total_samples = hparams.metadata.get(chunk_idx + 1, 0)
                total_steps = (total_samples + global_batch_size - 1) // global_batch_size
                pbar = tqdm(train_loader, 
                            desc=f"Epoch {epoch}|Chunk {chunk_idx}", 
                            total=total_steps, 
                            unit="batch", 
                            position=0)
            else:
                pbar = train_loader_iter

            for i, batch in enumerate(pbar):
                optimizer.zero_grad()
                model_inputs, ground_truth = parse_batch_gpu(batch, device_id, mel_transform, hparams)
                model_outputs = model(model_inputs)
                output_length = model_inputs[3]
                loss, loss_mel, loss_mel_postnet, loss_gate = criterion(model_outputs, ground_truth, output_length)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip_thresh)
                optimizer.step()
                
                if rank == 0:
                    pbar.set_postfix({'Loss': f"{loss.item():.4f}", # type: ignore
                                      'Mel': f"{loss_mel.item():.4f}",
                                      'Postnet': f"{loss_mel_postnet.item():.4f}",
                                      'Gate': f"{loss_gate.item():.4f}"}) # type: ignore
            if rank == 0:
                pbar.close() # type: ignore
                del pbar
            # --- VALIDATION ---
            if hparams.ddp_run:
                dist.barrier()

            stop_signal = torch.tensor(0).to(device_id) # 0: Continue, 1: Stop
            # Chỉ Rank 0 thực hiện tính toán Validation
            if rank == 0 and val_set is not None:
                model.eval()
                total_val_loss = 0.0
                with torch.no_grad():
                    val_progress = tqdm(val_set, desc="Validation", unit="batch", leave=False, position=1)
                    for val_batch in val_progress:
                        v_inputs, v_truth = parse_batch_gpu(val_batch, device_id, mel_transform, hparams)
                        v_outputs = model(v_inputs)
                        v_out_len = v_inputs[3]
                        v_loss, v_mel, v_mel_postnet, v_gate = criterion(v_outputs, v_truth, v_out_len)
                        total_val_loss += v_loss.item()
                    val_progress.close()
                
                avg_val_loss = total_val_loss / len(val_set)
                print(f"\n[Rank {rank}] Epoch {epoch} | Chunk {list_idx} | Train Loss: {loss.item():.5f} | Mel: {loss_mel.item():.5f} | Postnet: {loss_mel_postnet.item():.5f} | Gate: {loss_gate.item():.5f} \n \
                      | Val Loss: {avg_val_loss:.5f} | Mel: {v_mel.item():.5f} | Postnet: {v_mel_postnet.item():.5f} | Gate: {v_gate.item():.5f} ")

                # ===> [FEATURE 2] LOGIC EARLY STOPPING <===
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Lưu best model
                    save_name = f"{hparams.name_file_checkpoint}_best.pt"
                    save_checkpoint_chunk(raw_model, optimizer, best_val_loss, patience_counter, epoch, list_idx, save_name, hparams)
                    print(f"Saved NEW BEST model at epoch {epoch}, chunk {list_idx}: {save_name}")
                else:
                    patience_counter += 1
                    if patience_counter >= hparams.early_stopping_patience:
                        print(f"==> EARLY STOPPING TRIGGERED at epoch {epoch}, chunk {list_idx}!")
                        stop_signal = torch.tensor(1).to(device_id) # Bật tín hiệu dừng
                    print(f"Best Val Loss remains: {best_val_loss:.5f} | Patience: {patience_counter}/{hparams.early_stopping_patience}")
                    save_name = f"{hparams.name_file_checkpoint}_last.pt"
                    save_checkpoint_chunk(raw_model, optimizer, best_val_loss, patience_counter, epoch, list_idx, save_name, hparams)
                    print(f"Saved checkpoint at epoch {epoch}, chunk {list_idx}: {save_name}")

                model.train()
            
            if hparams.ddp_run:
                # Rank 0 truyền tín hiệu dừng cho các Rank khác
                dist.broadcast(stop_signal, src=0)
            
            if stop_signal.item() == 1:
                should_stop_global = True
                if rank == 0:
                    print("Master requested stop. Stopping all workers...")
            
            # Barrier lần nữa để đảm bảo tất cả cùng thoát hoặc cùng tiếp tục
            if hparams.ddp_run:
                dist.barrier()

        start_chunk_index = 0

    if hparams.ddp_run:
        dist.destroy_process_group()


def train_worker_by_step(rank, world_size, hparams):
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

    criterion = Tacotron2Loss().to(device_id)
    optimizer = AdamW(model.parameters(), lr=hparams.learning_rate, weight_decay=hparams.weight_decay)

    # --- 3. Load Checkpoint ---
    global_step = 0
    epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0  # Biến đếm cho Early Stopping

    path_to_checkpoint = os.path.join(hparams.checkpoint_path, hparams.name_file_checkpoint)
    if os.path.exists(path_to_checkpoint):
        map_loc = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(path_to_checkpoint, map_location=map_loc)
        
        raw_model.load_state_dict(checkpoint['model_state_dict']) #type: ignore
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        global_step = checkpoint['step']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        epoch = checkpoint.get('epoch', 0)
        # Reset patience khi resume training để tránh dừng ngay lập tức
        patience_counter = 0 
        print(f"[Rank {rank}] Resumed from step {global_step}, Epoch {epoch}.")

    # --- 4. Data ---
    train_loader, val_set = get_trainloader_valset(
        device_id, world_size, hparams, hparams.seed + epoch + 1
    )
    train_data_iter = iter(train_loader)

    # --- 5. Progress Bar ---
    progress_bar = None
    step_training = hparams.max_step_training + global_step
    if rank == 0:
        progress_bar = tqdm(initial=global_step, total=step_training, desc="Training", unit="step", position=0)

    # --- 6. Training Loop ---
    model.train()
    
    # Cờ để báo hiệu dừng toàn bộ hệ thống (Early Stopping Flag)
    should_stop_training = False

    while global_step < step_training and not should_stop_training:
        
        # --- [A] Lấy Batch & Xử lý Epoch ---
        try:
            batch = next(train_data_iter)
        except StopIteration:
            # ===> [FEATURE 1] KẾT THÚC EPOCH <===
            # 1. Thông báo
            if rank == 0:
                print(f"\n[Rank {rank}] Epoch {epoch} finished.")
                
                # 2. Lưu Checkpoint Epoch
                epoch_save_name = f"checkpoint_epoch_{epoch}.pt"
                save_checkpoint_step(raw_model, optimizer, best_val_loss, epoch, global_step, epoch_save_name, hparams)
                print(f"[Checkpoint] Saved epoch checkpoint: {epoch_save_name}")

            # 3. Tăng epoch và reset iterator
            epoch += 1
            # if hparams.ddp_run and hasattr(train_loader.sampler, 'set_epoch'):
            #     train_loader.sampler.set_epoch(epoch)
            
            train_data_iter = iter(train_loader)
            batch = next(train_data_iter)

        # --- [B] Training Step ---
        optimizer.zero_grad()
        model_inputs, ground_truth = raw_model.parse_batch(batch, rank) #type: ignore
        model_outputs = model(model_inputs)
        
        output_length = model_inputs[3]
        loss, loss_mel, loss_mel_postnet, loss_gate = criterion(
            model_outputs, ground_truth, output_length
        )
        
        loss.backward()
        
        # Gradient Clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip_thresh)
        
        optimizer.step()
        global_step += 1

        if rank == 0:
            progress_bar.update(1) # type: ignore
            progress_bar.set_postfix({  # type: ignore
                'Loss': f"{loss.item():.4f}",
                'Mel': f"{loss_mel.item():.4f}",
                'Postnet': f"{loss_mel_postnet.item():.4f}",
                'Gate': f"{loss_gate.item():.4f}"
            })

        # --- [C] Validation & Early Stopping ---
        if global_step % hparams.val_interval == 0:
            
            # [QUAN TRỌNG] Đồng bộ hóa trước khi validate
            # Để đảm bảo Rank 0 không validate trong khi Rank 1 đang chạy train tiếp
            if hparams.ddp_run:
                dist.barrier()

            stop_signal = torch.tensor(0).to(device_id) # 0: Continue, 1: Stop

            # Chỉ Rank 0 thực hiện tính toán Validation
            if rank == 0 and val_set is not None:
                model.eval()
                total_val_loss = 0.0
                with torch.no_grad():
                    val_progress = tqdm(val_set, desc="Validation", unit="batch", leave=False, position=1)
                    for val_batch in val_progress:
                        v_inputs, v_truth = raw_model.parse_batch(val_batch, rank) #type: ignore
                        v_outputs = model(v_inputs)
                        v_out_len = v_inputs[3]
                        v_loss, _, _, _ = criterion(v_outputs, v_truth, v_out_len)
                        total_val_loss += v_loss.item()
                    val_progress.close()
                
                avg_val_loss = total_val_loss / len(val_set)
                print(f"\n[Rank {rank}] Step {global_step} | Val Loss: {avg_val_loss:.5f} | Patience: {patience_counter}/{hparams.early_stopping_patience}")

                # ===> [FEATURE 2] LOGIC EARLY STOPPING <===
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Lưu best model
                    save_name = f"checkpoint_step_{global_step}_best.pt"
                    save_checkpoint_step(raw_model, optimizer, best_val_loss, epoch, global_step, save_name, hparams)
                    print(f"Saved NEW BEST model: {save_name}")
                else:
                    patience_counter += 1
                    if patience_counter >= hparams.early_stopping_patience:
                        print(f"==> EARLY STOPPING TRIGGERED at step {global_step}!")
                        stop_signal = torch.tensor(1).to(device_id) # Bật tín hiệu dừng

                model.train()

            # [QUAN TRỌNG] Đồng bộ quyết định dừng cho tất cả GPU
            if hparams.ddp_run:
                # Rank 0 truyền tín hiệu dừng cho các Rank khác
                dist.broadcast(stop_signal, src=0)
            
            if stop_signal.item() == 1:
                should_stop_training = True
                if rank == 0:
                    print("Master requested stop. Stopping all workers...")
            
            # Barrier lần nữa để đảm bảo tất cả cùng thoát hoặc cùng tiếp tục
            if hparams.ddp_run:
                dist.barrier()

    # --- 8. Cleanup ---
    if rank == 0:
        progress_bar.close() # type: ignore
    
    print(f"[Rank {rank}] Training process finished.")
    if hparams.ddp_run:
        dist.destroy_process_group()