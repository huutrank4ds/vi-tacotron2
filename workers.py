import torch
from config import Hparams
from torch.optim import AdamW
import torch.distributed as dist
from dataloader import get_trainloader_valset
from model import Tacotron2
from loss import Tacotron2Loss
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm
import os
from datetime import timedelta
import builtins

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

def save_checkpoint(model, optimizer, epoch, step, filepath, hparams: Hparams):
    model_state_dict = model.module.state_dict()
    checkpoint_dict = {
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step
    }
    if not os.path.exists(hparams.checkpoint_path):
        os.makedirs(hparams.checkpoint_path)
    torch.save(checkpoint_dict, os.path.join(hparams.checkpoint_path, filepath))
    print(f"Đã lưu checkpoint tại {os.path.join(hparams.checkpoint_path, filepath)}")

def save_checkpoint_step(model, optimizer, best_val_loss, epoch, step, filepath, hparams: Hparams):
    model_state_dict = model.module.state_dict()
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

def train_worker(rank, world_size, hparams: Hparams):
    # --- 1. KHỞI TẠO DDP VÀ DATALOADER ---
    device_id = rank % torch.cuda.device_count()
    if hparams.ddp_run:
        init_distributed_training(device_id, world_size, hparams)

    # Giả sử get_trainloader_valset trả về loader dùng TextMelCollate
    train_loader, val_set = get_trainloader_valset(
        device_id, 
        world_size, 
        hparams
    )

    # Load model bên trong hàm worker
    model = Tacotron2(hparams)
    model = model.to(device_id)
    model = DDP(model, device_ids=[device_id])

    # --- 2. KHỞI TẠO HÀM LOSS VÀ OPTIMIZER ---
    criterion = Tacotron2Loss().to(device_id) 
    optimizer = AdamW(model.parameters(), lr=hparams.learning_rate, weight_decay=hparams.weight_decay)
    
    print(f"[Rank {rank}] Bắt đầu huấn luyện...")
    
    for epoch in range(hparams.epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            
            optimizer.zero_grad()

            # Dùng parse_batch của model để chuẩn bị dữ liệu
            # Hàm này đã tự chuyển tensor sang GPU
            model_inputs, ground_truth = model.module.parse_batch(batch, rank=device_id)
            
            # Forward pass
            # model_inputs là tuple: (text_padded, input_lengths, ...)
            model_outputs = model(model_inputs)

            output_length = model_inputs[3]  # output_lengths

            loss, loss_mel, loss_mel_postnet, loss_gate = criterion(
                model_outputs, ground_truth, output_length
            )
            
            # Backward và Optimize
            loss.backward()
            optimizer.step()
            
            if rank == 0 and step % 1 == 0:
                print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}")
                print(f"Mel: {loss_mel.item()}, Postnet: {loss_mel_postnet.item()}, Gate: {loss_gate.item()}")
        if device_id == 0 and val_set is not None:
            # Thực hiện đánh giá trên tập validation
            model.eval()
            best_val_loss = float('inf')
            total_val_loss = 0.0
            with torch.no_grad():
                for batch in val_set:
                    model_inputs, ground_truth = model.module.parse_batch(batch)
                    model_outputs = model(model_inputs)
                    output_lengths = model_inputs[4]
                    val_loss, _, _, _ = criterion(
                        model_outputs, ground_truth, output_lengths
                    )
                    total_val_loss += val_loss.item()
            avg_val_loss = total_val_loss / len(val_set)
            print(f"[Rank {rank}] Epoch {epoch} Validation Loss: {avg_val_loss}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_checkpoint(model, optimizer, epoch, step, f"checkpoint_epoch_{epoch}.pt", hparams)
        if hparams.ddp_run:
            dist.barrier()  # Đồng bộ hóa các tiến trình sau mỗi epoch

    print(f"[Rank {rank}] Huấn luyện hoàn tất.")
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

    # --- 3. Optimizer & Loss ---
    criterion = Tacotron2Loss().to(device_id)
    optimizer = AdamW(model.parameters(), lr=hparams.learning_rate, weight_decay=hparams.weight_decay)

    # --- 4. Load Checkpoint ---
    global_step = 0
    epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0  # [NEW] Biến đếm cho Early Stopping

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

    # --- 5. Data ---
    train_loader, val_set = get_trainloader_valset(
        device_id, world_size, hparams, hparams.seed + epoch + 1
    )
    train_data_iter = iter(train_loader)

    # --- 6. Progress Bar ---
    progress_bar = None
    step_training = hparams.max_step_training + global_step
    if rank == 0:
        progress_bar = tqdm(initial=global_step, total=step_training, desc="Training", unit="step", position=0)

    # --- 7. Training Loop ---
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

            val_loss_tensor = torch.tensor(0.0).to(device_id)
            stop_signal = torch.tensor(0).to(device_id) # 0: Continue, 1: Stop

            # Chỉ Rank 0 thực hiện tính toán Validation
            if rank == 0 and val_set is not None:
                model.eval()
                total_val_loss = 0.0
                with torch.no_grad():
                    val_progress = tqdm(val_set, desc="Validation", unit="batch", leave=False, position=1)
                    for val_batch in val_progress:
                        v_inputs, v_truth = raw_model.parse_batch(val_batch) #type: ignore
                        v_outputs = model(v_inputs)
                        v_out_len = v_inputs[3]
                        v_loss, _, _, _ = criterion(v_outputs, v_truth, v_out_len)
                        total_val_loss += v_loss.item()
                    val_progress.close()
                
                avg_val_loss = total_val_loss / len(val_set)
                val_loss_tensor = torch.tensor(avg_val_loss).to(device_id)
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


# def train_worker_by_step(rank, world_size, hparams):
#     """
#     Hàm worker huấn luyện chính.
#     Hỗ trợ cả chế độ DDP (Nhiều GPU) và Single-GPU.
#     """
    
#     # --- 1. Cấu hình Device & DDP ---
#     if hparams.ddp_run:
#         # Chế độ đa GPU
#         device_id = rank % torch.cuda.device_count()
#         init_distributed_training(device_id, world_size, hparams)
#     else:
#         # Chế độ đơn GPU
#         device_id = 0 
#         if torch.cuda.is_available():
#             torch.cuda.set_device(device_id)
#         print(f"[Info] Running in Single-GPU mode on device {device_id}")

#     # --- 2. Khởi tạo Model ---
#     model = Tacotron2(hparams)
#     model = model.to(device_id)

#     # Chỉ bọc DDP nếu đang chạy chế độ DDP
#     if hparams.ddp_run:
#         model = DDP(model, device_ids=[device_id])

#     # [QUAN TRỌNG] Tạo biến tham chiếu đến model gốc
#     # Dùng biến này để gọi load_state_dict hoặc các hàm custom như parse_batch
#     # Tránh lỗi "AttributeError: 'Tacotron2' object has no attribute 'module'" khi chạy đơn GPU
#     raw_model = model.module if hparams.ddp_run else model

#     # --- 3. Loss & Optimizer ---
#     criterion = Tacotron2Loss().to(device_id)
#     optimizer = AdamW(model.parameters(), lr=hparams.learning_rate, weight_decay=hparams.weight_decay)

#     print(f"[Rank {rank}] Starting training setup...")

#     # --- 4. Load Checkpoint ---
#     global_step = 0
#     epoch = 0
#     best_val_loss = float('inf')

#     path_to_checkpoint = os.path.join(hparams.checkpoint_path, hparams.name_file_checkpoint)
#     if os.path.exists(path_to_checkpoint):
#         # Map location để tránh load nhầm GPU
#         map_loc = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'
#         checkpoint = torch.load(path_to_checkpoint, map_location=map_loc)
        
#         # Dùng raw_model để load weights
#         raw_model.load_state_dict(checkpoint['model_state_dict']) #type: ignore
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
#         global_step = checkpoint['step']
#         best_val_loss = checkpoint.get('best_val_loss', float('inf'))
#         epoch = checkpoint.get('epoch', 0)
        
#         print(f"[Rank {rank}] Loaded checkpoint from {hparams.checkpoint_path} at step {global_step}.")

#     # --- 5. DataLoader ---
#     train_loader, val_set = get_trainloader_valset(
#         device_id, 
#         world_size, 
#         hparams,
#         hparams.seed + epoch + 1
#     )
#     train_data_iter = iter(train_loader)

#     # --- 6. Progress Bar (Chỉ hiện ở Rank 0) ---
#     progress_bar = None
#     step_training = hparams.max_step_training + global_step
#     if rank == 0:
#         progress_bar = tqdm(initial=global_step, total=step_training, desc="Training", unit="step", position=0)

#     # --- 7. Training Loop ---
#     model.train()
    
#     while global_step < step_training:
#         # Lấy batch dữ liệu tiếp theo
#         try:
#             batch = next(train_data_iter)
#         except StopIteration:
#             print(f"[Rank {rank}] Epoch {epoch} finished. Reinitializing iterator...")
#             epoch += 1
#             # Nếu cần shuffle lại sampler trong DDP, bạn nên set_epoch ở đây (nếu dùng DistributedSampler)
#             # if hparams.ddp_run and hasattr(train_loader.sampler, 'set_epoch'):
#             #     train_loader.sampler.set_epoch(epoch)
            
#             train_data_iter = iter(train_loader)
#             batch = next(train_data_iter)

#         # --- Forward & Backward ---
#         optimizer.zero_grad()
        
#         # Dùng raw_model để gọi hàm custom parse_batch
#         # (Hàm này thường xử lý việc chuyển dữ liệu sang GPU)
#         model_inputs, ground_truth = raw_model.parse_batch(batch) #type: ignore
        
#         # Forward pass: Dùng 'model' (có wrapper DDP) để đảm bảo đồng bộ gradient
#         model_outputs = model(model_inputs)
        
#         output_length = model_inputs[3]
#         loss, loss_mel, loss_mel_postnet, loss_gate = criterion(
#             model_outputs, ground_truth, output_length
#         )
        
#         loss.backward()
        
#         # (Optional) Gradient Clipping
#         # torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip_thresh)
        
#         optimizer.step()

#         global_step += 1

#         # --- Logging (Rank 0) ---
#         if rank == 0:
#             progress_bar.update(1) # type: ignore
#             progress_bar.set_postfix({ # type: ignore
#                 'Loss': f"{loss.item():.4f}",
#                 'Mel': f"{loss_mel.item():.4f}",
#                 'Post': f"{loss_mel_postnet.item():.4f}",
#                 'Gate': f"{loss_gate.item():.4f}"
#             })

#         # --- Validation & Save Checkpoint ---
#         if global_step % hparams.val_interval == 0:
#             # Chỉ chạy validation ở rank 0 để tránh trùng lặp và tắc nghẽn
#             if rank == 0 and val_set is not None:
#                 model.eval()
#                 total_val_loss = 0.0
                
#                 # Tắt gradient để tiết kiệm bộ nhớ
#                 with torch.no_grad():
#                     val_progress = tqdm(val_set, desc="Validation", unit="batch", leave=False, position=1)
                    
#                     for val_batch in val_progress:
#                         # Dùng raw_model để parse batch
#                         v_inputs, v_truth = raw_model.parse_batch(val_batch) #type: ignore
                        
#                         # Forward pass validation
#                         v_outputs = model(v_inputs)
                        
#                         v_out_len = v_inputs[3]
#                         val_loss, _, _, _ = criterion(v_outputs, v_truth, v_out_len)
                        
#                         total_val_loss += val_loss.item()
#                         val_progress.set_postfix({'Val Loss': f"{val_loss.item():.4f}"})
                    
#                     val_progress.close()
                
#                 # Tính trung bình loss
#                 avg_val_loss = total_val_loss / len(val_set)
#                 print(f"\n[Rank {rank}] Step {global_step} | Validation Loss: {avg_val_loss:.5f}")
                
#                 # Lưu checkpoint nếu tốt hơn
#                 if avg_val_loss < best_val_loss:
#                     best_val_loss = avg_val_loss
#                     save_name = f"checkpoint_step_{global_step}.pt"
#                     # Lưu ý: Truyền raw_model vào hàm save để không lưu wrapper DDP
#                     save_checkpoint_step(raw_model, optimizer, best_val_loss, epoch, global_step, save_name, hparams)
#                     print(f"Saved best model to {save_name}")

#                 # Quay lại chế độ train
#                 model.train()

#     # --- 8. Cleanup ---
#     if rank == 0:
#         progress_bar.close() # type: ignore
    
#     print(f"[Rank {rank}] Training complete.")
    
#     # Chỉ destroy nếu đã init
#     if hparams.ddp_run:
#         dist.destroy_process_group()


# def train_worker_by_step(rank, world_size, hparams: Hparams):
#     """
#     """
#     device_id = rank % torch.cuda.device_count()
#     if hparams.ddp_run:
#         init_distributed_training(device_id, world_size, hparams) 

#     # Load model bên trong hàm worker
#     model = Tacotron2(hparams)
#     model = model.to(device_id) 
#     model = DDP(model, device_ids=[device_id])

#     # Khởi tạo hàm loss và optimizer
#     criterion = Tacotron2Loss().to(device_id) 
#     optimizer = AdamW(model.parameters(), lr=hparams.learning_rate, weight_decay=hparams.weight_decay)

#     print(f"[Rank {rank}] Starting training...")

#     # Thiết lập biến đếm bước và best_val_loss
#     global_step = 0
#     epoch = 0
#     best_val_loss = float('inf')

#     # Load từ checkpoint nếu có
#     path_to_checkpoint = os.path.join(hparams.checkpoint_path, hparams.name_file_checkpoint)
#     if os.path.exists(path_to_checkpoint):
#         checkpoint = torch.load(path_to_checkpoint, map_location=f'cuda:{device_id}')
#         model.module.load_state_dict(checkpoint['model_state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         global_step = checkpoint['step']
#         best_val_loss = checkpoint.get('best_val_loss', float('inf'))
#         epoch = checkpoint.get('epoch', 0)
#         print(f"[Rank {rank}] Loaded checkpoint from {hparams.checkpoint_path} at step {global_step}.")

#     # Lấy DataLoader và Validation set
#     train_loader, val_set = get_trainloader_valset(
#         device_id, 
#         world_size, 
#         hparams,
#         hparams.seed + epoch + 1
#     )
#     # Tạo iterator từ DataLoader
#     train_data_iter = iter(train_loader) 

#     # Thiết lập thanh tiến trình nếu là rank 0
#     progress_bar = None
#     step_training = hparams.max_step_training + global_step
#     if rank == 0:
#         progress_bar = tqdm(initial=global_step, total=step_training, desc="Training", unit="step", position=0)

#     # Bắt đầu vòng lặp huấn luyện
#     model.train()
#     while global_step < step_training:
#         try:
#             batch = next(train_data_iter) # type: ignore
#         except StopIteration:
#             print("[Info] Reinitializing train data iterator... ")
#             train_data_iter = iter(train_loader)
#             batch = next(train_data_iter) # type: ignore
#             epoch += 1

#         model_inputs, ground_truth = model.module.parse_batch(batch, rank=device_id)
#         model_outputs = model(model_inputs)
#         output_length = model_inputs[3]  # output_lengths
#         loss, loss_mel, loss_mel_postnet, loss_gate = criterion(
#             model_outputs, ground_truth, output_length
#         )
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()

#         global_step += 1

#         if rank == 0:
#             progress_bar.update(1) # type: ignore
#             progress_bar.set_postfix({ # type: ignore
#                 'Loss': f"{loss.item():.4f}",
#                 'Mel': f"{loss_mel.item():.4f}",
#                 'Postnet': f"{loss_mel_postnet.item():.4f}",
#                 'Gate': f"{loss_gate.item():.4f}"
#             })

#         # Vào trạng thái đánh giá và lưu checkpoint theo interval
#         if global_step % hparams.val_interval == 0:
#             if rank == 0 and val_set is not None:
#                 model.eval()
#                 total_val_loss = 0.0
#                 with torch.no_grad():
#                     val_progress = tqdm(val_set, desc="Validation", unit="batch", leave=False, position=1)
#                     for batch in val_progress:
#                         model_inputs, ground_truth = model.module.parse_batch(batch, rank=device_id)
#                         model_outputs = model(model_inputs)
#                         output_lengths = model_inputs[3]
#                         val_loss, _, _, _ = criterion(
#                             model_outputs, ground_truth, output_lengths
#                         )
#                         total_val_loss += val_loss.item()
#                         val_progress.set_postfix({'Val Loss': f"{val_loss.item():.4f}"})
#                     val_progress.close()
#                 avg_val_loss = total_val_loss / len(val_set)
#                 print(f"\n[Rank {rank}] Step {global_step} Validation Loss: {avg_val_loss}")
#                 if avg_val_loss < best_val_loss:
#                     best_val_loss = avg_val_loss
#                     save_checkpoint_step(model, optimizer, best_val_loss, epoch, global_step, f"checkpoint_step_{global_step}.pt", hparams)
#                 model.train()
#     if rank == 0:
#         progress_bar.close()  # type: ignore
#     print(f"[Rank {rank}] Training complete.")
#     if hparams.ddp_run:
#         dist.destroy_process_group()
