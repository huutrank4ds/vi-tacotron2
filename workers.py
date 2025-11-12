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

def save_checkpoint_step(model, optimizer, best_val_loss, step, filepath, hparams: Hparams):
    model_state_dict = model.module.state_dict()
    checkpoint_dict = {
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'best_val_loss': best_val_loss
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

def train_worker_by_step(rank, world_size, hparams: Hparams):
    """
    """
    device_id = rank % torch.cuda.device_count()
    if hparams.ddp_run:
        init_distributed_training(device_id, world_size, hparams)

    # Lấy DataLoader và Validation set
    train_loader, val_set = get_trainloader_valset(
        device_id, 
        world_size, 
        hparams
    )
    # Tạo iterator từ DataLoader
    train_data_iter = iter(train_loader)  

    # Load model bên trong hàm worker
    model = Tacotron2(hparams)
    model = model.to(device_id) 
    model = DDP(model, device_ids=[device_id])

    # Khởi tạo hàm loss và optimizer
    criterion = Tacotron2Loss().to(device_id) 
    optimizer = AdamW(model.parameters(), lr=hparams.learning_rate, weight_decay=hparams.weight_decay)

    print(f"[Rank {rank}] Starting training...")

    # Thiết lập biến đếm bước và best_val_loss
    global_step = 0
    best_val_loss = float('inf')

    # Load từ checkpoint nếu có
    path_to_checkpoint = os.path.join(hparams.checkpoint_path, hparams.name_file_checkpoint)
    if os.path.exists(path_to_checkpoint):
        checkpoint = torch.load(path_to_checkpoint, map_location=f'cuda:{device_id}')
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        global_step = checkpoint['step']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"[Rank {rank}] Loaded checkpoint from {hparams.checkpoint_path} at step {global_step}.")

    # Thiết lập thanh tiến trình nếu là rank 0
    progress_bar = None
    if rank == 0:
        progress_bar = tqdm(initial=global_step, total=hparams.max_step_training, desc="Training", unit="step", position=0)

    # Bắt đầu vòng lặp huấn luyện
    model.train()
    while global_step < hparams.max_step_training:
        try:
            batch = next(train_data_iter) # type: ignore
        except StopIteration:
            print("[Info] Reinitializing train data iterator... ")
            train_data_iter = iter(train_loader)
            batch = next(train_data_iter) # type: ignore

        model_inputs, ground_truth = model.module.parse_batch(batch, rank=device_id)
        model_outputs = model(model_inputs)
        output_length = model_inputs[3]  # output_lengths
        loss, loss_mel, loss_mel_postnet, loss_gate = criterion(
            model_outputs, ground_truth, output_length
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        global_step += 1
        if rank == 0:
            progress_bar.update(1) # type: ignore
            progress_bar.set_postfix({ # type: ignore
                'Loss': f"{loss.item():.4f}",
                'Mel': f"{loss_mel.item():.4f}",
                'Postnet': f"{loss_mel_postnet.item():.4f}",
                'Gate': f"{loss_gate.item():.4f}"
            })

        # Vào trạng thái đánh giá và lưu checkpoint theo interval
        if global_step % hparams.val_interval == 0:
            if rank == 0 and val_set is not None:
                model.eval()
                total_val_loss = 0.0
                with torch.no_grad():
                    val_progress = tqdm(val_set, desc="Validation", unit="batch", leave=False, position=1)
                    for batch in val_progress:
                        model_inputs, ground_truth = model.module.parse_batch(batch, rank=device_id)
                        model_outputs = model(model_inputs)
                        output_lengths = model_inputs[3]
                        val_loss, _, _, _ = criterion(
                            model_outputs, ground_truth, output_lengths
                        )
                        total_val_loss += val_loss.item()
                        val_progress.set_postfix({'Val Loss': f"{val_loss.item():.4f}"})
                    val_progress.close()
                avg_val_loss = total_val_loss / len(val_set)
                print(f"\n[Rank {rank}] Step {global_step} Validation Loss: {avg_val_loss}")
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    save_checkpoint_step(model, optimizer, best_val_loss, global_step, f"checkpoint_step_{global_step}.pt", hparams)
                model.train()
    if rank == 0:
        progress_bar.close()  # type: ignore
    print(f"[Rank {rank}] Training complete.")
    if hparams.ddp_run:
        dist.destroy_process_group()
