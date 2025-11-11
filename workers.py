import torch
from config import Hparams
from torch.optim import AdamW
import torch.distributed as dist
from dataloader import get_trainloader_valset
from model import Tacotron2
from loss import Tacotron2Loss
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import os


def init_distributed_training(rank, world_size, hparams: Hparams):
    assert torch.cuda.is_available(), "CUDA không khả dụng cho DDP."
    print("Init DDP...")
    # Thiết lập GPU cho mỗi tiến trình
    torch.cuda.set_device(rank)

    dist.init_process_group(
        backend=hparams.ddp_backend,
        init_method=hparams.ddp_url,
        world_size=world_size,
        rank=rank
    )
    print(f"[Rank {rank}] DDP đã được khởi tạo trên GPU {rank % torch.cuda.device_count()}.")

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
    print(f"Đã lưu checkpoint tại {os.path.join(hparams.checkpoint_path, filepath)}")

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
        # Giả sử get_trainloader_valset trả về loader dùng TextMelCollate
    train_loader, val_set = get_trainloader_valset(
        device_id, 
        world_size, 
        hparams
    )
    train_data_iter = iter(train_loader)  # Tạo iterator từ DataLoader
    # Load model bên trong hàm worker
    model = Tacotron2(hparams)
    model = model.to(device_id)
    model = DDP(model, device_ids=[device_id])

    # --- 2. KHỞI TẠO HÀM LOSS VÀ OPTIMIZER ---
    criterion = Tacotron2Loss().to(device_id) 
    optimizer = AdamW(model.parameters(), lr=hparams.learning_rate, weight_decay=hparams.weight_decay)

    print(f"[Rank {rank}] Bắt đầu huấn luyện...")
    global_step = 0
    best_val_loss = float('inf')

    # Load từ checkpoint nếu có
    if os.path.exists(hparams.checkpoint_path):
        checkpoint = torch.load(hparams.checkpoint_path, map_location=f'cuda:{device_id}')
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        global_step = checkpoint['step']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"[Rank {rank}] Đã tải checkpoint từ {hparams.checkpoint_path} tại step {global_step}.")

    progress_bar = None
    if rank == 0:
        progress_bar = tqdm(initial=global_step, total=hparams.max_step_training, desc="Training", unit="step")

    model.train()
    while global_step < hparams.max_step_training:
        try:
            batch = next(train_data_iter) # type: ignore
        except StopIteration:
            train_data_iter = iter(train_loader)
            batch = next(train_data_iter) # type: ignore

        optimizer.zero_grad()
        model_inputs, ground_truth = model.module.parse_batch(batch, rank=device_id)
        model_outputs = model(model_inputs)
        output_length = model_inputs[3]  # output_lengths
        loss, loss_mel, loss_mel_postnet, loss_gate = criterion(
            model_outputs, ground_truth, output_length
        )
        loss.backward()
        optimizer.step()
        global_step += 1
        if rank == 0:
            progress_bar.update(1) # type: ignore
            progress_bar.set_postfix({ # type: ignore
                'Loss': f"{loss.item():.4f}",
                'Mel': f"{loss_mel.item():.4f}",
                'Postnet': f"{loss_mel_postnet.item():.4f}",
                'Gate': f"{loss_gate.item():.4f}"
            })

        if global_step % hparams.val_interval == 0 and device_id == 0 and val_set is not None:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch in val_set:
                    model_inputs, ground_truth = model.module.parse_batch(batch)
                    model_outputs = model(model_inputs)
                    output_lengths = model_inputs[3]
                    val_loss, _, _, _ = criterion(
                        model_outputs, ground_truth, output_lengths
                    )
                    total_val_loss += val_loss.item()
            avg_val_loss = total_val_loss / len(val_set)
            print(f"\n[Rank {rank}] Step {global_step} Validation Loss: {avg_val_loss}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_checkpoint_step(model, optimizer, best_val_loss, global_step, f"checkpoint_step_{global_step}.pt", hparams)
            model.train()
        if hparams.ddp_run:
            dist.barrier()  # Đồng bộ hóa các tiến trình sau mỗi validation
    if rank == 0:
        progress_bar.close()  # type: ignore
    print(f"[Rank {rank}] Huấn luyện hoàn tất.")
    if hparams.ddp_run:
        dist.destroy_process_group()
