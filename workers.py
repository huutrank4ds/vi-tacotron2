import torch
import os
from config import Hparams
from torch.optim import AdamW
import torch.distributed as dist
from dataloader import get_trainloader_valset
from model import Tacotron2
from loss import Tacotron2Loss
from torch.nn.parallel import DistributedDataParallel as DDP


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

def save_checkpoint(model, optimizer, epoch, step, filepath):
    checkpoint_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step
    }
    torch.save(checkpoint_dict, filepath)
    print(f"Đã lưu checkpoint tại {filepath}")

def train_worker(rank, world_size, hparams: Hparams):
    # --- 1. KHỞI TẠO DDP VÀ DATALOADER ---
    device_id = rank % torch.cuda.device_count()
    if hparams.ddp_run:
        init_distributed_training(device_id, world_size, hparams)

    # Giả sử get_trainloader_valset trả về loader dùng TextMelCollate
    train_loader, val_set = get_trainloader_valset(
        device_id, 
        world_size, 
        hparams.batch_size
    )

    # Load model bên trong hàm worker
    model = Tacotron2(hparams)
    model = model.to(device_id)
    model = DDP(model, device_ids=[device_id])

    # --- 2. KHỞI TẠO HÀM LOSS VÀ OPTIMIZER ---
    criterion = Tacotron2Loss().to(device_id) 
    optimizer = AdamW(model.parameters(), lr=hparams.learning_rate, weight_decay=hparams.weight_decay)
    
    print(f"[Rank {rank}] Bắt đầu huấn luyện...")
    model.train()
    for epoch in range(hparams.epochs):
        for step, batch in enumerate(train_loader):
            
            optimizer.zero_grad()

            # Dùng parse_batch của model để chuẩn bị dữ liệu
            # (Hàm này đã tự chuyển tensor sang GPU)
            # Lưu ý: dùng model.module để truy cập hàm của model gốc
            model_inputs, ground_truth = model.module.parse_batch(batch)
            
            # Forward pass
            # model_inputs là tuple: (text_padded, input_lengths, ...)
            model_outputs = model(model_inputs)
            
            # Tính loss bằng criterion (Hàm loss riêng)
            # model_outputs là list: [mel_out, mel_post_out, gate_out, ...]
            # ground_truth là tuple: (mel_padded, gate_padded)
            # Chúng ta cũng cần output_lengths để masking
            output_lengths = model_inputs[4] # Lấy output_lengths từ inputs

            loss, loss_mel, loss_mel_postnet, loss_gate = criterion(
                model_outputs, ground_truth, output_lengths
            )
            
            # Backward và Optimize
            loss.backward()
            optimizer.step()
            
            if rank == 0 and step % 50 == 0:
                print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}")
                print(f"    Mel: {loss_mel.item()}, Postnet: {loss_mel_postnet.item()}, Gate: {loss_gate.item()}")
            
            if step == 200:
                print(f"[Rank {rank}] Đã hoàn thành 200 bước.")
                break
        if device_id == 0:
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
                    total_val_loss += val_loss
            avg_val_loss = total_val_loss / len(val_set)
            print(f"[Rank {rank}] Epoch {epoch} Validation Loss: {avg_val_loss.item()}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_checkpoint(model, optimizer, epoch, step, f"checkpoint_epoch_{epoch}.pt")
        if hparams.ddp_run:
            dist.barrier()  # Đồng bộ hóa các tiến trình sau mỗi epoch

    print(f"[Rank {rank}] Huấn luyện hoàn tất.")
    dist.destroy_process_group()