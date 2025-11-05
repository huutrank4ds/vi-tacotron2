import torch
import os
from config import Hparams
from torch.optim import AdamW
import torch.distributed as dist
from create_dataloader import create_dataloader 
from model import Tacotron2
from loss import Tacotron2Loss
from torch.nn.parallel import DistributedDataParallel as DDP

def train_worker(rank, world_size, hparams: Hparams):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(
        backend='nccl', 
        init_method='env://', 
        world_size=world_size, 
        rank=rank
    )
    torch.cuda.set_device(rank)
    local_rank = rank

    print(f"[Rank {rank}] Đã thiết lập DDP trên GPU {local_rank}.")
    
    # Giả sử create_dataloader trả về loader dùng TextMelCollate
    train_loader = create_dataloader(
        rank, 
        world_size, 
        hparams.batch_size, 
        hparams.num_workers
    )

    # Load model bên trong hàm worker
    model = Tacotron2(hparams)
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # --- 2. KHỞI TẠO HÀM LOSS VÀ OPTIMIZER ---
    # (Giả sử file loss.py của bạn chứa class Tacotron2Loss)
    criterion = Tacotron2Loss().to(local_rank) 
    optimizer = AdamW(model.parameters(), lr=hparams.learning_rate)
    
    print(f"[Rank {rank}] Bắt đầu huấn luyện...")
    model.train()
    for epoch in range(1):
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

    print(f"[Rank {rank}] Huấn luyện hoàn tất.")
    dist.destroy_process_group()