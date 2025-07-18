

import os
import torch

class CFG:

    device = torch.device("mps" if torch.mps.is_available() else "cpu")

    # 데이터 경로
    train_dir = "/Users/iujeong/0.local/4.slice/s_train/npy"
    val_dir = "/Users/iujeong/0.local/4.slice/s_val/npy"
    test_dir = "/Users/iujeong/0.local/4.slice/s_test/npy"

    # 모델 및 로그 저장 경로
    save_root = "/Users/iujeong/0.local/6.modelling/result"
    save_pth_dir = os.path.join(save_root, "pth")
    save_log_dir = os.path.join(save_root, "logs")
    save_csv_dir = os.path.join(save_root, "csv")

    # 학습 설정
    num_epochs = 50
    batch_size = 8
    lr = 3e-4
    # pin_memory = True

    # 기타 설정
    seed = 42
    num_workers = 0
    image_size = (160, 192)
    img_height = 160
    img_width = 192
    use_aug = False     #True    # 데이터 증강 여부

    # 모델 관련
    model_name = "UNet_3"  # 또는 "nnUNet"

    # 최적화 관련
    optimizer = "Adam"
    scheduler = "None"  # 또는 "StepLR"
    loss_name = "BCEDice"

    # 모델 저장 기준
    save_best_metric = "dice"  # 또는 "val_loss"