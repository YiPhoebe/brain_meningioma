

import os

class CFG:
    # 데이터 경로
    train_dir = "/Users/iujeong/0.local/4.slice/s_train/npy"
    val_dir = "/Users/iujeong/0.local/4.slice/s_val/npy"
    test_dir = "/Users/iujeong/0.local/4.slice/s_test/npy"

    # 모델 저장 경로
    save_dir = "/Users/iujeong/0.local/6.modelling/checkpoints"

    # 학습 설정
    num_epochs = 50
    batch_size = 8
    lr = 1e-4

    # 기타 설정
    seed = 42
    num_workers = 4
    image_size = (160, 192)