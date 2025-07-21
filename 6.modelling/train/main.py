# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
from utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from .trainer import run_training
from utils.data_utils import get_loader
from model.lg2unetr import SwinUNETR
from model.unet.unet_model import UNet
from model.mednextv1.MedNextV1 import MedNeXt
from monai.networks.nets import SwinUNETR as oSwinUNETR
# from networks.UXNet_3D.network_backbone import UXNET
# from monai.networks.nets import UNETR
# from networks.MedNeXt.MedNextV1 import MedNeXt

from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import BasicUNet, BasicUNetPlusPlus
#from monai.networks.nets import SwinUNETR
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction
from thop import profile
from thop import clever_format

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline for BRATS Challenge")
parser.add_argument("--model", default="lg2unetr", type=str, help="model selection")
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
parser.add_argument("--logdir", default="test", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--fold", default=0, type=int, help="data fold")
parser.add_argument("--pretrained_model_name", default="model.pt", type=str, help="pretrained model name")
parser.add_argument("--data_dir", default="./dataset", type=str, help="dataset directory")
parser.add_argument("--json_list", default="./jsons/dataset-random.json", type=str, help="dataset json file")
parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")
parser.add_argument("--max_epochs", default=500, type=int, help="max number of training epochs")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")
parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--val_every", default=10, type=int, help="validation frequency")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization name")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--feature_size", default=24, type=int, help="feature size")
parser.add_argument("--in_channels", default=4, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=3, type=int, help="number of output channels")
parser.add_argument("--cache_dataset", action="store_true", help="use monai Dataset class")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=256, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=256, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=1, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
parser.add_argument("--resume_ckpt", action="store_true", help="resume training from pretrained checkpoint")
parser.add_argument("--smooth_dr", default=1, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=1, type=float, help="constant added to dice numerator to avoid zero")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument("--spatial_dims", default=2, type=int, help="spatial dimension of input data")
parser.add_argument(
    "--pretrained_dir",
    default="./pretrained_models/",
    type=str,
    help="pretrained checkpoint directory",
)
parser.add_argument("--squared_dice", action="store_true", help="use squared Dice")
parser.add_argument("--batch_dice", action="store_true", help="use batch option in diceloss calculation")

def main():
    args = parser.parse_args()
    print("Arguments:", args)
    args.amp = not args.noamp
    # args.logdir = "./runs/" + args.logdir
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print("Found total gpus", args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args)


def main_worker(gpu, args):
    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = False
    data_tag = args.json_list.split("/")[-1].split(".")[0] if args.json_list else "default"
    data_tag = data_tag.replace("dataset_split_", "")
    args.out_base = args.model + "-" + data_tag
    
    loader = get_loader(args)
    print(args.rank, " gpu", args.gpu)
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)
    inf_size = [args.roi_x, args.roi_y, args.roi_z]
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    pretrained_pth = os.path.join(pretrained_dir, model_name)

    if args.model == "unet0":
        model = UNet(args.in_channels, args.out_channels)
    elif args.model == "unet1":
        model = BasicUNet(spatial_dims=2, features=(64, 128, 256, 512, 1024, 64),
                          in_channels=args.in_channels, out_channels=args.out_channels)
    elif args.model == "unet1s":
        model = BasicUNet(spatial_dims=2, features=(32, 64, 128, 256, 512, 32),
                          in_channels=args.in_channels, out_channels=args.out_channels)
    elif args.model == "swinunetr":
        model = oSwinUNETR(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            use_checkpoint=args.use_checkpoint,
            spatial_dims=2)
    elif args.model == "lg2unetr":
        model = SwinUNETR(
            img_size=(args.roi_x, args.roi_y),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            use_checkpoint=args.use_checkpoint,
            spatial_dims=2
        )
    elif args.model == "unetpp0":
        model = BasicUNetPlusPlus(
            spatial_dims=2, features=(32, 64, 128, 256, 512, 32),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
        )
    elif args.model == "unetpp0D":
        model = BasicUNetPlusPlus(
            spatial_dims=2, features=(32, 64, 128, 256, 512, 32),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            deep_supervision=True
        )

    # parser.add_argument("--in_channels", default=4, type=int, help="number of input channels")

    
    # elif args.model == "mednext0":
        # model = MedNeXt(
        #     in_channels = args.in_channels, 
        #     n_channels = 32,
        #     n_classes = args.out_channels, 
        #     exp_r=2,                         
        #     kernel_size=kernel_size,         
        #     deep_supervision=ds,             
        #     do_res=True,                     
        #     do_res_up_down = True,
        #     block_counts = [2,2,2,2,2,2,2,2,2],
        #     dim='2d'
        # )
    else:
        raise ValueError("Unsupported Model: " + str(args.model))

    '''model = UXNET(
        in_chans=args.in_channels,
        out_chans=args.out_channels,
        depths=[2, 4, 2, 2],
        feat_size=[args.feature_size, args.feature_size*2, args.feature_size*4, args.feature_size*8],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        spatial_dims=3,
    )
    model = UNETR(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        feature_size=args.feature_size,
        hidden_size=960,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
    )
    model = MedNeXt(
        in_channels = args.in_channels, 
        n_channels = 60,
        n_classes = args.out_channels, 
        exp_r = [3,4,4,4,4,4,4,4,3],       
        kernel_size = 3,         
        deep_supervision = False,             
        do_res = True,                     
        do_res_up_down = True,
        block_counts = [3,4,4,4,4,4,4,4,3],
        dim = '3d'
    )'''
    

    # print(model)
    from torchinfo import summary
    if data_tag.endswith('lgg'):
        summary(model, input_size=(args.batch_size, 3, 256, 256))
    else:
        # summary(model, input_size=(args.batch_size, 1,160, 192))
        summary(model, input_size=(args.batch_size, args.in_channels, args.roi_x , args.roi_y))
    if args.resume_ckpt:
        model_dict = torch.load(pretrained_pth, weights_only=False)["state_dict"]
        model.load_state_dict(model_dict, False)
        print("Using pretrained weights")

    if args.squared_dice:
        dice_loss = DiceLoss(
            to_onehot_y=False, sigmoid=True, squared_pred=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr
        )
    else:
        dice_loss = DiceLoss(to_onehot_y=False, sigmoid=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr,
                             batch=args.batch_dice)
    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(argmax=False, threshold=0.5)
    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True, ignore_empty=False)
    if (args.spatial_dims == 3):
        model_inferer = partial(
            sliding_window_inference,
            roi_size=inf_size,
            sw_batch_size=args.sw_batch_size,
            predictor=model,
            overlap=args.infer_overlap,
        )
    else:
        model_inferer = model
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)
    # if (args.spatial_dims == 3):
    #     ran = torch.randn(1, 4, 128, 128, 128)
    # else:
    #     if data_tag.endswith('lgg'):
    #         ran = torch.randn(1, 3, 256, 256)
    #     else:
    #         ran = torch.randn(1, 1, 160, 192)
    # ran = ran.cuda(args.gpu)
    # flops, params = profile(model, inputs=(ran,))
    # print("Thop Results", flops, params)
    # flops, params = clever_format([flops, params], "%.1f")
    # print("Thop Results", flops, params)

    best_acc = 0
    start_epoch = 0

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k.replace("backbone.", "")] = v
        model.load_state_dict(new_state_dict, strict=False)
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))

    model.cuda(args.gpu)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)
    else:
        scheduler = None

    semantic_classes = ["Dice_Val_TC", "Dice_Val_WT", "Dice_Val_ET"]

    print (loader[0].dataset.transform.transforms)
    accuracy = run_training(
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        loss_func=dice_loss,
        acc_func=dice_acc,
        args=args,
        model_inferer=model_inferer,
        scheduler=scheduler,
        start_epoch=start_epoch,
        post_sigmoid=post_sigmoid,
        post_pred=post_pred,
        semantic_classes=semantic_classes,
    )
    return accuracy


if __name__ == "__main__":
    main()