# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND  , either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
# import pdb
import shutil
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather

from monai.data import decollate_batch


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, acc_func, args,
                post_sigmoid=None, post_pred=None):
    model.train()
    print_freq = len(loader) // 10
    start_time = time.time()
    run_loss = AverageMeter()
    run_acc = AverageMeter()
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
        data, target = data.cuda(args.rank), target.cuda(args.rank)
        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            logits = model(data)
            if not isinstance(logits, list):
                loss = loss_func(logits, target)
            elif len(logits) ==1 :
                loss = loss_func(logits[0], target)
            elif len(logits) == 4:
                deep_supervision_weights = [0.1, 0.2, 0.3, 0.4]
                loss = 0
                for out, w in zip(logits, deep_supervision_weights):
                    lossi = loss_func(out, target)
                    loss += w * lossi
            elif len(logits) == 5:
                deep_supervision_weights = [0.05, 0.1, 0.15, 0.2, 0.5]
                loss = 0
                for out, w in zip(logits, deep_supervision_weights):
                    resized_target = torch.nn.functional.interpolate(
                        target, size=out.shape[2:], mode="nearest")
                    lossi = loss_func(out, resized_target)
                    loss += w * lossi
            else:
                raise ValueError("Unsupported logits length: " + str(len(logits)))

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)
            
        #if args.rank == 0:
        val_labels_list = decollate_batch(target)
        val_outputs_list = decollate_batch(logits if not isinstance(logits, list) else logits[-1])
        val_output_convert = [post_pred(post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]
        # val_output_convert = [post_pred(post_sigmoid(val_pred_tensor)) for val_pred_tensor in logits]
        acc_func.reset()
        acc_func(y_pred=val_output_convert, y=val_labels_list)
        # acc_func(y_pred=post_pred(post_sigmoid(logits)), y=target)
        acc, not_nans = acc_func.aggregate()
        run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
        
        if (idx + 1) % print_freq == 0 or idx == len(loader) - 1:
            print(
                "Epoch {}/{} {}/{}".format(epoch + 1, args.max_epochs, idx + 1, len(loader)),
                "loss: {:.4f}".format(run_loss.avg), f"acc: {np.mean(run_acc.avg):.5f}",
                "time {:.2f}s".format(time.time() - start_time),
            )
            start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg, run_acc.avg


def val_epoch(model, loader, epoch, acc_func, loss_func,
              args, model_inferer=None, post_sigmoid=None, post_pred=None):
    model.eval()
    print_freq = len(loader) // 10
    start_time = time.time()
    run_acc = AverageMeter()
    run_loss = AverageMeter()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            with autocast(enabled=args.amp):
                logits = model_inferer(data)
            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits if not isinstance(logits, list) else logits[-1])
            val_output_convert = [post_pred(post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)
            loss = loss_func(logits if not isinstance(logits, list) else logits[-1], target)
            acc, not_nans = acc_func.aggregate()
            acc = acc.cuda(args.rank)
            if args.distributed:
                pass
                # acc_list, not_nans_list = distributed_all_gather(
                #     [acc, not_nans], out_numpy=True, is_valid=idx < loader.sampler.valid_length
                # )
                # for al, nl in zip(acc_list, not_nans_list):
                #     run_acc.update(al, n=nl)
            else:
                run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
                run_loss.update(loss.item(), n=args.batch_size)

            #if args.rank == 0:
            if (idx +1) % print_freq == 0 or idx == len(loader) - 1:
                if args.out_channels == 3:
                    Dice_TC = run_acc.avg[0]
                    Dice_WT = run_acc.avg[1]
                    Dice_ET = run_acc.avg[2]
                    print(
                        "Val {}/{} {}/{}".format(epoch + 1, args.max_epochs, idx + 1, len(loader)),
                        ", Dice_TC:",
                        Dice_TC,
                        ", Dice_WT:",
                        Dice_WT,
                        ", Dice_ET:",
                        Dice_ET,
                        ", time {:.2f}s".format(time.time() - start_time),
                    )
                elif args.out_channels == 1:
                    Dice_WT = run_acc.avg[0]
                    print(
                        "Val {}/{} {}/{}".format(epoch + 1, args.max_epochs, idx + 1, len(loader)),
                        ", Dice:",
                        Dice_WT, f"Loss b{args.batch_size}:", run_loss.avg,
                        f", Logits: {0 if not isinstance(logits,list) else len(logits)}"
                        ", time {:.2f}s".format(time.time() - start_time),
                    )
                start_time = time.time()

    return run_acc.avg, run_loss.avg


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    post_sigmoid=None,
    post_pred=None,
    semantic_classes=None,
):
    writer = None
    #if args.logdir is not None and args.rank == 0:
    print("Len datasets: ", f"train: {len(train_loader)}, val: {len(val_loader)}" )
    if args.logdir is not None:
        writer = SummaryWriter(log_dir=args.logdir)
        #if args.rank == 0:
        print("Writing Tensorboard logs to ", args.logdir)
    # train_loader.dataset[0]
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.0
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch + 1)
        epoch_time = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, acc_func= acc_func,
            args=args, post_sigmoid=post_sigmoid,  post_pred=post_pred,
        )
        #if args.rank == 0:
        print(
            "Final training  {}/{}".format(epoch + 1, args.max_epochs),
            "loss: {:.4f}".format(train_loss), f"acc: {np.mean(train_acc):.5f}",
            "time {:.2f}s".format(time.time() - epoch_time),
        )
        for name, param in model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"NaN or Inf in {name} at epoch {epoch}")
        #if args.rank == 0 and writer is not None:
        if writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
            writer.add_scalar("train_acc", np.mean(train_acc), epoch)
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_acc, val_loss = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                loss_func=loss_func,
                model_inferer=model_inferer,
                args=args,
                post_sigmoid=post_sigmoid,
                post_pred=post_pred,
            )

            #if args.rank == 0:
            if args.out_channels == 3:
                dice_values = f'DIce_TC: {val_acc[0]:.4f}, Dice_WT: {val_acc[1]:.4f}, Dice_ET: {val_acc[2]:.4f}'
            else:
                dice_values = f'Dice: {val_acc[0]}'
            print(
                    "Final validation stats {}/{}".format(epoch + 1, args.max_epochs),
                    ",", dice_values, f'Loss: {val_loss}',
                    ", time {:.2f}s".format(time.time() - epoch_time),
            )
            
            if writer is not None:
                writer.add_scalar("Mean_Val_Dice", np.mean(val_acc), epoch)
                writer.add_scalar("Val_Loss", val_loss, epoch)
                if semantic_classes is not None:
                    for val_channel_ind in range(len(semantic_classes)):
                        if val_channel_ind < val_acc.size:
                            writer.add_scalar(semantic_classes[val_channel_ind], val_acc[val_channel_ind], epoch)
            val_avg_acc = np.mean(val_acc)
            if val_avg_acc > val_acc_max:
                print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                val_acc_max = val_avg_acc
                b_new_best = True
                #if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                if args.logdir is not None and args.save_checkpoint:
                    save_checkpoint(
                        model, epoch, args, best_acc=val_acc_max, filename=args.out_base+".pt", optimizer=optimizer, scheduler=scheduler
                    )
            #if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
            if args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename=args.out_base+"_final.pt")
                if b_new_best:
                    print("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, args.out_base+"_final.pt"), os.path.join(args.logdir, args.model+".pt"))

        if scheduler is not None:
            scheduler.step()

    print("Training Finished !, Best Accuracy: ", val_acc_max)

    return val_acc_max