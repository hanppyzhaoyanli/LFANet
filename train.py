import argparse
import logging
import os
import os.path as osp
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from unet.compair_net import mobileUNet, Mobilev2UNet, AttU_Net, R2U_Net, R2AttU_Net, SegNet
from utils.metrics import *
from unet import NestedUNet
from unet import *
from utils.dataset import BasicDataset
from config import UNetConfig
from losses import LovaszLossSoftmax
from losses import LovaszLossHinge
from losses import dice_coeff
from thop import profile
from unet.ghostunet import *
from unet.gan import *
import time
import math
cfg = UNetConfig()


def train_net(net, cfg):

    dataset = BasicDataset(cfg.images_dir, cfg.masks_dir, cfg.scale)
    train = BasicDataset(cfg.images_dir, cfg.masks_dir, cfg.scale)
    val = BasicDataset(cfg.val_img_dir, cfg.val_lab_dir, cfg.scale)
    n_train = int(len(train))
    n_val = int(len(val))
 
    begin = time.time()
    train_loader = DataLoader(train,
                              batch_size=cfg.batch_size,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True)
    val_loader = DataLoader(val,
                            batch_size=cfg.batch_size,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True)

    writer = SummaryWriter(comment=f'_Model_{cfg.model}_LR_{cfg.lr}_BS_{cfg.batch_size}_Opt_{cfg.optimizer}')
    global_step = 0

    logging.info(f'''Starting training:
        Model type:      {cfg.model}
        Epochs:          {cfg.epochs}
        Batch size:      {cfg.batch_size}
        Learning rate:   {cfg.lr}
        Optimizer:       {cfg.optimizer}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {cfg.save_cp}
        Device:          {device.type}
        Images scaling:  {cfg.scale}
    ''')

    if cfg.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(),
                               lr=cfg.lr)
        

    elif cfg.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(net.parameters(),
                                  lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    else:
        optimizer = optim.SGD(net.parameters(),
                              lr=cfg.lr,
                              momentum=cfg.momentum,
                              weight_decay=cfg.weight_decay,
                              nesterov=cfg.nesterov)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=cfg.lr_decay_milestones,
                                               gamma = cfg.lr_decay_gamma)
    # # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if cfg.n_classes > 1 else 'max',
    #                                                  verbose=True, patience=2)
    if cfg.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
        # criterion = LovaszLossSoftmax()

    else:
        criterion = LovaszLossHinge()

    # net.load_state_dict(torch.load('./checkpoints/gland/xx/epoch_50.pth'))

    for epoch in range(cfg.epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{cfg.epochs}', unit='img') as pbar:
            for batch in train_loader:
                batch_imgs = batch['image']
                batch_masks = batch['mask']
                assert batch_imgs.shape[1] == cfg.n_channels, \
                        f'Network has been defined with {cfg.n_channels} input channels, ' \
                        f'but loaded images have {batch_imgs.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'

                batch_imgs = batch_imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if cfg.n_classes == 1 else torch.long
                batch_masks = batch_masks.to(device=device, dtype=mask_type)

                inference_masks = net(batch_imgs)

                if cfg.n_classes == 1:
                    inferences = inference_masks.squeeze(1)
                    masks = batch_masks.squeeze(1)
                else:
                    inferences = inference_masks
                    masks = batch_masks

                if cfg.deepsupervision:
                    loss = 0
                    for inference_mask in inferences:
                        loss += criterion(inference_mask, masks)
                    loss /= len(inferences)
                else:
                    ''' crossentropyloss'''
                    loss = criterion(inferences, masks.squeeze(1))
                    ''' lovazmaxloss'''
                    # loss = criterion(inferences, masks)

                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)
                writer.add_scalar('model/lr', optimizer.param_groups[0]['lr'], global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                scheduler.step()
                pbar.update(batch_imgs.shape[0])
                global_step += 1

        end = time.time()
        print(f'\n{cfg.model} training time/sum:',format(end-begin,'.4f')+'s')
        num_pic=math.floor(n_train/cfg.batch_size)*cfg.batch_size*(epoch+1)
        print(f'{cfg.model} training time/pic:', format((end-begin)/num_pic,'.4f')+'s')

        # if global_step % (len(dataset) // (0.5 * cfg.batch_size)) == 0:
        cytoplasm_dice, nucleus_dice, val_score,cytoplasm_jac, nuleus_jac,\
        cytoplasm_acc, nuleus_acc,cytoplasm_sen,nuleus_sen,\
        cytoplasm_spen,nuleus_spen,cytoplasm_pre,nuleus_pre,\
        cytoplasm_rec,nuleus_rec,cytoplasm_tve, nuleus_tve = eval_net(net, val_loader, device, n_val, cfg)
        print(
'''cytoplasm_dice:{:.4f},nucleus_dice:{:.4f},
val_score:{:.4f},
cytoplasm_jac:{:.4f}, nuleus_jac:{:.4f},
cytoplasm_acc:{:.4f},nucleus_acc:{:.4f},
cytoplasm_sen:{:.4f}, nuleus_sen:{:.4f},
cytoplasm_spen:{:.4f},nucleus_spen:{:.4f},
cytoplasm_pre:{:.4f}, nuleus_pre:{:.4f},
cytoplasm_rec:{:.4f},nucleus_rec:{:.4f}
cytoplasm_tve:{:.4f},nucleus_tve:{:.4f}'''
.format(cytoplasm_dice, nucleus_dice, val_score,cytoplasm_jac, nuleus_jac,
cytoplasm_acc, nuleus_acc,cytoplasm_sen,nuleus_sen,
cytoplasm_spen,nuleus_spen,cytoplasm_pre,nuleus_pre,
cytoplasm_rec,nuleus_rec,cytoplasm_tve,nuleus_tve))
        # scheduler.step(nucleus_dice)
        if cfg.n_classes > 1:
            logging.info('Validation cross entropy: {:.4f}'.format(val_score))
            writer.add_scalar('cytoplasm_dice/test', cytoplasm_dice, global_step)
            writer.add_scalar('nucleus_dice/test', nucleus_dice, global_step)
            writer.add_scalar('cytoplasm_jac/test', cytoplasm_jac, global_step)
            writer.add_scalar('nucleus_jac/test', nuleus_jac, global_step)
            writer.add_scalar('cytoplasm_acc/test', cytoplasm_acc, global_step)
            writer.add_scalar('nuleus_acc/test', nuleus_acc, global_step)
            writer.add_scalar('nuleus_sen/test', nuleus_sen, global_step)
            writer.add_scalar('cytoplasm_sen/test', cytoplasm_sen, global_step)
            writer.add_scalar('cytoplasm_spen/test', cytoplasm_spen, global_step)
            writer.add_scalar('nuleus_spen/test', nuleus_spen, global_step)
            writer.add_scalar('cytoplasm_pre/test', cytoplasm_pre, global_step)
            writer.add_scalar('nuleus_pre/test', nuleus_pre, global_step)
            writer.add_scalar('cytoplasm_rec/test', cytoplasm_rec, global_step)
            writer.add_scalar('nuleus_rec/test', nuleus_rec, global_step)
            writer.add_scalar('cytoplasm_tve/test', cytoplasm_tve, global_step)
            writer.add_scalar('nuleus_tve/test', nuleus_tve, global_step)

        else:
            logging.info('Validation Dice Coeff: {}'.format(val_score))
            writer.add_scalar('Dice/test', val_score, global_step)

        writer.add_images('images', batch_imgs, global_step)
        if cfg.deepsupervision:
            inference_masks = inference_masks[-1]
        if cfg.n_classes == 1:
            writer.add_images('masks/true', batch_masks, global_step)
            inference_mask = torch.sigmoid(inference_masks) > cfg.out_threshold
            writer.add_images('masks/inference',
                              inference_mask,
                              global_step)
        else:
            # writer.add_images('masks/true', batch_masks, global_step)
            ids = inference_masks.shape[1]  # N x C x H x W
            inference_masks = torch.chunk(inference_masks, ids, dim=1)
            for idx in range(0, len(inference_masks)):
                inference_mask = torch.sigmoid(inference_masks[idx]) > cfg.out_threshold
                writer.add_images('masks/inference_'+str(idx),
                                  inference_mask,
                                  global_step)

        if cfg.save_cp:
            try:
                os.mkdir(cfg.checkpoints_dir)
                logging.info('Created checkpoint directory')
            except OSError:
                pass

            ckpt_name = 'epoch_' + str(epoch + 1) + '.pth'
            torch.save(net.state_dict(),
                       osp.join(cfg.checkpoints_dir, ckpt_name))
            logging.info(f'Checkpoint {epoch + 1} saved !')
    writer.close()

def eval_net(net, loader, device, n_val, cfg):
    """
    Evaluation without the densecrf with the dice coefficient
    """
    net.eval()
    tot = 0
    cytoplasm_dice=0
    nuleus_dice=0
    cytoplasm_jac=0
    nuleus_jac=0
    cytoplasm_acc=0
    nuleus_acc=0
    cytoplasm_rec=0
    nuleus_rec=0
    cytoplasm_spen=0
    nuleus_spen=0
    cytoplasm_sen=0
    nuleus_sen=0
    cytoplasm_pre=0
    nuleus_pre=0
    cytoplasm_tve=0
    nuleus_tve=0

    with tqdm(total=n_val, desc='Validation round', unit='img', leave=False) as pbar:
        for batch in loader:
            imgs = batch['image']
            true_masks = batch['mask']

            imgs = imgs.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if cfg.n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)

            if cfg.deepsupervision:
                masks_preds = net(imgs)
                loss = 0
                for masks_pred in masks_preds:
                    tot_cross_entropy = 0
                    for true_mask, pred in zip(true_masks, masks_pred):
                        pred = (pred > cfg.out_threshold).float()
                        if cfg.n_classes > 1:
                            sub_cross_entropy = F.cross_entropy(pred.unsqueeze(dim=0), true_mask).item()

                        else:
                            sub_cross_entropy = dice_coeff(pred, true_mask.squeeze(dim=1)).item()
                        tot_cross_entropy += sub_cross_entropy
                    tot_cross_entropy = tot_cross_entropy / len(masks_preds)
                    tot += tot_cross_entropy
            else:
                masks_pred = net(imgs)
                for true_mask, pred in zip(true_masks, masks_pred):
                    pred = (pred > cfg.out_threshold).float()
                    cell_mask=true_mask.clone()
                    true_mask[true_mask==2]=0  #cytoplasm
                    cell_mask[cell_mask==1]=0
                    cell_mask[cell_mask==2]=1  #nuleus

                    if cfg.n_classes > 1:
                        cytoplasm_dice+=dice_coeff(pred[1,...].unsqueeze(dim=0),true_mask.unsqueeze(dim=0).float() ).item()
                        nuleus_dice+=dice_coeff(pred[2,...].unsqueeze(dim=0),cell_mask.unsqueeze(dim=0).float() ).item()
                        tot += F.cross_entropy(pred.unsqueeze(dim=0), true_mask.unsqueeze(dim=0).squeeze(1)).item()
                        cytoplasm_jac+=jaccard(pred[1,...].unsqueeze(dim=0),true_mask.unsqueeze(dim=0).float() ).item()
                        nuleus_jac+=jaccard(pred[2,...].unsqueeze(dim=0),cell_mask.unsqueeze(dim=0).float() ).item()
                        cytoplasm_acc+=accuracy(pred[1,...].unsqueeze(dim=0),true_mask.unsqueeze(dim=0).float() ).item()
                        nuleus_acc+=accuracy(pred[2,...].unsqueeze(dim=0),cell_mask.unsqueeze(dim=0).float() ).item()
                        cytoplasm_sen+=sensitivity(pred[1,...].unsqueeze(dim=0),true_mask.unsqueeze(dim=0).float() ).item()
                        nuleus_sen+=sensitivity(pred[2,...].unsqueeze(dim=0),cell_mask.unsqueeze(dim=0).float() ).item()
                        cytoplasm_spen+=specificity(pred[1,...].unsqueeze(dim=0),true_mask.unsqueeze(dim=0).float() ).item()
                        nuleus_spen+=specificity(pred[2,...].unsqueeze(dim=0),cell_mask.unsqueeze(dim=0).float() ).item()
                        cytoplasm_pre+=precision(pred[1,...].unsqueeze(dim=0),true_mask.unsqueeze(dim=0).float() ).item()
                        nuleus_pre+=precision(pred[2,...].unsqueeze(dim=0),cell_mask.unsqueeze(dim=0).float() ).item()
                        cytoplasm_rec+=recall(pred[1,...].unsqueeze(dim=0),true_mask.unsqueeze(dim=0).float() ).item()
                        nuleus_rec+=recall(pred[2,...].unsqueeze(dim=0),cell_mask.unsqueeze(dim=0).float() ).item()
                        cytoplasm_tve+=tversky(pred[1,...].unsqueeze(dim=0),true_mask.unsqueeze(dim=0).float() ).item()
                        nuleus_tve+=tversky(pred[2,...].unsqueeze(dim=0),cell_mask.unsqueeze(dim=0).float() ).item()

                        
                    else:
                        tot += dice_coeff(pred, true_mask.squeeze(dim=1)).item()

            pbar.update(imgs.shape[0])

    return cytoplasm_dice/n_val, nuleus_dice/n_val, tot/n_val, cytoplasm_jac/n_val, nuleus_jac/n_val,\
           cytoplasm_acc/n_val, nuleus_acc/n_val,cytoplasm_sen/n_val,nuleus_sen/n_val,\
           cytoplasm_spen/n_val,nuleus_spen/n_val,cytoplasm_pre/n_val,nuleus_pre/n_val,\
           cytoplasm_rec/n_val,nuleus_rec/n_val,cytoplasm_tve/n_val,nuleus_tve/n_val


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net=mynet()
    
    # input = torch.randn(1, 3, 128, 128)
    # flops, params = profile(net, (input,))
    # print('FLOPs = ' + str(1*flops/1000**3) + 'Gmac')
    # print('Params = ' + str(params/1000**2) + 'M')
    # print('net parameters:', sum(param.numel() for param in net.parameters()))

    logging.info(f'Network:\n'
                 f'\t{cfg.model} model\n'
                 f'\t{cfg.n_channels} input channels\n'
                 f'\t{cfg.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if cfg.bilinear else "Dilated conv"} ')

    if cfg.load:
        net.load_state_dict(
            torch.load(cfg.load, map_location=device)
        )
        logging.info(f'Model loaded from {cfg.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        # begin = time.time()
        train_net(net=net, cfg=cfg)
        # end = time.time()
        # print(f'\t{cfg.model} training time:', format(end-begin,'.4f')+'s')
        # num_pic=math.floor(n_train/cfg.batch_size)*cfg.batch_size*cfg.epochs
        # print(f'\t{cfg.model} training time/pic:', format((end-begin)/num_pic,'.4f')+'s')

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

