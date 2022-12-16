# --------------------------------------------------------
# Training script for the object-centric autoencoder. 
# Uses timm=0.5.4 for creating learning rate schduler and optimizer.
# References: 
# https://github.com/rwightman/pytorch-image-models
# -------------------------------------------------------- 

import time
import random
import argparse
import numpy as np
from datetime import timedelta

import torch
import torch.nn.functional as F

from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler

from model import ObjectViTAE
from evaluate import evaluate
import data
import utils


def main(args):
    # Print the arguments used
    print(f'Args: {args}') 

    # Get available device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Set the seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Intialize logger
    logger = utils.Logger(args)

    # Intialize the dataloaders
    train_loader, val_loader, test_loader = data.get_loaders(args)

    # Initialize the model
    model = ObjectViTAE(args).to(device)
    print(f'Amount of model parameters: {utils.count_parameters(model)}M')

    # Intialize the optimizer
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    # Intialize the learning rate scheduler and scheduler for 
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    scheduler = utils.Scheduler(args)

    # I cont_training is true, load the last checkpoint to continue from
    # Otherwise start from scratch
    if args.cont_training:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % 0}
        (model, 
        optimizer, 
        start_epoch, 
        step,
        _) = utils.load_checkpoint(args, model, optimizer, map_location=map_location)
    else:
        start_epoch, step = 0, 0

    # Set the schdules to start_epoch
    lr_scheduler.step(start_epoch)
    scheduler.step(start_epoch)

    # Start timer
    start_t = time.time()

    # Loop through epochs starting at epoch start_epoch
    for epoch in range(start_epoch, num_epochs):

        model.train()

        # Train for one epoch 
        step, tot_loss = train(
            model, 
            optimizer, 
            scheduler, 
            train_loader, 
            logger,
            epoch, 
            step, 
            device
        )

        # Evaluate and log  
        if epoch % args.eval_freq == 0:
            model.eval()
            mse, ari_fg, ari, miou, visuals = evaluate(
                model, 
                val_loader, 
                mask_ratio=scheduler.mask_ratio, 
                device=device
            )
            logger.write_eval_stats(step, mse, ari_fg, ari, miou, visuals)
            print(f'[Epoch {epoch+1} step {step} time {timedelta(seconds=int(time.time() - start_t))}] \
                        Train loss: {tot_loss / len(train_loader)}')
            print(f'Validation mse: {mse}, ari_fg: {100.*ari_fg}, \
                        ari: {100.*ari}, mIoU: {100.*miou}')
    
        # Saving
        if epoch % args.save_freq == 0:
            utils.save_checkpoint(args, model, optimizer, epoch+1, step+1)
            print('Model saved.')

        # Update schedulers
        lr_scheduler.step(epoch + 1)
        scheduler.step(epoch + 1)

    # Final evaluation on test set  
    model.eval()
    mse, ari_fg, ari, miou, visuals = evaluate(model, test_loader, mask_ratio=0.0, device=device)
    # Print and add the evaluation metrics and visuals to Tensorboard
    logger.write_eval_stats(step, mse, ari_fg, ari, miou, visuals, dataset='test')

    print(f'[Epoch {epoch+1} step {step} \
            time {timedelta(seconds=int(time.time() - start_t))}]')
    print(f'Test set mse: {mse}, ari_fg: {100.*ari_fg}, \
            ari: {100.*ari}, mIoU: {100.*miou}')

    # If the dataset is ClevrTex, evaluate also on the OOD and CAMO data
    if args.dataset == 'clevrtex':
        # Get the loaders
        ood_loader, camo_loader = data.get_generalization_loaders(args)

        # Do the evalaution on both datasets
        ood_mse, ood_ari_fg, ood_ari, ood_miou, ood_visuals = evaluate(
            model, 
            ood_loader, 
            mask_ratio=0.0,
            device=device
        )
        camo_mse, camo_ari_fg, camo_ari, camo_miou, camo_visuals = evaluate(
            model, 
            camo_loader, 
            mask_ratio=0.0,
            device=device
        )

        # Write to Tensorboard and print metrics
        logger.write_eval_stats(step, ood_mse, ood_ari_fg, ood_ari, ood_miou, ood_visuals, dataset='ood')
        logger.write_eval_stats(step, camo_mse, camo_ari_fg, camo_ari, camo_miou, camo_visuals, dataset='camo')
        print(f'OOD set mse: {ood_mse}, ari_fg: {100.*ood_ari_fg}, \
                ari: {100.*ood_ari}, mIoU: {100.*ood_miou}')
        print(f'CAMO test set mse: {camo_mse}, ari_fg: {100.*camo_ari_fg}, \
                ari: {100.*camo_ari}, mIoU: {100.*camo_miou}')

    # Save the final model
    utils.save_checkpoint(args, model, optimizer, epoch, step)
    print('Model saved.')

    # Get the total time of the experiment
    print(f'Total time: {timedelta(seconds=int(time.time() - start_t))}')
    

def train(model, optimizer, scheduler, data_loader, logger, epoch, step, device):
    """ Training for one epoch.
    """
    # Get the current mask ratio and scales on losses
    mask_ratio, scale_pixel_ent, scale_obj_ent, noise_scale = scheduler.values()

    # Intialize tot_loss for gathering loss
    tot_loss = 0.0

    # Iterate through batches 
    for imgs in data_loader:

        # Move to device
        imgs = imgs.to(device, non_blocking=True)
        
        # Forward 
        preds, pred_masks, _, _, _, _, _ = model(
            imgs, 
            mask_ratio=mask_ratio, 
            noise=noise_scale
        )
        
        # Caluculate mean of the output masks for object entropy loss
        mean_masks = pred_masks.mean(dim=(-1,-2))

        # Compute reconstruction loss and entropy losses
        recon_loss = F.mse_loss(preds, imgs)
        pixel_ent = -(pred_masks * torch.log(pred_masks + 1e-8)).sum(dim=1).mean() 
        obj_ent = -(mean_masks * torch.log(mean_masks + 1e-8)).sum(dim=1).mean() 

        # Full loss as a weighted sum
        loss = recon_loss + scale_pixel_ent * pixel_ent + scale_obj_ent * obj_ent

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update values for logging
        logger.update(loss, recon_loss, pixel_ent, obj_ent)
        tot_loss += loss.detach()
            
        # Tensorboard logging
        if step % args.log_freq == 0:
            logger.write_train_stats(
                step, 
                optimizer.param_groups[0]['lr'], 
                scheduler
            )

        # Increment step
        step += 1

    return step, tot_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Training arguments')

    # Basic info
    parser.add_argument('--root', type=str, default='/', help='root directory')
    parser.add_argument('--data', type=str, default='data/', help='location of data')
    parser.add_argument('--dataset', type=str, default='tetrominoes', choices=['clevrtex', 'clevr6', 'multi_dsprites', 'tetrominoes'], help='the dataset to use')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/', help='directory for checkpoints')
    parser.add_argument('--model_id', type=str, default='base', help='model id for saving and logging')
    
    # Logging
    parser.add_argument('--log_freq', type=int, default=100, help='frequency to log to tensorboard with, in steps')
    parser.add_argument('--eval_freq', type=int, default=10, help='frequency to evalaute with, in epochs')
    parser.add_argument('--save_freq', type=int, default=10, help='frequency to save with, defined in epochs')

    # Training hyperparameters
    parser.add_argument('--seed', type=int, default=53535, help='seed used') 
    parser.add_argument('--num_workers', type=int, default=5, help='number of workers in dataloader')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--opt', default='adamw', type=str, help='optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum')
    parser.add_argument('--weight-decay', type=float, default=0.05, help='weight decay)')
    parser.add_argument('--sched', default='cosine', type=str, help='learning rate scheduler')
    parser.add_argument('--lr', type=float, default=5e-4,help='base learning rate')
    parser.add_argument('--warmup_lr', type=float, default=1e-5, help='initial learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='minimum learning rate at the end of training')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='number of warmup epochs')
    parser.add_argument('--cooldown_epochs', type=int, default=30, help='number of cooldown epochs')

    # Model hyperparamaters
    parser.add_argument('--patch_size', type=int, default=5, help='height/width of patches')
    parser.add_argument('--init_mask_ratio', type=float, default=0.75, help='initial mask ratio')
    parser.add_argument('--num_slots', type=int, default=4, help='number of slots')
    parser.add_argument('--depth', type=int, default=4, help='encoder depth')
    parser.add_argument('--embed_dim', type=int, default=192, help='dimensionality of encoder embeddings')
    parser.add_argument('--num_heads', type=int, default=4, help='amount of heads in encoder')
    parser.add_argument('--decoder_depth', type=int, default=2, help='decoder depth')
    parser.add_argument('--decoder_embed_dim', type=int, default=128, help='dimensionality of decoder embeddings')
    parser.add_argument('--decoder_num_heads', type=int, default=4, help='amount of heads in decoder')
    parser.add_argument('--mlp_ratio', type=int, default=2, help='ratio of hidden dimensionality o input dim for mlps')

    # Loss function hyperparameters
    parser.add_argument('--scale_pixel_ent', type=float, default=1e-2, help='scale for pixel entropy loss')
    parser.add_argument('--scale_obj_ent', type=float, default=3e-3, help='scale for object entropy loss')
    parser.add_argument('--init_scale_pixel_ent', type=float, default=1e-4, help='initial scale for pixel entropy loss')
    parser.add_argument('--init_scale_obj_ent', type=float, default=1e-4, help='initial scale for object entropy loss')

    # Extra (only used for ClevrTex)
    parser.add_argument('--init_noise_scale', type=float, default=0.0, help='initial scale for noise injection')

    # Continue from checkpoint?
    parser.add_argument('--cont_training', action='store_true', default=False,
                        help='This flag enables training from an existing checkpoint.')

    # Parse arguments
    args = parser.parse_args()

    # Add saving path for checkpoint and logs
    args.save_path = f'{args.checkpoint_dir}{args.dataset}/model-{args.model_id}/'

    # Add dataset info to args
    data.add_data_info(args)

    # Create folder structure needed if not already in place
    utils.create_folder_structure(args)

    # Run training
    main(args)
