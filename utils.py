# --------------------------------------------------------
# Helper function or classes for setup, training and evaluation.
# -------------------------------------------------------- 

import os
import colorcet as cc

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter


# Helper for creating folder structure for model runs
def create_folder_structure(args):
    tensorboard_path = os.path.join(args.save_path, 'tensorboard')
    dirs = [args.save_path, tensorboard_path]
    for d in dirs:
        if not os.path.exists(args.save_path):    
            os.makedirs(args.save_path, exist_ok=True)
    

# Helper for getting number of paramaters
def count_parameters(model):
    count = sum(p.numel() for p in model.parameters() if p.requires_grad) / 10**6
    return count


# Helpers for saving and loading
def save_checkpoint(args, model, optimizer, epoch, step):
    checkpoint = {
        'args': args,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'step': step
    }
    file_path = os.path.join(args.save_path, 'checkpoint.pt')
    torch.save(checkpoint, file_path)

# Load the last checkpoint
def load_checkpoint(args, model, optimizer, map_location):
    file_path = os.path.join(args.save_path, 'checkpoint.pt')
    checkpoint = torch.load(file_path, map_location=map_location)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    step = checkpoint['step']
    args = checkpoint['args']
    return model, optimizer, epoch, step, args

# Load only the model
def load_model_checkpoint(args, model, map_location):
    file_path = os.path.join(args.save_path, 'checkpoint.pt')
    checkpoint = torch.load(file_path, map_location=map_location)
    model.load_state_dict(checkpoint['model'])
    return model

# Function for creating visualisations of the model output 
def create_visualisations(model, 
        imgs, img_masks, 
        preds, pred_masks, 
        pred_inds, attn, 
        ids_restore, num_samples=4):
    # Get the parameters of the in- and outputs
    device = imgs.device
    batch_size = imgs.size(0)
    num_slots = model.num_slots
    patch_size = model.patch_size
    h = w = model.res

    # Select random samples from the batch for all of the inputs
    idxs = torch.multinomial(torch.ones((batch_size,)), num_samples).to(device)
    all_data = [imgs, img_masks, preds, pred_masks, 
                    pred_inds, attn, ids_restore]
    all_data = [torch.index_select(data, index=idxs, dim=0) for data in all_data]
    (imgs, img_masks, preds, pred_masks, 
    pred_inds, attn, ids_restore) = all_data

    # Scale image to [0,1]
    imgs = (imgs / 2 + .5).clamp(0, 1)

    # Unpatchify the masked out patches
    img_masks = img_masks.unsqueeze(-1).repeat(1, 1, patch_size**2*3)
    img_masks = model.unpatchify(img_masks)

    # Visualization of the masked input
    imgs_masked = imgs * (1. - img_masks)

    # Construct visuals of the attention maps in bottleneck
    # we have to use the code inserting mask tokens from MAE, 
    # to fill the spots that were masked out
    # Reference: https://github.com/facebookresearch/mae
    empty = torch.zeros([attn.shape[0], ids_restore.shape[1] - attn.shape[1], num_slots], device=device)
    attn = torch.cat([attn, empty], dim=1)  
    attn = torch.gather(attn, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, attn.shape[2]))  

    # Then we reshape, repeat and unpatchify to construct attention masks
    # of image shapes
    attn = attn.permute(0,2,1)
    attn = attn.reshape(-1,attn.size(2), 1)
    attn = attn.repeat(1, 1, patch_size**2*3)
    attn = model.unpatchify(attn)
    attn = attn.reshape(num_samples, num_slots, 3, attn.size(2), attn.size(3))   
    attn = imgs.unsqueeze(1).repeat(1,num_slots,1,1,1) * attn + (1. - attn)

    # Construct visualisation of the predicted segmentations
    pred_segmentations = torch.argmax(pred_masks, dim=1).repeat(1,3,1,1)
    colors = torch.tensor(cc.glasbey_category10[:num_slots], device=device). \
                            reshape(num_slots,3,1,1).expand(num_slots,3,h,w)
    pred_segmentations = torch.gather(colors, dim=0, index=pred_segmentations.long())

    # Scale the predictions and and individual predictions 
    preds = (preds / 2 + .5).clamp(0, 1)
    pred_inds = (pred_inds / 2 + .5).clamp(0, 1)
    
    # Compute individual predictions with alpha channel
    pred_inds_alpha = pred_inds * pred_masks + (1. - pred_masks)

    # Create visusalisation for attention/segmentation masks in bottleneck
    attn_masks = torch.cat(
        [imgs.unsqueeze(1), 
        attn],
        dim=1
    )
    attn_masks_grid = torchvision.utils.make_grid(
        attn_masks.view(-1, attn_masks.size(2), 
        attn_masks.size(3), attn_masks.size(4)), 
        nrow=1+num_slots, pad_value=0.5
    )

    # Create visualisation of predicted segmentation and individual recons with alpha channel
    segmentations = torch.cat(
        [pred_segmentations.unsqueeze(1),
        pred_inds_alpha],
        dim=1
    )
    segmentations_grid = torchvision.utils.make_grid(
        segmentations.view(-1, segmentations.size(2), 
        segmentations.size(3), segmentations.size(4)), 
        nrow=1+num_slots, pad_value=0.5
    )

    # Construct visual of predictions and individual predictions
    recons = torch.cat(
        [preds.unsqueeze(1),
        pred_inds],
        dim=1
    )
    recons_grid = torchvision.utils.make_grid(
        recons.view(-1, recons.size(2), 
        recons.size(3), recons.size(4)), 
        nrow=1+num_slots, pad_value=0.5
    )

    # Construct summary visual of img, masked img and prediction and segmentation
    summary_visual = torch.cat(
        [imgs.unsqueeze(1), 
        imgs_masked.unsqueeze(1),
        preds.unsqueeze(1),
        pred_segmentations.unsqueeze(1),
        pred_inds_alpha],
        dim=1
    )
    summary_visual_grid = torchvision.utils.make_grid(
        summary_visual.view(-1, summary_visual.size(2), 
        summary_visual.size(3), summary_visual.size(4)), 
        nrow=4+num_slots, pad_value=0.5
    )
    
    return (attn_masks_grid, segmentations_grid, recons_grid, summary_visual_grid)


class Scheduler:
    """ Class for keeping track of loss weights and potential noise for the ClevrTex data.
    """
    def __init__(self, args):
        # Define constants
        self.epochs = args.epochs
        self.warmup_epochs = args.warmup_epochs
        self.delta = self.epochs - self.warmup_epochs
        self.init_mask_ratio = args.init_mask_ratio
        self.init_scale_pixel_ent = args.init_scale_pixel_ent
        self.init_scale_obj_ent = args.init_scale_obj_ent
        self.noise = args.init_noise_scale

        # Final values of varying paramaters 
        self.scale_pixel_ent_final = args.scale_pixel_ent
        self.scale_obj_ent_final = args.scale_obj_ent
        self.noise_final = 0.0

        # Initialize values of varying hyperparameters
        self.mask_ratio = args.init_mask_ratio
        self.scale_pixel_ent = args.init_scale_pixel_ent
        self.scale_obj_ent = args.init_scale_obj_ent
        self.noise = args.init_noise_scale

    def step(self, epoch):
        # If warmup is done, start linear increase of weigths and decrease of mask ration
        if epoch > self.warmup_epochs:
            # How many epochs past warmup is it currently
            current = 1.0 * epoch - self.warmup_epochs

            # Calculate new values of mask ratio and scales for entropy losses
            self.mask_ratio = max(self.init_mask_ratio - self.init_mask_ratio  \
                                    * (current / self.delta), 0.0)
            self.scale_pixel_ent = self.init_scale_pixel_ent \
                                    + min(1.0, current / self.delta) \
                                    * (self.scale_pixel_ent_final - self.init_scale_pixel_ent)
            self.scale_obj_ent = self.init_scale_obj_ent \
                                    + min(1.0, current / self.delta) \
                                    * (self.scale_obj_ent_final - self.init_scale_obj_ent)

            # In case noise was used in warmup, it is instantly removed afterwards
            self.noise = self.noise_final

    def values(self):
        # Return the values of the parameters
        return (
            self.mask_ratio, 
            self.scale_pixel_ent, 
            self.scale_obj_ent, 
            self.noise
        )


class Logger:
    """Tracks training statisics and writes to Tensorboard.
    Also includes method for writing evalaution information to Tensorboard.
    """
    def __init__(self, args):
        # Create the summarywriter for tensorboard    
        tensorboard_path = os.path.join(args.save_path, 'tensorboard')
        self.writer = SummaryWriter(tensorboard_path, flush_secs=30)

        # Initialize values for tracking losses
        self.n = 0
        self.loss = 0.
        self.recon_loss = 0.
        self.pixel_ent_loss = 0.
        self.obj_ent_loss = 0.
        
    def update(self, loss, recon_loss, pixel_ent_loss, obj_ent_loss):
        # Update values, and amount of batches
        self.n += 1
        self.loss += loss.detach()
        self.recon_loss += recon_loss.detach()
        self.pixel_ent_loss += pixel_ent_loss.detach()
        self.obj_ent_loss += obj_ent_loss.detach()
       
    def write_train_stats(self, step, lr, scheduler):
        # Write rolling average of loss to tensorbaord
        self.writer.add_scalar('train/loss', self.loss / self.n, step)
        self.writer.add_scalar('train/recon_loss', self.recon_loss / self.n, step)
        self.writer.add_scalar('train/pixel_ent_loss', self.pixel_ent_loss / self.n, step)
        self.writer.add_scalar('train/obj_ent_loss', self.obj_ent_loss / self.n, step)

        # Write information on current values of paramaters changing
        self.writer.add_scalar('train/lr', lr, step)
        self.writer.add_scalar('train/scale_pixel_ent', scheduler.scale_pixel_ent, step)
        self.writer.add_scalar('train/scale_obj_ent', scheduler.scale_obj_ent, step)
        self.writer.add_scalar('train/mask_ratio', scheduler.mask_ratio, step)

        # Reset the tracking
        self.n = 0
        self.loss = 0.
        self.recon_loss = 0.
        self.pixel_ent_loss = 0.
        self.obj_ent_loss = 0.

    def write_eval_stats(self, step, mse, ari_fg, ari, miou, visuals, dataset='val'):
        # Write the segmentation metrics from evluation to Tensorboard
        self.writer.add_scalar(f'{dataset}/mse', mse, step)
        self.writer.add_scalar(f'{dataset}/ari_fg', 100.*ari_fg, step)
        self.writer.add_scalar(f'{dataset}/ari', 100.*ari, step)
        self.writer.add_scalar(f'{dataset}/mIoU', 100.*miou, step)

        # Unpack visualisations
        (attn_masks, 
        pred_segementation, 
        pred_reconstructions,
        summary_visual) = visuals
        
        # Add qualitative evaluation of segmentation to Tensorboard
        self.writer.add_image(f'{dataset}/attn_masks', attn_masks, step)
        self.writer.add_image(f'{dataset}/pred_segmentation', pred_segementation, step)
        self.writer.add_image(f'{dataset}/pred_reconstuctions', pred_reconstructions, step)
        self.writer.add_image(f'{dataset}/summary_visual', summary_visual, step)

