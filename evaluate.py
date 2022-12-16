# --------------------------------------------------------
# Evalaution script for the object centric model. 
# Focused on segmentation metrics and qualitative evaluation of segmentations. 
# Uses code released with ClevrTex for the computation of segmentation metrics.
# # References: 
# https://github.com/karazijal/clevrtex-generation
# -------------------------------------------------------- 

import random
import argparse
import numpy as np

import torch
import torch.nn as nn

from model import ObjectViTAE
from thirdparty.clevrtex_eval import CLEVRTEX_Evaluator
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

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Intialize the dataloaders
    train_loader, val_loader, test_loader = data.get_loaders(args)

    # Initialize the model
    model = ObjectViTAE(args).to(device)
    print(f'Amount of model parameters: {utils.count_parameters(model)}M')

    # Load model only
    model = utils.load_model_checkpoint(args, model, map_location=device)
    
    # Evaluate the model on the test data
    model.eval()
    mse, ari_fg, ari, mIoU, _ = evaluate(model, test_loader, mask_ratio=0.0, device=device)
    print(f'mse: {mse}, ari_fg: {100.*ari_fg}, ari: {100.*ari}, mIoU: {100.*mIoU}')

    # If the dataset is ClevrTex, evaluate also on the Outd datasets
    if args.dataset == 'clevrtex':
        ood_loader, camo_loader = data.get_generalization_loaders(args)
        ood_mse, ood_ari_fg, ood_ari, ood_miou, _ = evaluate(model, ood_loader, mask_ratio=0.0, device=device)
        camo_mse, camo_ari_fg, camo_ari, camo_miou, _ = evaluate(model, camo_loader, mask_ratio=0.0, device=device)
        print(f'OOD set mse: {ood_mse}, ari_fg: {100.*ood_ari_fg}, ari: {100.*ood_ari}, mIoU: {100.*ood_miou}')
        print(f'CAMO test set mse: {camo_mse}, ari_fg: {100.*camo_ari_fg}, ari: {100.*camo_ari}, mIoU: {100.*camo_miou}')


def evaluate(model, data_loader, mask_ratio, device):
    """ Evalaute the model qualitatively and quantitatively on a dataset.
    """
    # Initialize evaluator for metrics calculation
    evaluator = CLEVRTEX_Evaluator()

    # Iterate data_loader, expects loader that return also ground truth segmentation
    for imgs, gt_segmentations in data_loader:
        imgs = imgs.to(device)
        gt_segmentations = gt_segmentations.to(device)
        with torch.no_grad():
            preds, pred_masks, pred_inds, slots, attn, img_masks, ids_restore = model(imgs, mask_ratio=mask_ratio)
            _, _ = evaluator.update(
                preds,
                pred_masks,
                imgs,
                gt_segmentations
            )

    # Get metrics from the evaluation 
    mse = evaluator.stats['MSE'].value()
    ari_fg = evaluator.stats['ARI_FG'].value()
    ari = evaluator.stats['ARI'].value()
    mIoU = evaluator.stats['mIoU'].value()

    # Create visualasation of the models outputs
    visuals = utils.create_visualisations(
        model, 
        imgs, 
        img_masks,
        preds,
        pred_masks, 
        pred_inds, 
        attn, 
        ids_restore
    )
    
    return mse, ari_fg, ari, mIoU, visuals


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Evaluation arguments')

    # Basic info
    parser.add_argument('--root', type=str, default='/', help='root directory')
    parser.add_argument('--data', type=str, default='data/', help='location of data')
    parser.add_argument('--dataset', type=str, default='tetrominoes', choices=['clevrtex', 'clevr6', 'multi_dsprites', 'tetrominoes'], help='the dataset to use')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/', help='directory for checkpoints')
    parser.add_argument('--model_id', type=str, default='base', help='model id for saving and logging')
    
    # Training hyperparameters
    parser.add_argument('--seed', type=int, default=53535, help='seed used') 
    parser.add_argument('--num_workers', type=int, default=5, help='number of workers in dataloader')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    
    # Model hyperparamaters
    parser.add_argument('--patch_size', type=int, default=5, help='height/width of patches')
    parser.add_argument('--mask_ratio', type=float, default=0.75, help='initial mask ratio')
    parser.add_argument('--num_slots', type=int, default=4, help='number of slots')
    parser.add_argument('--depth', type=int, default=4, help='encoder depth')
    parser.add_argument('--embed_dim', type=int, default=192, help='dimensionality of encoder embeddings')
    parser.add_argument('--num_heads', type=int, default=4, help='amount of heads in encoder')
    parser.add_argument('--decoder_depth', type=int, default=2, help='decoder depth')
    parser.add_argument('--decoder_embed_dim', type=int, default=128, help='dimensionality of decoder embeddings')
    parser.add_argument('--decoder_num_heads', type=int, default=4, help='amount of heads in decoder')
    parser.add_argument('--mlp_ratio', type=int, default=2, help='ratio of hidden dimensionality o input dim for mlps')

    # Parse arguments
    args = parser.parse_args()

    # Add dataset info to args
    data.add_data_info(args)

    # Path to model
    args.save_path = f'{args.checkpoint_dir}{args.dataset}/model-{args.model_id}/'

    # Run
    main(args)

