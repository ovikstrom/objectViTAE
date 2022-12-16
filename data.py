# --------------------------------------------------------
# Code for constructing data loaders. We use dataset code from EMORL for
# Tetrominoes, Multi-dSprites and CLEVR6. 
# For ClevrTex we use the dataset code provided with the data and paper.
# We also define some custom collate functions. 
# Finally this script contains a function with names for datasets. 
# If names need to be changed you can do this here or add a argument with the name.
# References: 
# https://github.com/pemami4911/EfficientMORL
# https://github.com/karazijal/clevrtex-generation
# -------------------------------------------------------- 

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from thirdparty.datasets import HdF5Dataset
from thirdparty.clevrtex_eval import CLEVRTEX

# Helper for creating dataloaders
# For Tetrominoes, multi-dSprites and CLEVR6 we use simply 
# a subset of the training set for tracking metrics during training.
# We call this validation set altought its simply a subset of the train data.
# A size of 320 is used for this set (similarly to the size of the final test set)
# On ClevrTex we follow the implmentation in https://github.com/karazijal/clevrtex-generation
# which keeps a separate validation set, and we evaluate on that during training.
def get_loaders(args):
    if args.dataset == 'tetrominoes':   
        train_data = HdF5Dataset(args.data, get_name(args.dataset), d_set='train')

        # Indices for trains set samples to track metrics on
        indices = torch.multinomial(torch.ones((len(train_data),)), 320)

        val_data = Subset(HdF5Dataset(args.data, get_name(args.dataset), masks=True, d_set='train'),
                            indices=indices)
        test_data = HdF5Dataset(args.data, get_name(args.dataset, d_set='test'), 
                                    masks=True, d_set='test')
        collate_train_fn = collate_train
        collate_test_fn = collate_test
        
    elif args.dataset == 'multi_dsprites':
        train_data = HdF5Dataset(args.data, get_name(args.dataset), d_set='train')

        # Indices for trains set samples to track metrics on
        indices = torch.multinomial(torch.ones((len(train_data),)), 320)

        val_data = Subset(HdF5Dataset(args.data, get_name(args.dataset), masks=True, d_set='train'),
                            indices=indices)
        test_data = HdF5Dataset(args.data, get_name(args.dataset, d_set='test'), 
                                    masks=True, d_set='test')
        collate_train_fn = collate_train
        collate_test_fn = collate_test

    elif args.dataset == 'clevr6':
        train_data = HdF5Dataset(args.data, get_name(args.dataset),
                                    clevr_preprocess_style='clevr-large', d_set='train')
        
        # Indices for trains set samples to track metrics on
        indices = torch.multinomial(torch.ones((len(train_data),)), 320)

        val_data = Subset(HdF5Dataset(args.data, get_name(args.dataset), masks=True, 
                                     clevr_preprocess_style='clevr-large', d_set='train'), indices=indices)
        test_data = HdF5Dataset(args.data, get_name(args.dataset, d_set='test'), 
                                    masks=True, clevr_preprocess_style='clevr-large', d_set='test')
        collate_train_fn = collate_train
        collate_test_fn = collate_test

    elif args.dataset == 'clevrtex':
        train_data = CLEVRTEX(Path(args.data), split='train')
        val_data = CLEVRTEX(Path(args.data), split='val')
        test_data = CLEVRTEX(Path(args.data), split='test')
        collate_train_fn = collate_tex_train
        collate_test_fn = collate_tex_test

    else:
        raise NotImplementedError

    # Construct the dataloaders
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collate_train_fn
    )

    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_test_fn
    )
    
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_test_fn
    )

    return train_loader, val_loader, test_loader


# Getting the generalization test sets for ClevrTex
def get_generalization_loaders(args):
    ood_data = CLEVRTEX(Path(args.data), dataset_variant='outd')
    camo_data = CLEVRTEX(Path(args.data), dataset_variant='camo', split='test')
    collate_test_fn = collate_tex_test

    # Construct the dataloaders
    ood_loader = DataLoader(
        ood_data,
        batch_size=args.batch_size,
        shuffle=True,
        sampler=None,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=collate_test_fn
    )
    
    camo_loader = DataLoader(
        camo_data,
        batch_size=args.batch_size,
        shuffle=True,
        sampler=None,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=collate_test_fn
    )

    return ood_loader, camo_loader


# Helper for getting the name to Tetrominoes, Multi-dSprites and CLEVR6 files
def get_name(dataset, d_set='train'):
    if d_set == 'train':
        if dataset == 'clevr6':
            name = 'clevr6_train.h5'
        elif dataset == 'multi_dsprites':
            name = 'multi_dsprites_train.h5'
        elif dataset == 'tetrominoes':
            name = 'tetrominoes_train.h5'
        else:
            raise NotImplementedError
    elif d_set == 'test':
        if dataset == 'clevr6':
            name = 'clevr6_test.h5'
        elif dataset == 'multi_dsprites':
            name = 'multi_dsprites_test.h5'
        elif dataset == 'tetrominoes':
            name = 'tetrominoes_test.h5'
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return name


# Helpers for getting dataset specific information
def get_resolution(dataset):
    if dataset == 'tetrominoes':
        return 35
    elif dataset == 'multi_dsprites':
        return 64
    elif dataset == 'clevr6':
        return 128
    elif dataset == 'clevrtex':
        return 128
    else:
        raise NotImplementedError
    
def get_channels(dataset):
    if dataset in ['tetrominoes', 'multi_dsprites', 'clevr6', 'clevrtex']:
        return 3
    else:
        raise NotImplementedError


# Helper for adding dataset specific info to args
def add_data_info(args):
    args.img_resolution = get_resolution(args.dataset)
    args.img_channels = get_channels(args.dataset)


# Collate functions
# We use separate 
def collate_train(batch):
    imgs = torch.stack([torch.FloatTensor(x['imgs']) for x in batch])
    return imgs

def collate_test(batch):
    imgs = torch.stack([torch.FloatTensor(x['imgs']) for x in batch])
    # For eval sets we also have masks
    masks = torch.stack([torch.FloatTensor(x['masks']) for x in batch])
    return imgs, masks

def collate_tex_train(batch):
    imgs = torch.stack([x[1] for x in batch])
    # The clevrtex imgs are not already scaled to [-1,1] so we do it here
    imgs = 2. * imgs - 1.
    return imgs

def collate_tex_test(batch):
    imgs = torch.stack([x[1] for x in batch])
    # The clevrtex imgs are not already scaled to [-1,1] so we do it here
    imgs = 2. * imgs - 1.
    # For eval sets we also have masks
    masks = torch.stack([x[2] for x in batch])
    return imgs, masks
