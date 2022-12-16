# --------------------------------------------------------
# The code has mainly been copied from:
# https://github.com/pemami4911/EfficientMORL
#
# A copy of the license for the original code can be found in this folder (LICENSE_EMORL).
#
# Changes includes removing the use of the package sacred and setting defaults in intialisation. 
# In addition, we resize masks for evalaution in a way which should be consisten which 
# what is done in for Iodine: https://github.com/deepmind/deepmind-research/blob/master/iodine/modules/data.py
# and for the ClevrTex dat
# -------------------------------------------------------- 

import torch
import numpy as np
import h5py
from pathlib import Path
from PIL import Image


class HdF5Dataset(torch.utils.data.Dataset):
    """
    The .h5 dataset is assumed to be organized as follows:
    {train|val|test}/
        imgs/  <-- a tensor of shape [dataset_size,H,W,C]
        masks/ <-- a tensor of shape [dataset_size,num_objects,H,W,C]
        factors/  <-- a tensor of shape [dataset_size,...]
    """
    def __init__(self, data_path, h5_path, masks=False, factors=False, clevr_preprocess_style='', 
                 d_set='train'):
        super(HdF5Dataset, self).__init__()
        self.h5_path = str(Path(data_path, h5_path))
        self.d_set = d_set.lower()
        self.masks = masks
        self.factors = factors
        self.clevr_preprocess_style = clevr_preprocess_style


    def preprocess(self, img):
        """
        img is assumed to be an array of integers each in 0-255 
        We preprocess them by mapping the range to -1,1
        
        """
        PIL_img = Image.fromarray(np.uint8(img))
        # square center crop of 192 x 192
        if self.clevr_preprocess_style == 'clevr-large':
            PIL_img = PIL_img.crop((64,29,256,221))
            PIL_img = PIL_img.resize((128,128))
        elif self.clevr_preprocess_style == 'clevr-small':
            PIL_img = PIL_img.crop((64,29,256,221))
            PIL_img = PIL_img.resize((96,96))

        # H,W,C --> C,H,W
        img = np.transpose(np.array(PIL_img), (2,0,1))

        # image range is -1,1
        img = img / 255.
        img = (img * 2) - 1  # to [-1,1]
        
        return img

    def preprocess_mask_clevr(self, mask):
        """
        [num_objects, h, w, c]
        Returns the square mask of size 192x192
        """
        o,h,w,c = mask.shape
        masks = []
        for i in range(o):
            mask_ = mask[i,:,:,0]
            PIL_mask = Image.fromarray(mask_, mode="F")
            # square center crop of 192 x 192
            PIL_mask = PIL_mask.crop((64,29,256,221))

            # --------------------------------------------------------
            # Added resizing of mask with nearest neighbour
            # which should consistent with procedure in ClevrTex:
            # https://github.com/karazijal/clevrtex-generation
            # and the one used with Iodine:
            # https://github.com/deepmind/deepmind-research/blob/master/iodine/modules/data.py
            # --------------------------------------------------------
            PIL_mask = PIL_mask.resize((128,128), resample=Image.NEAREST)

            masks += [np.array(PIL_mask)[...,None]]
        mask = np.stack(masks)  # [o,h,w,c]
        mask = np.transpose(mask, (0,3,1,2))  # [o,c,h,w]
        return mask    
    

    def __len__(self):
        with h5py.File(self.h5_path,  'r') as data:
            data_size, _, _, _ = data[self.d_set]['imgs'].shape
            return data_size

    def __getitem__(self, i):
        with h5py.File(self.h5_path,  'r') as data:
            outs = {}
            outs['imgs'] = self.preprocess(data[self.d_set]['imgs'][i].astype('float32')).astype('float32')
            if self.masks:
                if self.clevr_preprocess_style == 'clevr-large' or self.clevr_preprocess_style == 'clevr-small':
                    outs['masks'] = self.preprocess_mask_clevr(data[self.d_set]['masks'][i].astype('float32'))
                else:
                    outs['masks'] = np.transpose(data[self.d_set]['masks'][i].astype('float32'), (0,3,1,2))
            if self.factors:
                outs['factors'] = data[self.d_set]['factors'][i]
            return outs