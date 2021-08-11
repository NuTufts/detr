from __future__ import print_function, division
import os
import torch
import ROOT as rt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import datasets.transforms as T


class ubooneDetection(torch.utils.data.Dataset):
    def __init__(self, root_file_path, planes=[2], random_access=False, return_masks=False):
        # the ROOT tree expected in the file
        self.chain = rt.TChain("detr")
        
        if type(root_file_path) is str:
            if not os.path.exists(root_file_path):
                raise RuntimeError("Root file path does not exist: ",root_file_path)
            self.chain.Add( root_file_path )

        self.nentries = self.chain.GetEntries()
        self.random_access = random_access
        self.return_masks = return_masks
        self._current_entry = 0
        self._current_nbytes = 0
        self._nloaded = 0
        self.planes = planes
        

    def __getitem__(self, idx):
        #img, target = super(CocoDetection, self).__getitem__(idx)

        ok = False
        while not ok:
            if not self.random_access:
                entry = self._current_entry
            else:
                entry = np.random.randint(0,self.nentries)
        
            self._current_nbytes = self.chain.GetEntry(entry)
            self._nloaded += 1
            if self._current_nbytes==0:
                raise RuntimeError("Error reading entry %d"%(entry))

            img_v    = [ self.chain.image_v.at(p).tonumpy() for p in self.planes ]
            annote_v = [ self.chain.bbox_v.at(p).tonumpy()[:,:5] for p in self.planes ]

            # check the image is OK
            ok = True
            for p in range(len(img_v)):
                img = img_v[p]
                npix = (img>10.0).sum()
                if npix<20:
                    ok = False
                bbox = annote_v[p]
                nboxes = bbox.shape[0]
                if nboxes>5:
                    ok = False

            if not self.random_access:
                self._current_entry = entry + 1
            else:
                self._current_entry = entry
            
        if len(self.planes)>1:
            target = {'image_id':entry, 'annotations':annote_v }
            imgout = img_v
        else:
            target = {'image_id':entry, 'annotations':annote_v[0]}
            imgout = img_v[0]
            
        return imgout, target

    def __len__(self):
        return self.nentries


