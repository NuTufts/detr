import os
import torch
import ROOT as rt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from util.misc import NestedTensor,collate_fn

class ubooneDetection(torch.utils.data.Dataset):
    def __init__(self, root_file_path,
                 num_predictions=None,                 
                 planes=[2],
                 num_workers=1,
                 num_channels=1,
                 random_access=False,
                 return_masks=False):
        """
        Parameters:
          root_file_path: the location of a ROOT file to load data from
          num_predictions: if not None, then number of targets is padded or truncated to be a fixed size. Needed for DETR.
        """
        # the ROOT tree expected in the file
        #print("ubooneDetection")
        self._num_workers = num_workers
        if num_workers<=0:
            self._num_workers = 1
        self._num_predictions = num_predictions
        self._num_channels = num_channels
        
        self.chains = {}
        for i in range(self._num_workers):
            self.chains[i] = rt.TChain("detr")        
            if type(root_file_path) is str:
                if not os.path.exists(root_file_path):
                    raise RuntimeError("Root file path does not exist: ",root_file_path)
                self.chains[i].Add( root_file_path )

        self.nentries = self.chains[0].GetEntries()
        self.random_access = random_access
        self.return_masks = return_masks

        self._current_entry  = [0 for i in range(self._num_workers)]
        self._current_nbytes = [0 for i in range(self._num_workers)]
        self._nloaded        = [0 for i in range(self._num_workers)]
        self.planes = planes

        # dictionary from PDG to class type
        self.pdg2class = {-11:1,
                          11:1,
                          22:2,
                          -13:3,
                          13:4,
                          2212:5,
                          211:6,
                          -211:6,
                          2112:7}
        self.misc_class = 8
                          
        

    def __getitem__(self, idx):
        #img, target = super(CocoDetection, self).__getitem__(idx)

        workerinfo = torch.utils.data.get_worker_info()
        workerid = 0
        if workerinfo is not None:
            workerid = workerinfo.id
            #print(workerinfo)
            #print("workerid: ",workerid," chain=",self.chains[workerid])

        chain = self.chains[workerid]
        current = self._current_entry[workerid]
        current_nbytes = 0
        nloaded = 0
        
        ok = False
        entry = 0
        while not ok:
            # we loop until we get an acceptable entry

            # set the tree entry number, either randomly or by next entry number
            if not self.random_access:
                entry = current
            else:
                entry = np.random.randint(0,self.nentries)
        
            current_nbytes = chain.GetEntry(entry)
            nloaded += 1
            if current_nbytes==0:
                raise RuntimeError("Error reading entry %d"%(entry))

            # file stores images as arrays of shape (H,W), we expand  to (C=1,H,W)
            img_v    = [ np.expand_dims(chain.image_v.at(p).tonumpy(), axis=0) for p in self.planes ]
            annote_v = [ chain.bbox_v.at(p).tonumpy()[:,:4] for p in self.planes ]
            class_v  = [ chain.pdg_v.at(p).tonumpy() for p in self.planes ]

            # check the image is OK
            # - image has minimum number of pixels
            # - only use images with <=5 boxes (simple examples, presumably)
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

            # if reading file sequentially, iterate entry number
            if not self.random_access:
                entry += 1

            # if fails check, go back to beginning and try again
            if not ok:
                continue

            # make mask images
            if self.return_masks:
                mask_v   = [ chain.masks_v.at(p).tonumpy() for p in self.planes ]            
                maskimg_v = []
                for i,p in enumerate(self.planes):
                    mask = mask_v[i]
                    img  = img_v[i]
                    nmask = annote_v[i].shape[0]
                    planemaskimg_v = []
                    for ii in range(nmask):
                        iimask = mask[ mask[:,2]==ii ]
                        np_mask = np.zeros( (img.shape[1],img.shape[2]), dtype=np.uint8 )
                        np_mask[ iimask[:,0], iimask[:,1] ] = 1
                        planemaskimg_v.append( np_mask )
                    if len(planemaskimg_v)>0:
                        maskimg = np.stack( planemaskimg_v, axis=0 )
                    else:
                        maskimg = np.zeros( (0,img.shape[1],img.shape[2]), dtype=np.uint8 )
                    maskimg_v.append(maskimg)

            # convert pdg into class codes
            for p,pdg in enumerate(class_v):
                for i in range(pdg.shape[0]):
                    if pdg[i] in self.pdg2class:
                        pdg[i] = self.pdg2class[pdg[i]]
                    else:
                        pdg[i] = self.misc_class
                
        # we found a good entry

        # normalize the image
        img_norm_v = [ self._normalize( img, num_channels=self._num_channels ) for img in img_v ]
            
        if len(self.planes)>1:
            raise RuntimeError("Not fully implemented")
            target = {'image_id':np.array([entry],dtype=np.long), 'boxes':annote_v }
            if self.return_masks:
                target['masks'] = maskimg_v
            imgout = img_norm_v
        else:
            # single image returned
            target = {'image_id':np.array([entry],dtype=np.long), 'boxes':annote_v[0], 'labels':class_v[0].astype(np.long) }
            if self.return_masks:
                target['masks'] = maskimg_v[0]
            
            if self._num_predictions is not None:
                fixed_pred = np.zeros( (self._num_predictions,4), dtype=np.float32 )
                nbbox = target['boxes'].shape[0]
                if nbbox<self._num_predictions:
                    fixed_pred[:nbbox,:] = target['boxes'][:,:]
                else:
                    fixed_pred[:,:] = target['boxes'][:self._num_predictions,:]
                target['boxes'] = fixed_pred
            imgout = img_norm_v[0]

            for name,arr in target.items():
                target[name] = torch.from_numpy(arr)

        w = imgout.shape[1]
        h = imgout.shape[2]
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
            
        self._nloaded[workerid] += nloaded
        self._current_entry[workerid] = entry
        return torch.from_numpy(imgout), target

    def __len__(self):
        return self.nentries

    def _normalize(self, img_tensor, max_pixval=200.0, mip_peak=40.0, mip_std=20.0, num_channels=1 ):        
        """
        From torchvision.data readme:
        
          All pre-trained models expect input images normalized in the same way, 
          i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), 
          where H and W are expected to be at least 224. 
          The images have to be loaded in to a range of [0, 1] 
          and then normalized using mean = [0.485, 0.456, 0.406] 
          and std = [0.229, 0.224, 0.225]. 

        However, we have greyscale images with a long tail.
        The MIP peak is around 40 with a std of about 20, We center around this.
        We clip at 200.

        Parameters:
          img_tensor: (1,H,W) tensor
        """
        img_tensor = np.clip( img_tensor, 0, max_pixval )
        img_tensor -= mip_peak
        img_tensor *= (1.0/mip_std)
        if num_channels>1:
            img_tensor = np.tile( img_tensor.reshape(-1), num_channels ).reshape( (num_channels, img_tensor.shape[1], img_tensor.shape[2]) )

        return img_tensor

    def print_status(self):
        for i in range(self._num_workers):
            print("worker: entry=%d nloaded=%d"%(self._current_entry[i],self._nloaded[i]))


def build( image_set, args ):
    if not os.path.exists(args.uboone_path):
        assert "provided uboone detection path {%s} does not exist"%(args.uboone_path)

    datafile = "test_detr2d.root"
    dataset = ubooneDetection( args.uboone_path+"/"+datafile,
                               random_access=True,
                               num_workers=0,
                               num_predictions=None,
                               num_channels=3,
                               return_masks=args.masks )
    return dataset
    
            
if __name__ == "__main__":

    import time

    niter = 10
    num_workers = 0
    batch_size = 4
    
    test = ubooneDetection( "test_detr2d.root", random_access=True,
                            num_workers=num_workers,
                            num_predictions=None,
                            num_channels=3,
                            return_masks=True )
    loader = torch.utils.data.DataLoader(test,batch_size=batch_size,
                                         num_workers=num_workers,
                                         collate_fn=collate_fn,
                                         persistent_workers=False)

    start = time.time()
    for iiter in range(niter):
        img, data = next(iter(loader))
        print("ITER[%d]"%(iiter))
        print(" len(data)=",len(data))
        print(" img.shape=",img.tensors.shape," ",img.tensors.dtype)
        print(" mask.shape=",img.mask.shape)
        print(" data[0][boxes]=",data[0]['boxes'].shape,data[0]['boxes'].dtype)
        print(" data[0]['masks']=",data[0]['masks'].shape)
        print(" data[0]['labels']=",data[0]['labels'].shape,data[0]['labels'].dtype,data[0]['labels'])
        print(" max: ", img.tensors.max())
        print(" min: ", img.tensors.min())
        print(" mean: ", img.tensors.mean())
        print(" std: ", img.tensors.std())
    end = time.time()
    elapsed = end-start
    sec_per_iter = elapsed/float(niter)
    print("sec per iter: ",sec_per_iter)
    
    loader.dataset.print_status()
