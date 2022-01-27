import os
import torch
import ROOT as rt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from util.misc import NestedTensor,collate_fn
import random
from PIL import Image
import torchvision.transforms as trans
import datasets.transforms as T
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
#from .pycocotools.uboone import UbooneAnnotation

class ubooneDetection(torch.utils.data.Dataset):
    def __init__(self, root_file_path, transforms,
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

        self._transforms = transforms

        # dictionary from PDG to class type
        #old = '''
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

        self.ok_entries = []
        self.notok_dets = []
        self.annotations = UbooneAnnotation(self)
                          
    #comment = '''
    def __getitem__(self, idx):
        workerinfo = torch.utils.data.get_worker_info()
        workerid = 0
        if workerinfo is not None:
            workerid = workerinfo.id
        chain = self.chains[workerid]
        current = self._current_entry[workerid]
        current_nbytes = 0
        nloaded = 0

        entry = idx

        current_nbytes = chain.GetEntry(entry)
        nloaded += 1
        if current_nbytes == 0:
            raise RuntimeError("Error reading entry %d" % (entry))

        # file stores images as arrays of shape (H,W), we expand  to (C=1,H,W)
        img_v = [np.expand_dims(chain.image_v.at(p).tonumpy(), axis=0) for p in self.planes]
        annote_v = [chain.bbox_v.at(p).tonumpy()[:, :4] for p in self.planes]
        class_v = [chain.pdg_v.at(p).tonumpy() for p in self.planes]

        ok = True
        for p in range(len(img_v)):
            notok_det_idx = []
            img = img_v[p]
            npix = (img > 10.0).sum()
            if npix < 20:
                ok = False
            bboxes = annote_v[p]
            nbboxes = bboxes.shape[0]
            if nbboxes > 10:
                ok = False
            for i, bbox in enumerate(bboxes):
                if bbox[2]*bbox[3] < (100**2)/512:
                    notok_det_idx.append(i)
                elif bbox[2] < 5/512 or bbox[3] < 5/512:
                    notok_det_idx.append(i)
            self.notok_dets.append(notok_det_idx)
        if ok:
            self.ok_entries.append(entry)

        # if reading file sequentially, iterate entry number
        if not self.random_access:
            entry += 1

        # make mask images
        if self.return_masks:
            mask_v = [chain.masks_v.at(p).tonumpy() for p in self.planes]
            maskimg_v = []
            for i, p in enumerate(self.planes):
                mask = mask_v[i]
                img = img_v[i]
                nmask = annote_v[i].shape[0]
                planemaskimg_v = []
                for ii in range(nmask):
                    iimask = mask[mask[:, 2] == ii]
                    np_mask = np.zeros((img.shape[1], img.shape[2]), dtype=np.uint8)
                    np_mask[iimask[:, 0], iimask[:, 1]] = 1
                    planemaskimg_v.append(np_mask)
                if len(planemaskimg_v) > 0:
                    maskimg = np.stack(planemaskimg_v, axis=0)
                else:
                    maskimg = np.zeros((0, img.shape[1], img.shape[2]), dtype=np.uint8)
                maskimg_v.append(maskimg)

        # convert pdg into class codes
        for p, pdg in enumerate(class_v):
            for i in range(pdg.shape[0]):
                if pdg[i] in self.pdg2class:
                    pdg[i] = self.pdg2class[pdg[i]]
                else:
                    pdg[i] = self.misc_class

        # we found a good entry

        # normalize the image
        img_norm_v = [self._normalize(img, num_channels=self._num_channels) for img in img_v]

        if len(self.planes) > 1:
            raise RuntimeError("Not fully implemented")
            target = {'image_id': np.array([entry], dtype=np.long), 'boxes': annote_v}
            if self.return_masks:
                target['masks'] = maskimg_v
            imgout = img_norm_v
        else:
            # single image returned
            target = {'image_id': np.array([entry], dtype=np.long), 'boxes': annote_v[0],
                      'labels': class_v[0].astype(np.long)}
            if self.return_masks:
                target['masks'] = maskimg_v[0]

            if self._num_predictions is not None:
                fixed_pred = np.zeros((self._num_predictions, 4), dtype=np.float32)
                nbbox = target['boxes'].shape[0]
                if nbbox < self._num_predictions:
                    fixed_pred[:nbbox, :] = target['boxes'][:, :]
                else:
                    fixed_pred[:, :] = target['boxes'][:self._num_predictions, :]
                target['boxes'] = fixed_pred
            imgout = img_norm_v[0]

            for name, arr in target.items():
                target[name] = torch.from_numpy(arr)

        w = imgout.shape[1]
        h = imgout.shape[2]
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        self._nloaded[workerid] += nloaded
        self._current_entry[workerid] = entry
        
        return torch.from_numpy(imgout), target
    #'''
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
                               random_access=True, #does not work with normal access
                               transforms=None,
                               num_workers=0,
                               num_predictions=None,
                               num_channels=3,
                               return_masks=args.masks)
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


from collections import defaultdict
import itertools
import time


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class UbooneAnnotation():
    def __init__(self, dataset):
        tic = time.time()
        self.dataset = dataset
        self.annotations = []
        self.images = []
        self.categories = []
        self.json_dict = self.prepare()
        self.createIndex()
        toc = time.time()
        print('UbooneAnnotation Created (t={:0.2f}s).'.format(toc - tic))

    def prepare(self):
        id = 0
        for idx in range(self.dataset.nentries):
            #for idx in self.dataset.ok_entries:
            #get the targets for the given entry
            img, target = self.dataset[idx]
            ok = idx in self.dataset.ok_entries
            self.images.append({'id': target['image_id'][0], 'image_tensor': img})
            for i, (bbox, label) in enumerate(zip(target['boxes'], target['labels'])):
                detection = {}
                detection['id'] = id
                detection['image_id'] = target['image_id'][0].item()
                detection['category_id'] = label.item()
                detection['bbox'] = bbox.tolist()
                detection['area'] = (bbox[2] * bbox[3]).item()
                detection['iscrowd'] = 0
                detection['ignore'] = 0 if ok else 1
                if i in self.dataset.notok_dets[idx]:
                    detection['ignore'] = 1
                self.annotations.append(detection)
                id += 1
        for cat, label in self.dataset.pdg2class.items():
            cat_dict = {}
            cat_dict['supercategory'] = 'particle'
            cat_dict['id'] = label
            cat_dict['name'] = cat
            self.categories.append(cat_dict)
        cat_dict = {}
        cat_dict['supercategory'] = 'particle'
        cat_dict['id'] = 8
        cat_dict['name'] = 'misc'
        self.categories.append(cat_dict)

        return {'categories': self.categories, 'annotations': self.annotations}

    def __str__(self):
        return str(self.json_dict)

    def createIndex(self):
        # create index
        anns, cats, imgs = {}, {}, {}
        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
        for ann in self.annotations:
            imgToAnns[ann['image_id']].append(ann)
            anns[ann['id']] = ann

        for img in self.images:
            imgs[img['id']] = img

        for pdg, num in self.dataset.pdg2class.items():
            cats[num] = pdg

        for ann in self.annotations:
            catToImgs[ann['category_id']].append(ann['image_id'])

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    def getImgIds(self):
        return list(range(0, self.dataset.nentries))

    def getCatIds(self):
        return list(range(1, 9))

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds) == 0 else [ann for ann in anns if ann['category_id'] in catIds]
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if ann['area']
                                                   > areaRng[0] and ann['area'] < areaRng[1]]
        if not iscrowd == None:
            ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:
            ids = [ann['id'] for ann in anns]
        return ids

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def loadRes(self, resFile, dataset):
        from pycocotools.coco import COCO
        import time
        import copy
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = UbooneAnnotation(dataset)
        res.images = self.images
        print('Loading and preparing results...')
        tic = time.time()
        if type(resFile) == str:
            anns = json.load(open(resFile))
        elif type(resFile) == np.ndarray:
            anns = self.loadNumpyAnnotations(resFile)
        else:
            anns = resFile
        assert type(anns) == list, 'results in not an array of objects'
        annsImgIds = [ann['image_id'] for ann in anns]
        assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
            'Results do not correspond to current coco set'
        if 'caption' in anns[0]:
            imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
            res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
            for id, ann in enumerate(anns):
                ann['id'] = id + 1
        elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
            res.categories = copy.deepcopy(self.categories)
            for id, ann in enumerate(anns):
                bb = ann['bbox']
                x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
                if not 'segmentation' in ann:
                    ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann['area'] = bb[2] * bb[3]
                ann['id'] = id + 1
                ann['iscrowd'] = 0
        elif 'segmentation' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                # now only support compressed RLE format as segmentation results
                ann['area'] = maskUtils.area(ann['segmentation'])
                if not 'bbox' in ann:
                    ann['bbox'] = maskUtils.toBbox(ann['segmentation'])
                ann['id'] = id + 1
                ann['iscrowd'] = 0
        elif 'keypoints' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                s = ann['keypoints']
                x = s[0::3]
                y = s[1::3]
                x0, x1, y0, y1 = np.min(x), np.max(x), np.min(y), np.max(y)
                ann['area'] = (x1 - x0) * (y1 - y0)
                ann['id'] = id + 1
                ann['bbox'] = [x0, y0, x1 - x0, y1 - y0]
        print('DONE (t={:0.2f}s)'.format(time.time() - tic))

        res.annotations = anns
        res.createIndex()
        return res

    def annToRLE(self, ann):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        t = self.imgs[ann['image_id']]
        h, w = t['height'], t['width']
        segm = ann['segmentation']
        if type(segm) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, h, w)
            rle = maskUtils.merge(rles)
        elif type(segm['counts']) == list:
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, h, w)
        else:
            # rle
            rle = ann['segmentation']
        return rle
