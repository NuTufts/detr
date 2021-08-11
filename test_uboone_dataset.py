from __future__ import print_function, division
import os,sys
import ROOT as rt
import time
from datasets.uboone import ubooneDetection

data = ubooneDetection( "test_detr2d.root", random_access=True )

nentries = len(data)
print("Num of entries: ",nentries)

start = time.time()
for n in range(nentries):
    entrydata = data[n]
    #print(n,entrydata)
    print("ENTRY %d"%(n))

elapsed = time.time()-start
sec_per_entry = elapsed/float(nentries)
print("Elapsed: ",elapsed," sec")
print("Time per entry: ",sec_per_entry," sec")
print("Num loaded: ",data._nloaded)


