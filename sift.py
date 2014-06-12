import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from scipy import ndimage as nd
from scipy.misc import *
import sys
import math
from matplotlib import pyplot

def prep_cascade(img, octave_size,octave_blur, scale_size,scale_ratio):
 retlist=[]
 invrat=1./scale_ratio
 stblur=1
 for scl in range(scale_size):
  octave=[]
  for blr in range (octave_size):
   cblur=stblur*blr*octave_blur
   octave.append(nd.filters.gaussian_filter1d(nd.filters.gaussian_filter1d(img,cblur,0),cblur,1))
  retlist.append(np.array(octave))
  img=imresize(img,invrat)
  stblur*=scale_ratio
 return retlist
def calc_DoG(cascade):
 DoG=[]
 for octave in cascade:
   DoG.append((np.roll(octave,-1,0)-octave)[:-1])
 return DoG