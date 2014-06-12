import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from scipy import ndimage as nd
from scipy.misc import *
import sys
import math
from matplotlib import pyplot
try:
    import scipy.ndimage.interpolation as ndii
except ImportError:
    import ndimage.interpolation as ndii

print sys.argv
import sift


def translation(im0, im1):
    """Return translation vector to register images."""
    shape = im0.shape
    f0 = fft2(im0)
    f1 = fft2(im1)
    ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
    t0, t1 = np.unravel_index(np.argmax(ir), shape)
    if t0 > shape[0] // 2:
        t0 -= shape[0]
    if t1 > shape[1] // 2:
        t1 -= shape[1]
    return [t0, t1]

def logpolar(image, angles=None, radii=None):
    """Return log-polar transformed image and log base."""
    shape = image.shape
    center = shape[0] / 2, shape[1] / 2
    if angles is None:
        angles = shape[0]
    if radii is None:
        radii = shape[1]
    theta = np.empty((angles, radii), dtype=np.float64)
    theta.T[:] = -np.linspace(0, np.pi, angles, endpoint=False)
    #d = radii
    d = np.hypot(shape[0]-center[0], shape[1]-center[1])
    log_base = 10.0 ** (math.log10(d) / (radii))
    radius = np.empty_like(theta)
    radius[:] = np.power(log_base, np.arange(radii,
                                                   dtype=np.float64)) - 1.0
    x = radius * np.sin(theta) + center[0]
    y = radius * np.cos(theta) + center[1]
    output = np.empty_like(x)
    ndii.map_coordinates(image, [x, y], output=output)
    return output, log_base


def highpass(shape):
    """Return highpass filter to be multiplied with fourier transform."""
    x = np.outer(
        np.cos(np.linspace(-math.pi/2., math.pi/2., shape[0])),
        np.cos(np.linspace(-math.pi/2., math.pi/2., shape[1])))
    return (1.0 - x) * (2.0 - x)
def similarity(im0, im1):
    """Return similarity transformed image im1 and transformation parameters.
        
        Transformation parameters are: isotropic scale factor, rotation angle (in
        degrees), and translation vector.
        
        A similarity transformation is an affine transformation with isotropic
        scale and without shear.
        
        Limitations:
        Image shapes must be equal and square.
        All image areas must have same scale, rotation, and shift.
        Scale change must be less than 1.8.
        No subpixel precision.
        
        """
    if im0.shape != im1.shape:
        raise ValueError("Images must have same shapes.")
    elif len(im0.shape) != 2:
        #if (len
        raise ValueError("Images must be 2 dimensional.")
    
    f0 = fftshift(abs(fft2(im0)))
    f1 = fftshift(abs(fft2(im1)))
    
    h = highpass(f0.shape)
    f0 *= h
    f1 *= h
    del h
    
    f0, log_base = logpolar(f0)
    f1, log_base = logpolar(f1)
    
    f0 = fft2(f0)
    f1 = fft2(f1)
    r0 = abs(f0) * abs(f1)
    ir = abs(ifft2((f0 * f1.conjugate()) / r0))
    i0, i1 = np.unravel_index(np.argmax(ir), ir.shape)
    angle = 180.0 * i0 / ir.shape[0]
    scale = log_base ** i1
    
    if scale > 1.8:
        ir = abs(ifft2((f1 * f0.conjugate()) / r0))
        i0, i1 = np.unravel_index(np.argmax(ir), ir.shape)
        angle = -180.0 * i0 / ir.shape[0]
        scale = 1.0 / (log_base ** i1)
        if scale > 1.8:
            raise ValueError("Images are not compatible. Scale change > 1.8")
    
    if angle < -90.0:
        angle += 180.0
    elif angle > 90.0:
        angle -= 180.0
    
    im2 = ndii.zoom(im1, 1.0/scale)
    im2 = ndii.rotate(im2, angle)
    
    if im2.shape < im0.shape:
        t = np.zeros_like(im0)
        t[:im2.shape[0], :im2.shape[1]] = im2
        im2 = t
    elif im2.shape > im0.shape:
        im2 = im2[:im0.shape[0], :im0.shape[1]]
    
    f0 = fft2(im0)
    f1 = fft2(im2)
    ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
    t0, t1 = np.unravel_index(np.argmax(ir), ir.shape)
    
    if t0 > f0.shape[0] // 2:
        t0 -= f0.shape[0]
    if t1 > f0.shape[1] // 2:
        t1 -= f0.shape[1]
    
    im2 = ndii.shift(im2, [t0, t1])
    
    # correct parameters for ndimage's internal processing
    if angle > 0.0:
        d = int((int(im1.shape[1] / scale) * math.sin(math.radians(angle))))
        t0, t1 = t1, d+t0
    elif angle < 0.0:
        d = int((int(im1.shape[0] / scale) * math.sin(math.radians(angle))))
        t0, t1 = d+t1, d+t0
    scale = (im1.shape[1] - 1) / (int(im1.shape[1] / scale) - 1)
    
    return im2, scale, angle, [-t0, -t1]
def imshow(im0, im1, im2, im3=None, cmap=None, **kwargs):
    """Plot images using matplotlib."""
    if cmap is None:
        cmap = 'coolwarm'
    if im3 is None:
        im3 = abs(im2 - im0)
    pyplot.subplot(221)
    pyplot.imshow(im0, cmap, **kwargs)
    pyplot.subplot(222)
    pyplot.imshow(im1, cmap, **kwargs)
    pyplot.subplot(223)
    pyplot.imshow(im3, cmap, **kwargs)
    pyplot.subplot(224)
    pyplot.imshow(im2, cmap, **kwargs)
    pyplot.show()
sal=len(sys.argv)
if sal==2:
 k=nd.imread(sys.argv[1])
 print k.shape
 bl=nd.filters.gaussian_filter1d(k,5,0)
 bl=nd.filters.gaussian_filter1d(bl,5,1)
 imsave('blur.jpg',bl)

if sal==3:
 im0=imresize(nd.imread(sys.argv[1]),0.125)
 print sift.prep_cascade(im0,2,1.3,2,2)[0].shape
 
 #pyplot.imshow(im0, interpolation='bilinear')
 #pyplot.show()
 im0r=np.sum(im0, axis=2)/3.
 im1=imresize(nd.imread(sys.argv[2]),0.125)
 im1r=np.sum(im1, axis=2)/3.
 im2, scale, angle, (t0, t1) = similarity(im0r, im1r)
 print "Scale: {0} Angle: {1} Translate: {2}, {3}".format(scale,angle,t0,t1)
 imshow(im0r, im1r, im2)

