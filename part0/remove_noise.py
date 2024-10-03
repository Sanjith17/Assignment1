import numpy as np
import matplotlib
matplotlib.use('Agg')
from scipy import fftpack
import imageio
import numpy
import sys
import matplotlib.pyplot as plt


if __name__ == '__main__':
    if(len(sys.argv) < 2):
        raise Exception("error: please give an input image name as a parameter, like this: \n"
                     "python3 fourier.py input.jpg")
    
    # load in an image, convert to grayscale if needed
    image = imageio.imread(sys.argv[1], as_gray=True)

    # take the fourier transform of the image
    fft2 = fftpack.fftshift(fftpack.fft2(image))

        
    fft3 = fftpack.fftshift(fftpack.fft2(image))
    #print(fft2)
    fft2[79:88, 79:92 ] = 0
    fft2[120:127, 116:127] = 0

    imageio.imsave('fft_after_.png', (numpy.log(abs(fft2))* 255 /numpy.amax(numpy.log(abs(fft2)))).astype(numpy.uint8))
    ifft2 = abs(fftpack.ifft2(fftpack.ifftshift(fft2)))
    imageio.imsave('remove_noise_.png', ifft2.astype(numpy.uint8))

    #application of band pass filter
    #finding the size fo fft to create a filter to multiply with fft
    x, y = fft2.shape
    c_x, c_y = int(x / 2), int(y / 2)
    #making filter with desired size basd on the fft
    filter_ = numpy.ones((x, y), numpy.uint8)
    #radius of band stop range from origin to stop th "Hi" noise
    r_out = 30
    r_in = 20
    center = [c_x, c_y]
    x, y = numpy.ogrid[:x, :y]
    #making the band stop aarea by making the values as 0
    filter__area = numpy.logical_and(((x - center[0]) ** 2 + (y - center[1]) ** 2 >= r_in ** 2),
                            ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= r_out ** 2))
    filter_[filter__area] = 0

    # now take the inverse transform to convert back to an image

    fft3=fft3*filter_
    imageio.imsave('fft_tran_bands.png', (numpy.log(abs(fft3))* 255 /numpy.amax(numpy.log(abs(fft3)))).astype(numpy.uint8))
    ifft3 = abs(fftpack.ifft2(fftpack.ifftshift(fft3)))
    imageio.imsave('remove_noise_bandstop.png', ifft3.astype(numpy.uint8))
        
