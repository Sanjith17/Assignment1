import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from config import MAX_BRIGHTNESS

def apply_gaussian(PILimgObj, StandardDev = 1, ShowStage = True, ShowOut = False):
    """
    Module to implement gaussian blur for the image with standard parameter select for the kernel
    Ref: https://pillow.readthedocs.io/en/stable/_modules/PIL/ImageDraw2.html
    """
    if ShowStage: print(f"*==============Applying Gaussian Filter==============*\n")
    if PILimgObj.mode != 'L':PILimgObj.convert('L')
        
    NewPILimgObj = PILimgObj.filter(ImageFilter.GaussianBlur(radius = StandardDev))
    
    if ShowOut:
        plt.figure()
        plt.imshow(NewPILimgObj,'gray')
    
    if ShowStage: print("*--------------Applied Gaussian Filter--------------* \n")
    
    return NewPILimgObj

def apply_sobel(PILimgObj, as_pil = False, ShowStage = True, ShowOut = False):
    
    """
    Module to implement Sobel Operator for the image edge enhancement using seperated kernels
    Ref: https://www.youtube.com/watch?v=qHn9mctee1Q
         https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
    """
    sobel_filter_gx = np.array([-1, 0, 1, -2, 0, 2, -1, 0, 1]) / 8 # finite difference filter
    sobel_filter_gy = np.array([1, 2, 1, 0, 0, 0, -1, -2, -1]) / 8 # smoothing filter

    if ShowStage: print("*==============Applying Seperable Sobel Filter==============*\n")
    if PILimgObj.mode != 'L': PILimgObj = PILimgObj.convert('L')
    
    SobelImgGx = PILimgObj.filter(ImageFilter.Kernel((3, 3), sobel_filter_gx,1,0))
    SobelImgGy = PILimgObj.filter(ImageFilter.Kernel((3, 3), sobel_filter_gy,1,0))
    
    SobelMag = np.hypot(np.array(SobelImgGx),np.array(SobelImgGy))
    NormSobelMag = ((SobelMag / np.max(SobelMag)) * MAX_BRIGHTNESS).astype(np.uint8) 
    
    SobelGrad = np.arctan2(np.array(SobelImgGy), np.array(SobelImgGx))
    NormSobelGrad = (SobelGrad * MAX_BRIGHTNESS / (2*np.pi)).astype(np.uint8)
    
    if ShowStage: print("*--------------Applied Sobel Filters--------------* \n")

    if ShowOut:
        plt.figure()
        fig,(ax1, ax2) = plt.subplots(1,2,figsize = (12,8),layout="tight")
        ax1.imshow(NormSobelMag,'gray')
        ax1.set_title("Sobel Magnitude image")
        
        ax2.set_title("Sobel Gradient image")
        ax2.imshow(NormSobelGrad,'gray')
        plt.show()
    if as_pil:
        PILSobelMag = Image.fromarray(NormSobelMag)
        PILSobelGrad = Image.fromarray(NormSobelGrad)
        return PILSobelMag, PILSobelGrad
        
    return NormSobelMag, NormSobelGrad

def apply_non_max_suppression(npMagArrImg, npGradArr, ShowStage = True):
    """
    This code is the implementation of Non-Max Suppression of sobel images using the Sobel Magnitude and Sobel Gradient.
    The implementation is taken from reference of the following tutorial:
    https://www.youtube.com/watch?v=qHn9mctee1Q
    
    """
    ImgW,ImgH = npMagArrImg.shape
    #Creating the new maximum suppressed image array after local area Non-Max Suppression
    NonMaxArr = np.zeros((ImgW,ImgH))
    if ShowStage: print("*==============Performing Non Max Suppression==============* \n")
    for _ in range(ImgW-1):
        for __ in range(ImgH-1):
            if npGradArr[_][__] < 0: npGradArr[_][__] += 360
            # 0 degrees
            if ((__+1) < ImgH) and ((__-1) >= 0) and ((_+1) < ImgW) and ((_-1) >= 0):
                if (npGradArr[_][__] >= 337.5 or npGradArr[_][__] < 22.5) or (npGradArr[_][__] >= 157.5 and npGradArr[_][__] < 202.5):
                    if npMagArrImg[_][__] >= npMagArrImg[_][__ + 1] and npMagArrImg[_][__] >= npMagArrImg[_][__ - 1]:NonMaxArr[_][__] = npMagArrImg[_][__]
            # 45 degrees
            if (npGradArr[_][__] >= 22.5 and npGradArr[_][__] < 67.5) or (npGradArr[_][__] >= 202.5 and npGradArr[_][__] < 247.5):
                if npMagArrImg[_][__] >= npMagArrImg[_ - 1][__ + 1] and npMagArrImg[_][__] >= npMagArrImg[_ + 1][__ - 1]:NonMaxArr[_][__] = npMagArrImg[_][__]
            # 90 degrees
            if (npGradArr[_][__] >= 67.5 and npGradArr[_][__] < 112.5) or (npGradArr[_][__] >= 247.5 and npGradArr[_][__] < 292.5):
                if npMagArrImg[_][__] >= npMagArrImg[_ - 1][__] and npMagArrImg[_][__] >= npMagArrImg[_ + 1][__]:NonMaxArr[_][__] = npMagArrImg[_][__]
            # 135 degrees
            if (npGradArr[_][__] >= 112.5 and npGradArr[_][__] < 157.5) or (npGradArr[_][__] >= 292.5 and npGradArr[_][__] < 337.5):
                if npMagArrImg[_][__] >= npMagArrImg[_ - 1][__ - 1] and npMagArrImg[_][__] >= npMagArrImg[_ + 1][__ + 1]:NonMaxArr[_][__] = npMagArrImg[_][__]
    if ShowStage: print("*--------------Non Max Suppressed--------------* \n")
                    
    return Image.fromarray(NonMaxArr)

def apply_thresholds(PILImg,LowThresh =75, HighThresh = 120, ShowStage = True):
    """
    Function to check the values of the pixels and threshold to 255 or 0 based on the neighbourhood conditions in the image
    Ref: https://github.com/StefanPitur/Edge-detection---Canny-detector/blob/master/canny.py
    """
    ImgW, ImgH = PILImg.size
    ThresholdedImg = np.zeros(shape=(ImgH,ImgW))
    if ShowStage: print("*==============Starting Thresholding of the image==============*\n")
    for i in range(ImgH):
        for j in range(ImgW):
            x = PILImg.getpixel((j,i))
            if(x>=HighThresh):ThresholdedImg[i,j] = 255
            elif(x<LowThresh): ThresholdedImg[i,j] = 0
            else:
                ThresholdedImg[i,j] = 0
                for k in range(8):
                    if(i+k>=0 and i+k<ImgH and j+k>=0 and j+k<ImgW and PILImg.getpixel((j+k,i+k))>0):
                        ThresholdedImg[i,j] = 255
            if(i==0 or i==ImgH-1 or j==0 or j==ImgW-1):ThresholdedImg[i,j] = 0
    if ShowStage: print("*--------------Thresholded the image--------------*\n")
    return ThresholdedImg

def process_image(image, config):
    std, high_threshold, low_threshold, vectorized, num_lines = config.values()
    # gaussian = apply_gaussian(image, std)
    sobel, sobel_gradient = apply_sobel(image, as_pil=True)
    suppressed = apply_non_max_suppression(np.array(sobel), np.array(sobel_gradient))
    thresholded = apply_thresholds(suppressed, low_threshold, high_threshold)
    # Image.fromarray(thresholded).show()
    print("Canny Edge Image generated\n")
    return thresholded