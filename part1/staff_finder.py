import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
from PIL import Image,ImageFilter,ImageDraw
import matplotlib.pyplot as plt

#Reference: https://github.com/tbharathchandra/Canny-Operator-with-morphological-completion/blob/master/CannyOperator.ipynb and 
#https://www.youtube.com/watch?v=qHn9mctee1Q and https://stackoverflow.com/questions/15892116/is-the-sobel-filter-meant-to-be-normalized#:~:text=A%20mathematically%20correct%20normalization%20for,one%20gray%2Dlevel%20per%20pixel.
#https://gist.github.com/bygreencn/6a900fd2ff5d0101473acbc3783c4d92

#Setting the max pixel brightness for the image
MAX_BRIGHTNESS = 255

# Direction noramalized sobel filter init
sobel_filter_gx = np.array([-1, 0, 1, -2, 0, 2, -1, 0, 1]) / 8 # finite difference filter
sobel_filter_gy = np.array([1, 2, 1, 0, 0, 0, -1, -2, -1]) / 8 # smoothing filter

# SobelFilterGy = np.array([2, 1, 0, -1, -2, 2, 1, 0, -1, -2, 4, 2, 0, -2, -4, 2, 1, 0, -1, -2, 2, 1, 0, -1, -2]) # finite difference filter
# SobelFilterGx = np.array([2, 2, 4, 2, 2, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, -1, -1, -2, -1, -1, -2, -2, -4, -2, -2]) # smoothing filter

def ApplyGaussian(PILimgObj, StandardDev = 1, ShowStage = True, ShowOut = False):
    """
    Module to implement gaussian blur for the image with standard parameter select for the kernel
    Ref: https://pillow.readthedocs.io/en/stable/_modules/PIL/ImageDraw2.html
    """
    if ShowStage: print("*============== Applying Gaussian Filter ==============*\n")
    if PILimgObj.mode != 'L':PILimgObj.convert('L')
        
    NewPILimgObj = PILimgObj.filter(ImageFilter.GaussianBlur(radius = StandardDev))
    
    if ShowOut:
        plt.figure()
        plt.imshow(NewPILimgObj,'gray')
    
    if ShowStage: print("*-------------- Applied Gaussian Filter --------------* \n")
    
    return NewPILimgObj

def ApplySobel(PILimgObj, AsPil = False, ShowStage = True, ShowOut = False):
    
    """
    Module to implement Sobel Operator for the image edge enhancement using seperated kernels
    Ref: https://www.youtube.com/watch?v=qHn9mctee1Q
         https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
    """
    if ShowStage: print("*============== Applying Seperable Sobel Filter ==============*\n")
    if PILimgObj.mode != 'L': PILimgObj = PILimgObj.convert('L')
    
    SobelImgGx = PILimgObj.filter(ImageFilter.Kernel((3, 3), sobel_filter_gx,1,0))
    SobelImgGy = PILimgObj.filter(ImageFilter.Kernel((3, 3), sobel_filter_gy,1,0))
    
    SobelMag = np.hypot(np.array(SobelImgGx),np.array(SobelImgGy))
    NormSobelMag = ((SobelMag / np.max(SobelMag)) * MAX_BRIGHTNESS).astype(np.uint8) 
    
    SobelGrad = np.arctan2(np.array(SobelImgGy), np.array(SobelImgGx))
    NormSobelGrad = (SobelGrad * MAX_BRIGHTNESS / (2*np.pi)).astype(np.uint8)
    
    if ShowStage: print("*-------------- Applied Sobel Filters --------------* \n")

    if ShowOut:
        plt.figure()
        fig,(ax1, ax2) = plt.subplots(1,2,figsize = (12,8),layout="tight")
        ax1.imshow(NormSobelMag,'gray')
        ax1.set_title("Sobel Magnitude image")
        
        ax2.set_title("Sobel Gradient image")
        ax2.imshow(NormSobelGrad,'gray')
        plt.show()
    if AsPil:
        PILSobelMag = Image.fromarray(NormSobelMag)
        PILSobelGrad = Image.fromarray(NormSobelGrad)
        return PILSobelMag, PILSobelGrad
        
    return NormSobelMag, NormSobelGrad


def NonMaxSuppression(npMagArrImg, npGradArr, ShowStage = True):
    """
    This code is the implementation of Non-Max Suppression of sobel images using the Sobel Magnitude and Sobel Gradient.
    The implementation is based from reference of the following tutorial:
    https://www.youtube.com/watch?v=qHn9mctee1Q
    The implementation is improved from the ref by using numpy based masking and improves the performance vastly 
    with the help of chat gpt prompt requesting for ways for vectorization of the if and for cases into numpy masks
    
    """
    ImgW, ImgH = npMagArrImg.shape

    # Create the new maximum suppressed image array after local area Non-Max Suppression
    NonMaxArr = np.zeros((ImgW, ImgH))

    if ShowStage:
        print("*============== Performing Non Max Suppression ==============*\n")

    # Convert negative angles to their positive equivalent
    npGradArr[npGradArr < 0] += 360

    # Suppress pixels that are not local maxima along the direction of the gradient
    # 0 degrees
    mask1 = ((npGradArr >= 337.5) | (npGradArr < 22.5)) | ((npGradArr >= 157.5) & (npGradArr < 202.5))
    mask2 = (npMagArrImg >= np.roll(npMagArrImg, 1, axis=1)) & (npMagArrImg >= np.roll(npMagArrImg, -1, axis=1))
    NonMaxArr[mask1 & mask2] = npMagArrImg[mask1 & mask2]

    # 45 degrees
    mask1 = ((npGradArr >= 22.5) & (npGradArr < 67.5)) | ((npGradArr >= 202.5) & (npGradArr < 247.5))
    mask2 = (npMagArrImg >= np.roll(np.roll(npMagArrImg, 1, axis=0), -1, axis=1)) & (npMagArrImg >= np.roll(np.roll(npMagArrImg, -1, axis=0), 1, axis=1))
    NonMaxArr[mask1 & mask2] = npMagArrImg[mask1 & mask2]

    # 90 degrees
    mask1 = ((npGradArr >= 67.5) & (npGradArr < 112.5)) | ((npGradArr >= 247.5) & (npGradArr < 292.5))
    mask2 = (npMagArrImg >= np.roll(npMagArrImg, 1, axis=0)) & (npMagArrImg >= np.roll(npMagArrImg, -1, axis=0))
    NonMaxArr[mask1 & mask2] = npMagArrImg[mask1 & mask2]

    # 135 degrees
    mask1 = ((npGradArr >= 112.5) & (npGradArr < 157.5)) | ((npGradArr >= 292.5) & (npGradArr < 337.5))
    mask2 = (npMagArrImg >= np.roll(np.roll(npMagArrImg, -1, axis=0), -1, axis=1)) & (npMagArrImg >= np.roll(np.roll(npMagArrImg, 1, axis=0), 1, axis=1))
    NonMaxArr[mask1 & mask2] = npMagArrImg[mask1 & mask2]

    if ShowStage:
        print("*--------------Non Max Suppressed--------------*\n")

    return Image.fromarray(NonMaxArr)

def DoubleThresholding(PILImg,LowThresh =75, HighThresh = 120, ShowStage = True):
    """
    Function to check the values of the pixels and threshold to 255 or 0 based on the neighbourhood conditions in the image
    Ref: https://github.com/StefanPitur/Edge-detection---Canny-detector/blob/master/canny.py
    """
    if ShowStage: print("*============== Starting Thresholding of the image ==============*\n")
    ImgW, ImgH = PILImg.size
    img_arr = np.array(PILImg)
    ThresholdedImg = np.zeros_like(img_arr)
    for i in range(ImgH):
        for j in range(ImgW):
            x = img_arr[i, j]
            if x >= HighThresh:ThresholdedImg[i, j] = 255
            elif x < LowThresh:ThresholdedImg[i, j] = 0
            else:
                ThresholdedImg[i, j] = 0
                k_range = min(8, ImgH - i, ImgW - j)
                if k_range > 0:
                    sub_arr = img_arr[i:i+k_range, j:j+k_range]
                    if np.any(sub_arr > 0):ThresholdedImg[i, j] = 255
            if i in [0,ImgH - 1] or j in [0,ImgW - 1]:ThresholdedImg[i, j] = 0
    if ShowStage: print("*-------------- Thresholded the image --------------*\n")
    return ThresholdedImg

def AccumulateHough(NpImgArr,ThetaS=0,ThetaE=181,ThetaStep=1,vectorized=True,Debug = False,ShowStage=True):
    
    """
    This implementation of hough transform runs the vectorized version of Hough transformation using Numpy array compute methods for efficiently accumulating
    the hough transform of the canny image with reusable and pre determined cos and sin theta values and indices
    The implementation is roughly based on the skimage transform library implementation
    https://scikit-image.org/docs/dev/auto_examples/edges/plot_line_hough_transform.html
    """
    if ShowStage: print("*============== Starting Hough transform accumulation ==============*\n")
    #Setting the theta vectors for hough space 
    ThetaVec = np.deg2rad(np.arange(ThetaS,ThetaE,ThetaStep))
    #Calculating the cos and sin vectore for reuse
    CosVec,SinVec = np.cos(ThetaVec),np.sin(ThetaVec)
    #Diagonal ImageDiagLen
    ImageDiagLen = np.ceil(np.hypot(NpImgArr.shape[0],NpImgArr.shape[1])).astype(np.int_)
    #Maximmum possible rho value
    max_distance = np.ceil(2 * ImageDiagLen + 1).astype(np.int_)
    #if Debug: print(max_distance,ThetaVec.shape[0])
    HoughSpace = np.zeros((max_distance, ThetaVec.shape[0]))
    #Stars=ting the rho distance vectore for max lengths possible with number of steps in between
    RhosVec = np.linspace(-ImageDiagLen, ImageDiagLen, max_distance)

    ThetaValIndx, RhoValIndx = np.nonzero(NpImgArr)
    ImgRhoIndx, ThetaSize = ThetaValIndx.shape[0], ThetaVec.shape[0]
    if vectorized:
        # Vectorized implementation using np.add.at
        for j in range(ThetaSize):
            HoughSpace_idx = np.round(CosVec[j] * RhoValIndx + SinVec[j] * ThetaValIndx) + ImageDiagLen
            np.add.at(HoughSpace, (HoughSpace_idx.astype(int), j), 1)
    else:
        for i in range(ImgRhoIndx):
                x ,y= RhoValIndx[i],ThetaValIndx[i]
                for j in range(ThetaSize):
                    HoughSpace_idx = np.round(CosVec[j] * x + SinVec[j] * y) + ImageDiagLen
                    HoughSpace[int(HoughSpace_idx), j] += 1
    if ShowStage: print("*-------------- Accumulated Hough lines --------------* \n")
    if Debug:
        plt.figure(figsize = (20,15))
        plt.imshow(HoughSpace,'jet',extent=[-90,90,-ImageDiagLen,ImageDiagLen])
        plt.colorbar()
    return HoughSpace, RhosVec, ThetaVec

def ExtractingPeakLines(HoughSpace, thetas, num_peak=5, neighborhood_size=10,ShowStage=True):
    """
    A function to extract the peak values from the accumulator matrix and checks the neighbourhood values and 
    to suppress false detections.
    https://gist.github.com/bygreencn/6a900fd2ff5d0101473acbc3783c4d92
    """
    
    if ShowStage: print("*============== Extracting the hough space Peaks ==============*\n")
    peaks,theta_consider,epsilon,i = [],np.deg2rad(-90),np.deg2rad(5),0
    while(i<num_peak):
        idx = np.unravel_index(np.argmax(HoughSpace), HoughSpace.shape)
        if not i:
            peaks.append(idx)
            i = i+1
            theta_consider = thetas[idx[1]]
        else:
            if(thetas[idx[1]]>=theta_consider-epsilon and thetas[idx[1]]<=theta_consider+epsilon):
                peaks.append(idx)
                i = i+1
        idx_y, idx_x = idx
        for x in range((int)(idx_x-(neighborhood_size)//2),(int)(idx_x+(neighborhood_size)//2)):
            for y in range((int)(idx_y-(neighborhood_size)//2),(int)(idx_y+(neighborhood_size)//2)):
                if(x>=0 and x<HoughSpace.shape[1] and y>=0 and y<HoughSpace.shape[0]):
                    HoughSpace[y,x] = 0
    if ShowStage: print("*-------------- Extracted the hough space Peaks ==============*\n")
    return peaks

def OverlayLinesOnImg(PILImg, indicies, RhoVec, ThetaVec):
    """
    Drawing the detected lined from the peak indices in over the image using red coloed lines for the scaled values of image
    Ref: https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html
    """
    
    NewOverlay = Image.new("RGBA", PILImg.size)
    NewOverlay.paste(PILImg)
    img1 = ImageDraw.Draw(NewOverlay)

    for i in range(len(indicies)):
        rho,theta = RhoVec[indicies[i][0]],ThetaVec[indicies[i][1]]
        a,b = np.cos(theta),np.sin(theta)
        x0,y0 = a*rho,b*rho
        x1,y1 = int((x0 + 2000*(-b))),int((y0 + 2000*(a)))
        x2,y2 = int((x0 - 2000*(-b))),int((y0 - 2000*(a)))
        img1.line([(x1, y1), (x2, y2)], fill = "red")
    
    return NewOverlay

if __name__ == "__main__":
    
    if(len(sys.argv) < 2):
        raise Exception("error: please give an input image name as a parameter, like this: \n"
                     "python3 pichu_devil.py input.jpg")
    
    # Load an image 
    BaseImg = Image.open(sys.argv[1])
    if 'sample-input.png' in sys.argv[1]:
        std_devi = 1.4
        upper_thresh = 58
        lower_thresh = 24
        vectorized = True
        lines = 5
    elif 'music1.png' in sys.argv[1] or 'music2.png' in sys.argv[1] or 'music4.png' in sys.argv[1]:
        std_devi = 1
        upper_thresh = 58
        lower_thresh = 10
        vectorized = True
        lines = 10
    elif 'music3.png' in sys.argv[1]:
        std_devi = 0.3
        upper_thresh = 58
        lower_thresh = 10
        vectorized = True
        lines = 20
    elif 'rach.png' in sys.argv[1]:
        std_devi = 1
        upper_thresh = 58
        lower_thresh = 10
        vectorized = True
        lines = 50
    else:
        std_devi = 1.4
        upper_thresh = 58
        lower_thresh = 24
        vectorized = False
        lines = 5
    gaussian = ApplyGaussian(BaseImg, std_devi)
    #gaussian.save('Gaussian.png')
    sobel_image, sobel_gradient = ApplySobel(gaussian, AsPil=True)
    #sobel_image.save('Sobel_MagnitudeImg.png')
    #sobel_gradient.save('Sobel_GradientImg.png')
    suppressed_image = NonMaxSuppression(np.array(sobel_image),np.array(sobel_gradient))
    #suppressed_image.convert('RGB').save('NonMaxSuppressed_Image.png')
    threshold_image = DoubleThresholding(suppressed_image, lower_thresh, upper_thresh)
    threshold_image = Image.fromarray(threshold_image)
    #threshold_image.show()
    #threshold_image.convert('RGB').save('Double_ThreshImage.png')
    print("Canny Edge Image generated\n")
    accumulator, rhos, thetas = AccumulateHough(np.array(threshold_image),vectorized)
    #Image.fromarray(accumulator).convert('RGB').save("Hough Space.png")
    peak_indices = ExtractingPeakLines(accumulator, thetas, lines)        
    print("Lines are extracted\n")
    final_image = OverlayLinesOnImg(BaseImg, peak_indices, rhos, thetas)
    #final_image.show()
    final_image.save("output.png")
    print("Overlay image saved in the same directory\n")
