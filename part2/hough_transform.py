import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from scipy import stats

class HoughTransform:

    def __init__(self, theta_s=0, theta_e=181, theta_step=1, vectorized=True):
        self.theta_s = theta_s 
        self.theta_e = theta_e 
        self.theta_step = theta_step
        self.vectorized = vectorized

    def accumulate_hough(self, NpImgArr,ThetaS=0,ThetaE=181,ThetaStep=1,vectorized=True,Debug = False,ShowStage=True):
        
        """
        This implementation of hough transform runs the vectorized version of Hough transformation using Numpy array compute methods for efficiently accumulating
        the hough transform of the canny image with reusable and pre determined cos and sin theta values and indices
        The implementation is roughly based on the skimage transform library implementation
        https://scikit-image.org/docs/dev/auto_examples/edges/plot_line_hough_transform.html
        """
        if ShowStage: print("*==============Starting Hough transform accumulation==============*\n")
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
            ImgRhoIndx,ThetaSize = ThetaValIndx.shape[0],ThetaVec.shape[0]
            for i in range(ImgRhoIndx):
                    x ,y= RhoValIndx[i],ThetaValIndx[i]
                    for j in range(ThetaSize):
                        HoughSpace_idx = np.round(CosVec[j] * x + SinVec[j] * y) + ImageDiagLen
                        HoughSpace[int(HoughSpace_idx), j] += 1
        if ShowStage: print("*--------------Accumulated Hough lines--------------* \n")
        if Debug:
            plt.figure(figsize = (20,10))
            plt.imshow(HoughSpace,'jet',extent=[-90,90,-ImageDiagLen,ImageDiagLen])
            plt.colorbar()
        return HoughSpace, RhosVec, ThetaVec

    def hough_peaks(self, H, thetas, num_peak=5, neighborhood_size=10):
        """
        A function to extract the peak values from the accumulator matrix and checks the neighbourhood values and 
        to suppress false detections.
        https://gist.github.com/bygreencn/6a900fd2ff5d0101473acbc3783c4d92
        """

        peaks,theta_consider,epsilon,i = [],np.deg2rad(-90),np.deg2rad(5),0
        while(i<num_peak):
            idx = np.unravel_index(np.argmax(H), H.shape)
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
                    if(x>=0 and x<H.shape[1] and y>=0 and y<H.shape[0]):
                        H[y,x] = 0
        return peaks

    def overlay_hough_lines(self, PILImg, indicies, RhoVec, ThetaVec):
        """
        Drawing the detected lined from the peak indices in over the image using red coloed lines for the scaled values of image
        Ref: https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html
        """

        NewOverlay = Image.new("RGBA", PILImg.size)
        NewOverlay.paste(PILImg)
        img1 = ImageDraw.Draw(NewOverlay)
        lines = []

        for i in range(len(indicies)):
            rho,theta = RhoVec[indicies[i][0]],ThetaVec[indicies[i][1]]
            a,b = np.cos(theta),np.sin(theta)
            x0,y0 = a*rho,b*rho
            x1,y1 = int((x0 + 2000*(-b))),int((y0 + 2000*(a)))
            x2,y2 = int((x0 - 2000*(-b))),int((y0 - 2000*(a)))
            img1.line([(x1, y1), (x2, y2)], fill = "red")
            lines.append([[x1, y1], [x2, y2]])
        #rgbimg.show()
        NewOverlay.save("output.png")
        return NewOverlay, lines
    
    def get_horizontal_lines(self, lines):
        # a way to select only the horizontal lines
        horizontal_lines = []
        for line in lines:
            (x1, y1), (x2, y2) = line 
            if abs(y1 - y2) < 2:
                horizontal_lines.append(line)

        horizontal_lines.sort()

        return horizontal_lines
    
    def find_line_spacing(self, peaks):
        # Extract the y-coordinates of the peaks
        y_coords = [peak[0][1] for peak in peaks]
        
        # Compute the difference between successive y-coordinates
        spacings = [abs(y_coords[i+1] - y_coords[i]) for i in range(len(y_coords)-1)]
        
        # Sort the spacings in increasing order
        spacings.sort()
        space = stats.mode(spacings).mode
        
        return spacings, space
    
    def find_first_lines(self, lines, space):
        firstLines = [lines[0]]
        currentLine = lines[0]
        for i in range(1, len(lines)):
            if lines[i][0][1] - currentLine[0][1] > space * 2:
                firstLines.append(lines[i])
            currentLine = lines[i]

        first_lines = [l[0][1] for l in firstLines]

        return first_lines
    
    def hough_transform(self, image, base_image, num_lines):
        accumulator, rhos, thetas = self.accumulate_hough(image, self.vectorized)
        peak_indices = self.hough_peaks(accumulator, thetas, num_lines)
        final_image, final_lines = self.overlay_hough_lines(base_image, peak_indices, rhos, thetas)
        return final_image, final_lines


