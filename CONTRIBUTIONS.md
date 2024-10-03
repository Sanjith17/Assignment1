# Part 0
First we started with working on FFT of the image as that's where we do the processing. First we tried with low pass filter removing the area that has the noise
(Hi) and area greater than that. We were able to remove the noise. But the image was not having sharp edges and blurry. Then we tried with Band Stop filter in which we
stopped only the band that had noise in it. With this we were able to remove the noise and the image was also sharper compared to Low pass way. But we got black spots in 
the image. Now ewe removed only the Noise bu finding the co-ordinates of the area. We got a satisfying ouput from this technique.
In a trial to improve to performance, we tried to apply mean, median and gaussian with different sizes of boxes but the images was not expected and noise was not completely 
removed. 

All this things are done by Dhanyasree
 
# Part 1
Balajee referred to many sources to find the techniques to implement hough transform to find lines in the image. Initially he applied just Laplacian Filter to the image and 
performed hough transform on that. Sanjith referred to sources to extract the lines from the transform. He initially tried using a threshold value to find the points that 
occur more times and tried to plot lines from the extracted rhos and thetas. But this dint give proper output because of the font present in the background which also 
has considerable amount of count in the accumulator. The he referred to some sources and created a new function to extract lines which takes in the number of lines that are 
to be extracted from the images as a parameter. This also was not working well.
Then Balajee thought that there might be some problem in preprocessing and implemented Gaussian Blur before Laplacian filter to reduce the noise and blurr the edges in the
image. Then, we repeated all the other functions. The output improved but not good. Then Abhiroop tried using thresholding the filter output and repeated the same. Then Sanjith 
implemented Sobel function instead of Laplacian and thresholding the magnitude of the result and did the same. There lines were getting identified but one line was 
getting plotted at a single line. Then Balajee implemented Gradient in Sobel as well with magnitude plus Non Maximum Supression to overcome this problem and the performance improved. Then Abhiroop did fine tuning of threshold 
frequencies and standard deviation (for Gaussian Blur) and the code was working for the "sample-input.png" and 5 lines are getting plotted. Then this was applied on other music images (present in part 2). Each image has different parameterws (magic ones) that are to be checked by runnig the function over images and fixed.

# Part 2
