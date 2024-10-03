import numpy as np
from PIL import Image 

class TemplateMatcher:
    # Reference: https://github.com/yash120394/Computer-Vision/tree/master/Optical%20Music%20Recognition

    def __init__(self, template):
        self.template = np.array(Image.open(template).convert('L'))

    def rescale_template(self, new_height, min_max=True):
        height, width = self.template.shape
        new_width  = int(new_height * (width / height))
        size = new_height, new_width

        rescaled = []
        for h in range(size[0]):
            row = []
            for w in range(size[1]):
                row.append(self.template[int(height * h / size[0])][int(width * w / size[1])])
            rescaled.append(row)

        rescaled = np.array(rescaled)
        if min_max:
            rescaled = self.min_max_norm(rescaled, 1)
        
        self.template = rescaled

    def pad_image(self, image, h, w):
        height, width = image.shape
        padded = np.zeros((height + h - 1, width + w - 1))
        padded[:height, :width] = image 
        return padded

    def min_max_norm(self, image, max_value=255):
        return max_value * (image - image.min()) / (image.max() - image.min())

    def match_template(self, image):
        height, width = image.shape
        template_h, template_w = self.template.shape
        padded = self.pad_image(image, template_h, template_w)
        matched = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                patch = padded[i:template_h+i,j:template_w+j]
                matched[i,j] = np.sum(np.multiply(patch, self.template)) + np.sum(np.multiply((1-patch),(1-self.template)))
        
        matched_normalized = self.min_max_norm(matched)
        matched_normalized = matched_normalized.astype(np.uint8)
        return matched_normalized

    def find_template_position(self, image): 
        size = self.template.shape
        height, width = image.shape
        x, y, r = list(), list(), 0
        while r < height:
            flag, c = False, 0
            while c < width:
                if image[r][c] == 255:
                    x.append(r)
                    y.append(c)
                    flag = True
                    c += size[1]
                else:
                    c += 1
            if flag:
                r += size[0]
            else:
                r += 1
        return x, y

