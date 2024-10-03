# reference: https://github.com/yash120394/Computer-Vision/tree/master/Optical%20Music%20Recognition

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from PIL import Image,ImageFilter,ImageDraw
import matplotlib.pyplot as plt
from scipy import stats

from process import process_image
from hough_transform import HoughTransform
from template_matching import TemplateMatcher
from note_detection import NoteDetector
from process import apply_thresholds

from config import CONFIG

def overlay_notes(image, space, notes, x, y, plot_notes=True, color='red', name='eighth'):
    draw = ImageDraw.Draw(image)
    if plot_notes:
        for i in range(len(notes)):
            x1 = int(notes[i][0] - space/2)
            y1 = notes[i][1]
            x2 = int(notes[i][0] - space/2) + matcher1.template.shape[0]
            y2 = notes[i][1] + matcher1.template.shape[1]
            draw.rectangle(((y1, x1),(y2, x2)), outline=color, width=2)
            draw.text((y1-10, x1-10), str(notes[i][2]), fill=color)
    else:
        for i in range(len(x)):    
            x1 = x[i]
            y1 = y[i]
            x2 = x[i] + matcher2.template.shape[0]
            y2 = y[i] + matcher2.template.shape[1]
            draw.rectangle(((y1, x1),(y2, x2)), outline=color, width=2)
            draw.text((y1-10, x1-10), name, fill=color)

    return image

if __name__ == "__main__":
    
    if(len(sys.argv) < 2):
        raise Exception("error: please give an input image name as a parameter, like this: \n"
                     "python3 omr.py input.jpg")
    
    base_image = Image.open(sys.argv[1]).convert('L')
    image_name = sys.argv[1].split('/')[-1]
    _config = CONFIG[image_name]
    processed = process_image(base_image, _config)
    hough_transform = HoughTransform(vectorized=_config['vectorized'])

    final_image, final_lines = hough_transform.hough_transform(processed, base_image, num_lines=_config['lines'])
    final_horizontal_lines = hough_transform.get_horizontal_lines(final_lines)
    spacings, space = hough_transform.find_line_spacing(final_horizontal_lines)
    first_lines = hough_transform.find_first_lines(final_horizontal_lines, space)

    template1 = './template1.png'
    template2 = './template2.png'
    template3 = './template3.png'

    matcher1 = TemplateMatcher(template1)
    matcher2 = TemplateMatcher(template2)
    matcher3 = TemplateMatcher(template3)

    matcher1.rescale_template(space[0], min_max=True)
    matcher2.rescale_template(3 * space[0], min_max=True)
    matcher3.rescale_template(int(2.5 * space[0]), min_max=True)

    image = np.array(base_image)
    image = (image - image.min()) / (image.max() - image.min())

    image_match_1 = matcher1.match_template(image)
    image_match_2 = matcher2.match_template(image)
    image_match_3 = matcher3.match_template(image)

    image_match_1 = apply_thresholds(Image.fromarray(image_match_1).convert('L'), 220, 220)
    image_match_2 = apply_thresholds(Image.fromarray(image_match_2).convert('L'), 250, 250)
    image_match_3 = apply_thresholds(Image.fromarray(image_match_3).convert('L'), 245, 245)

    match1_count = np.count_nonzero(image_match_1 == 255)
    match2_count = np.count_nonzero(image_match_2 == 255)
    match3_count = np.count_nonzero(image_match_3 == 255)

    x1, y1 = matcher1.find_template_position(image_match_1)
    x2, y2 = matcher2.find_template_position(image_match_2)
    x3, y3 = matcher3.find_template_position(image_match_3)

    image_for_notes = np.zeros(shape = image_match_1.shape)
    for i in range(len(x1)):
        if(int(x1[i] + space/2) < image_for_notes.shape[0]):
            image_for_notes[int(x1[i]+space/2), y1[i]] = 255
        else:
            image_for_notes[x1[i], y1[i]] = 255
            
    note_detector = NoteDetector(space, first_lines)
    notes = note_detector.detect_notes(image_for_notes)

    print(notes)

    confidence1 = match1_count / (len(x1) * matcher1.template.shape[0] * matcher1.template.shape[1])
    confidence2 = match2_count / (len(x2) * matcher2.template.shape[0] * matcher2.template.shape[1])
    confidence3 = round(match3_count / (len(x3) * matcher3.template.shape[0] * matcher3.template.shape[1]), 5)

    confidences = np.array([confidence1, confidence2, confidence3]) 
    normalized_confidences = confidences / np.sqrt(np.sum(confidences ** 2))

    image = base_image.convert('RGB')
    overlay_notes(image, space, notes, x1, y1, plot_notes=True, color='red')
    overlay_notes(image, space, notes, x2, y2, plot_notes=False, color='green', name='quarter')
    overlay_notes(image, space, notes, x3, y3, plot_notes=False, color='blue', name='eighth')
    image.save('detected.png')

    # taken from # reference: https://github.com/yash120394/Computer-Vision/tree/master/Optical%20Music%20Recognition
    detect_temp1 = pd.DataFrame(data=notes, columns=['row','col', 'symbol_type'])
    detect_temp1['height'] = matcher1.template.shape[0]
    detect_temp1['width'] = matcher2.template.shape[1]
    detect_temp1['confidence'] = normalized_confidences[0]
    detect_temp1['row'] = detect_temp1['row'] - space/2 
    detect_temp1['row'] = detect_temp1['row'].astype('int')

    detect_temp2 = pd.DataFrame({'row': x2,'col': y2})
    detect_temp2['symbol_type'] = 'quarter_rest'
    detect_temp2['height'] = matcher2.template.shape[0]
    detect_temp2['width'] = matcher2.template.shape[1]
    detect_temp2['confidence'] = normalized_confidences[1]

    detect_temp3 = pd.DataFrame({'row': x3,'col': y3})
    detect_temp3['symbol_type'] = 'eighth_rest'
    detect_temp3['height'] = matcher3.template.shape[0]
    detect_temp3['width'] = matcher3.template.shape[1]
    detect_temp3['confidence'] = normalized_confidences[2]

    # concatenate three dataframes of respective template into one
    detect_temp = pd.concat([detect_temp1, detect_temp2, detect_temp3], axis=0)
    detect_temp = detect_temp[['row','col','height','width','symbol_type','confidence']]

    detect_temp.to_csv('detected.txt', header=False, index=False, sep='\t')

        







