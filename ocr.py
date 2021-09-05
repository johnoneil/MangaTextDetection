import pytesseract
import cv2
import numpy as np

from segmentation import dimensions_2d_slice
import defaults
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

class textBox:
    def __init__(self, x, y, w, h, text):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.text = text
    def export_as_dict(self):
        return f"{{x:{self.x},y:{self.y},w:{self.w},h:{self.h},text:\"{self.text}\"}}"

def draw_2d_slices(img,slices,color=(0,0,255),line_size=1):
    for entry in slices:
        vert=entry[0]
        horiz=entry[1]
        cv2.rectangle(img,(horiz.start,vert.start),(horiz.stop,vert.stop),color,line_size)

def segment_into_lines(img,component, min_segment_threshold=1):
    (ys,xs)=component[:2]
    w=xs.stop-xs.start
    h=ys.stop-ys.start
    x = xs.start
    y = ys.start
    aspect = float(w)/float(h)

    vertical = []
    start_col = xs.start
    for col in range(xs.start,xs.stop):
        count = np.count_nonzero(img[ys.start:ys.stop,col])
        if count<=min_segment_threshold or col==(xs.stop):
            if start_col>=0:
                vertical.append((slice(ys.start,ys.stop),slice(start_col,col)))
                start_col=-1
        elif start_col < 0:
            start_col=col

    #detect horizontal rows of non-zero pixels
    horizontal=[]
    start_row = ys.start
    for row in range(ys.start,ys.stop):
        count = np.count_nonzero(img[row,xs.start:xs.stop])
        if count<=min_segment_threshold or row==(ys.stop):
            if start_row>=0:
                horizontal.append((slice(start_row,row),slice(xs.start,xs.stop)))
                start_row=-1
        elif start_row < 0:
            start_row=row

    #we've now broken up the original bounding box into possible vertical
    #and horizontal lines.
    #We now wish to:
    #1) Determine if the original bounding box contains text running V or H
    #2) Eliminate all bounding boxes (don't return them in our output lists) that
    #   we can't explicitly say have some "regularity" in their line width/heights
    #3) Eliminate all bounding boxes that can't be divided into v/h lines at all(???)
    #also we will give possible vertical text runs preference as they're more common
    #if len(vertical)<2 and len(horizontal)<2:continue
    return (aspect, vertical, horizontal)

def create_textboxes(rects, texts):

    text_boxes = []

    for i, rect in enumerate(rects):
        x, y, w, h = dimensions_2d_slice(rect[:2])
        text_boxes.append(textBox(x,y,w,h,texts[i]))

    return text_boxes

def extend_bounding_rects(img, components):

    extended_rects = []

    for component in components:
        (y,x) = component[:2]
        x_padding = 5 #int(x.stop * 0.05)
        y_padding = 5 #int(y.stop * 0.05)

        x_start = max(0, x.start - x_padding)
        y_start = max(0, y.start - y_padding)
        x_stop = min(img.shape[1], x.stop + 2*x_padding)
        y_stop = min(img.shape[0], y.stop + 2*y_padding)

        roi = img[y_start:y_stop, x_start:x_stop]
        extended_rects.append(roi)

    return extended_rects

def filter_text(text):
    for ch in ['\n','\f','\"','\'']:
        if ch in text:
            text = text.replace(ch,' ')
    return text

def ocr_on_bounding_boxes(img, components, path=''):

    texts = []
    extended_rects = extend_bounding_rects(img, components)

    lang = defaults.OCR_LANGUAGE
    oem = 1
    psm = defaults.TEXT_DIRECTION
    config = f"-l {lang} --oem {oem} --psm {psm}"
    
    for rect in extended_rects:
        text = pytesseract.image_to_string(rect, config=(config))
        text = filter_text(text)
        texts.append(text)

    return texts

# TODO: 1. Export image to an html file and 
# annotate the page with the recognized text. (DONE)
# 2. Supress furigana when running OCR
# 3. Use 'jpn' package instead of 'jpn_vert'
# on bounding boxes that are likely to be
# read horizontally.