import pytesseract
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

# TODO 1. Export image to an html file and 
# annotate the page with the recognized text.
# 2. Supress furigana when running OCR