# Perform OCR on an image or
# images in a directory

import defaults
import arg
import ocr_dev as ocr

from imageio import imwrite
import cv2
import os
import shutil

def normalize_path(path):
    if path[-1] != '/': path=path+'/'
    return path

def copytree_(src, dst):
    extensions = ['.css', '.js']
    for f in os.listdir(src):
        if (os.path.splitext(f)[1] not in extensions):
            continue
        try:
            shutil.copy(src+f, dst)
        except OSError as e:
            os.mkdir(dst)
            shutil.copy(src+f, dst)

def save_text(texts, path):
    path = os.path.splitext(path)[0]+'.txt'
    try:
        with open(path, "w", encoding='utf-8') as txt_file:
            for text in texts:
                txt_file.write(" ".join(text) + "\n")
    except FileNotFoundError as e:
        os.mkdir(os.path.split(path)[0])
        with open(path, "w", encoding='utf-8') as txt_file:
            for text in texts:
                txt_file.write(" ".join(text) + "\n")

def save_image(img, path):
    try:
        imwrite(path, img)
    except FileNotFoundError as e:
        os.mkdir(os.path.split(path)[0])
        imwrite(path, img)

def generate_html(img_path, rects, texts, name='image'):

    text_boxes = ocr.create_textboxes(rects, texts)

    text_boxes_as_dict = '['
    for tb in text_boxes:
        text_boxes_as_dict += tb.export_as_dict() + ','
    text_boxes_as_dict = text_boxes_as_dict[:-1] + ']'

    path_to_template = defaults.PATH_TO_HTML_TEMPLATE
    f_template = open(path_to_template, 'r')
    template = f_template.read()
    f_template.close()

    anno_path = os.path.split(img_path)[0] + 'annotorious/'

    html_out = template.format(img_name=name, img_path=img_path, textBoxes=text_boxes_as_dict)

    return html_out

def save_webpage(html_out, html_path):

    with open(html_path, "w", encoding='utf-8') as txt_file:
        txt_file.write(html_out)

def get_output_directory(img_path, img_name, txt_path, txt_name, outfile):

    if (arg.is_defined('infile') and arg.is_defined('outfile')):
        img_path, img_name = os.path.split(outfile)
        img_path = normalize_path(img_path)
        txt_path = img_path
        txt_name = os.path.splitext(img_name)[0] + '.txt'

    if (arg.is_defined('inpath') and arg.is_defined('outfile')):
        img_path = normalize_path(outfile)
        txt_path = normalize_path(outfile)

    if arg.boolean_value('default_directory'):
        img_path = img_path+defaults.IMAGE_OUTPUT_DIRECTORY
        txt_path = txt_path+defaults.TEXT_OUTPUT_DIRECTORY

    return (img_path+img_name, txt_path+txt_name)
