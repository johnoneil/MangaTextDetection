# Perform OCR on an image or
# images in a directory

import defaults
import arg

from imageio import imwrite
import cv2
import os

def normalize_path(path):
    if path[-1] != '/': path=path+'/'
    return path

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
