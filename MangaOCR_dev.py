#!/usr/bin/python

import image_io as imgio
import connected_components as cc
import clean_page as clean
import ocr_dev as ocr
import segmentation as seg
import furigana
import defaults

import numpy as np
import cv2

import arg
import argparse
import sys
import os

def process_image(infile, outfile, path):

    path = imgio.normalize_path(path)
    img = cv2.imread(path+infile)
    img_copy = img.copy()
    gray = clean.grayscale(img)

    binary_threshold=arg.integer_value('binary_threshold',default_value=defaults.BINARY_THRESHOLD)
    if arg.boolean_value('verbose'):
        print('Binarizing with threshold value of ' + str(binary_threshold))
    inv_binary = cv2.bitwise_not(clean.binarize(gray, threshold=binary_threshold))

    segmented_image = seg.segment_image(inv_binary)
    segmented_image = segmented_image[:,:,2]

    components = cc.get_connected_components(segmented_image)
    cc.draw_bounding_boxes(img,components,color=(255,0,0),line_size=2)

    img_out, txt_out = imgio.get_output_directory(path, infile, path, os.path.splitext(infile)[0] + '.txt', outfile)
    texts = ocr.ocr_on_bounding_boxes(img_copy, components)

    imgio.save_image(img, img_out)
    imgio.save_text(texts, txt_out)



def main():

    # command to run
    # python MangaOCR_dev.py -i './test/test.jpg' -o './test/out.png' --additional_filtering

    parser = arg.parser
    parser = argparse.ArgumentParser(description='Generate text file containing OCR\'d text.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-i', '--infile', help='Input image to annotate.', dest='infile', type=str)
    group.add_argument('-p', '--inpath', help='Path to directory containing image(s)', dest='inpath', type=str)
    parser.add_argument('-o','--output', help='Output file or filepath.', dest='outfile')
    parser.add_argument('-s','--scheme', help='Output naming scheme. Appended to input filename', dest='scheme', type=str, default='_text')
    parser.add_argument('-v','--verbose', help='Verbose operation. Print status messages during processing', action="store_true")
    #parser.add_argument('-d','--debug', help='Annotate bounding boxes with numbers.', action="store_true")
    parser.add_argument('--furigana', help='Attempt to suppress furigana characters which interfere with OCR.', action="store_true")
    parser.add_argument('--sigma', help='Std Dev of gaussian preprocesing filter.',type=float,default=None)
    parser.add_argument('--binary_threshold', help='Binarization threshold value from 0 to 255.',type=int,default=defaults.BINARY_THRESHOLD)
    parser.add_argument('--additional_filtering', help='Attempt to filter false text positives by histogram processing.', action="store_true")
    parser.add_argument('--default_directory', help='Store output in predefined folders.', action="store_true")
    arg.value = parser.parse_args()

    infile = arg.string_value('infile')
    inpath = arg.string_value('inpath')
    scheme = arg.string_value('scheme')
    outfile = arg.string_value('outfile')

    if os.path.isfile(infile):
        if arg.boolean_value('verbose'):
            print('File exists. Performing OCR . . .')
        path_, infile_ = os.path.split(infile)
        process_image(infile_, outfile, path_)
        sys.exit(-1)

    infiles = os.listdir(inpath)

    if infiles:
        if arg.boolean_value('verbose'):
            print('Non-empty directory. Attempting to perform ocr on all files . . .')
        for infile_ in infiles:
            try:
                outfile_ = outfile
                process_image(infile_, outfile_, inpath)
            except AttributeError as e:
                if arg.boolean_value('verbose'):
                    print('Input file \"', infile_, '\" is not an image', sep='')
        sys.exit(-1)

    # More error handling
    if not (os.path.isfile(infile) or inpath):
        print('Please provide a regular existing input file. Use -h option for help.')
        sys.exit(-1)
    if not (infiles):
        print('Directory is empty. Place images on the desired folder. Use -h option for help.')
        sys.exit(-1)

if __name__ == '__main__':
    main()