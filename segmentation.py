#!/usr/bin/python
 # vim: set ts=2 expandtab:
"""
Module: segmentation.py
Desc: Segment raw manga scan into text/nontext areas
Author: John O'Neil
Email: oneil.john@gmail.com
DATE: Friday, August 30th 2013

  Input a manga raw scan image.
  Output a single image with text
  areas blocked in color. 
  
"""

import numpy as np
import math
import cv2
import sys
import scipy.ndimage
from scipy.misc import imsave
import run_length_smoothing as rls
import ocr
import argparse
import os

import connected_components as cc

parser = argparse.ArgumentParser(description='Segment raw Manga scan image.')
parser.add_argument('infile', help='Input (color) raw Manga scan image to clean.')
parser.add_argument('-o','--output', dest='outfile', help='Output (color) cleaned raw manga scan image.')
#parser.add_argument('-m','--mask', dest='mask', default=None, help='Output (binary) mask for non-graphical regions.')
#parser.add_argument('-b','--binary', dest='binary', default=None, help='Binarized version of input file.')
parser.add_argument('-v','--verbose', help='Verbose operation. Print status messages during processing', action="store_true")
parser.add_argument('--display', help='Display output using OPENCV api and block program exit.', action="store_true")
parser.add_argument('-d','--debug', help='Overlay input image into output.', action="store_true")
parser.add_argument('--sigma', help='Std Dev of gaussian preprocesing filter.',type=float,default=None)
parser.add_argument('--segment_threshold', help='Threshold for nonzero pixels to separete vert/horiz text lines.',type=int,default=1)
args = None


def segment_image(img, max_scale=4.0, min_scale=0.15):
  (h,w)=img.shape[:2]

  if(args and args.verbose):
    print 'Segmenting ' + str(h) + 'x' + str(w) + ' image.'

  gray = grayscale(img)

  #create gaussian filtered and unfiltered binary images
  binary_threshold = 180
  if args and args.verbose:
    print 'binarizing images with threshold value of ' + str(binary_threshold)
  binary = binarize(gray,threshold=binary_threshold)

  binary_average_size = cc.average_size(binary)
  if args and args.verbose:
    print 'average cc size for binaryized grayscale image is ' + str(binary_average_size)
  '''
  The necessary sigma needed for Gaussian filtering (to remove screentones and other noise) seems
  to be a function of the resolution the manga was scanned at (or original page size, I'm not sure).
  Assuming 'normal' page size for a phonebook style Manga is 17.5cmx11.5cm (6.8x4.5in).
  A scan of 300dpi will result in an image about 2000x1350, which requires a sigma of 1.5 to 1.8.
  I'm encountering many smaller images that may be nonstandard scanning dpi values or just smaller
  magazines. Haven't found hard info on this yet. They require sigma values of about 0.5 to 0.7.
  I'll therefore (for now) just calculate required (nonspecified) sigma as a linear function of vertical
  image resolution.
  '''
  sigma = (1.0/676.0)*float(h)-1.3
  if args and args.sigma:
    sigma = args.sigma
  if args and args.verbose:
    print 'Applying Gaussian filter with sigma (std dev) of ' + str(sigma)
  gaussian_filtered = scipy.ndimage.gaussian_filter(gray, sigma=sigma)
  
  gaussian_binary = binarize(gaussian_filtered,threshold=binary_threshold)
  
  #Draw out statistics on average connected component size in the rescaled, binary image
  average_size = cc.average_size(gaussian_binary)
  if args and args.verbose:
    print 'Binarized Gaussian filtered image average cc size: ' + str(average_size)
  max_size = average_size*max_scale
  min_size = average_size*min_scale

  #primary mask is connected components filtered by size
  mask = cc.form_mask(gaussian_binary, max_size, min_size)

  #secondary mask is formed from canny edges
  canny_mask = form_canny_mask(gaussian_filtered, mask=mask)

  #final mask is size filtered connected components on canny mask
  final_mask = cc.form_mask(canny_mask, max_size, min_size)

  #apply mask and return images
  cleaned = cv2.bitwise_not(final_mask * binary)

  vertical_smoothing_threshold = 0.75*average_size
  horizontal_smoothing_threshold = 0.75*average_size
  if args and args.verbose:
    print 'Applying run length smoothing with vertical threshold ' + str(vertical_smoothing_threshold) \
    +' and horizontal threshold ' + str(horizontal_smoothing_threshold)
  run_length_smoothed = rls.RLSO( cv2.bitwise_not(cleaned), vertical_smoothing_threshold, horizontal_smoothing_threshold)
  components = cc.get_connected_components(run_length_smoothed)
  text = np.zeros((h,w),np.uint8)
  text_columns = np.zeros((h,w),np.uint8)
  text_rows = np.zeros((h,w),np.uint8)
  for component in components:
    seg_thresh = 2
    if args and args.segment_threshold:
      seg_thresh = args.segment_threshold
    (aspect, v_lines, h_lines) = ocr.segment_into_lines(cv2.bitwise_not(cleaned), component,min_segment_threshold=seg_thresh)
    if len(v_lines)<2 and len(h_lines)<2:continue
    
    ocr.draw_2d_slices(text,[component],color=255,line_size=-1)
    ocr.draw_2d_slices(text_columns,v_lines,color=255,line_size=-1)
    ocr.draw_2d_slices(text_rows,h_lines,color=255,line_size=-1)
  
  if args and args.debug:
    text = 0.5*text + 0.5*gray
    text_rows = 0.5*text_rows+0.5*gray
    text_colums = 0.5*text_columns+0.5*gray

  segmented_image = np.zeros((h,w,3), np.uint8)
  segmented_image[:,:,0] = text_columns
  segmented_image[:,:,1] = text_rows
  segmented_image[:,:,2] = text
  return segmented_image

def segment_image_file(filename):
  img = cv2.imread(filename)
  return segment_image(img)

def grayscale(img):
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  return gray

def binarize(img, threshold=190, white=255):
  (t,binary) = cv2.threshold(img, threshold, white, cv2.THRESH_BINARY_INV )
  return binary

def form_canny_mask(img, mask=None):
  edges = cv2.Canny(img, 128, 255, apertureSize=3)
  if mask is not None:
    mask = mask*edges
  contours,hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

  temp_mask = np.zeros(img.shape,np.uint8)
  for c in contours:
    #also draw detected contours into the original image in green
    #cv2.drawContours(img,[c],0,(0,255,0),1)
    hull = cv2.convexHull(c)
    cv2.drawContours(temp_mask,[hull],0,255,-1)
    #cv2.drawContours(temp_mask,[c],0,255,-1)
    #polygon = cv2.approxPolyDP(c,0.1*cv2.arcLength(c,True),True)
    #cv2.drawContours(temp_mask,[polygon],0,255,-1)
  return temp_mask


if __name__ == '__main__':

  args = parser.parse_args()
  infile = args.infile
  outfile = infile + '.segmented.png'
  if args.outfile is not None:
    outfile = args.outfile
  binary_outfile = infile + '.binary.png'
  #if args.binary is not None:
  #  binary_outfile = args.binary
  #mask = args.mask

  if not os.path.isfile(infile):
    print 'Please provide a regular existing input file. Use -h option for help.'
    sys.exit(-1)

  if args.verbose:
    print '\tProcessing file ' + infile
    print '\tGenerating output ' + outfile

  segmented = segment_image_file(infile)

  imsave(outfile,segmented)
  #cv2.imwrite(outfile,segmented)
  #if binary is not None:
  #  cv2.imwrite(binary_outfile, binary)
  
  if args.display:
    cv2.imshow('Segmented', segmented)
    #cv2.imshow('Cleaned',cleaned)
    #if args.mask is not None:
    #  cv2.imshow('Mask',mask)
    if cv2.waitKey(0) == 27:
      cv2.destroyAllWindows()
    cv2.destroyAllWindows()

