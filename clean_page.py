#!/usr/bin/python
 # vim: set ts=2 expandtab:
"""
Module: clean_page
Desc: initial cleanup and non text removal for raw manga scan
Author: John O'Neil
Email: oneil.john@gmail.com
DATE: Sunday, August 4th 2012

  Input a manga raw scan image.
  Output three things:
  1. A binary version of the original image
  2. a binary mask where 1 is text, 0 not
  3. a cleaned version of the original image
    with the mask applied (i.e. text only)
  binary represents (as much as possible)
  text only.
  This is a tough problem, but this file
  represents my best solution so far.

  Subsequent tooling and processing should
  seqment the resultant image and masks
  for text isolation into lines and OCR. 
  
"""

import numpy as np
import cv2
import sys
import scipy.ndimage
import argparse
import os

import connected_components as cc
import arg
import defaults


def clean_page(img, max_scale=defaults.CC_SCALE_MAX, min_scale=defaults.CC_SCALE_MIN):
  #img = cv2.imread(sys.argv[1])
  (h,w,d)=img.shape

  gray = grayscale(img)

  #create gaussian filtered and unfiltered binary images
  sigma = arg.float_value('sigma',default_value=defaults.GAUSSIAN_FILTER_SIGMA)
  if arg.boolean_value('verbose'):
    print('Binarizing image with sigma value of ' + str(sigma))
  gaussian_filtered = scipy.ndimage.gaussian_filter(gray, sigma=sigma)
  binary_threshold = arg.integer_value('binary_threshold',default_value=defaults.BINARY_THRESHOLD)
  if arg.boolean_value('verbose'):
    print('Binarizing image with sigma value of ' + str(sigma))
  gaussian_binary = binarize(gaussian_filtered, threshold=binary_threshold)
  binary = binarize(gray, threshold=binary_threshold)
  
  #Draw out statistics on average connected component size in the rescaled, binary image
  average_size = cc.average_size(gaussian_binary)
  #print 'Initial mask average size is ' + str(average_size)
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
  return (cv2.bitwise_not(binary), final_mask, cleaned)

def clean_image_file(filename):
  img = cv2.imread(filename)
  return clean_page(img)

def grayscale(img):
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  #adjust histogram to maximize black/white range (increase contrast, decrease brightness)??
  #gray = cv2.equalizeHist(gray)
  return gray

def binarize(img, threshold=190, white=255):
  (t,binary) = cv2.threshold(img, threshold, white, cv2.THRESH_BINARY_INV )
  return binary

def form_canny_mask(img, mask=None):
  edges = cv2.Canny(img, 128, 255, apertureSize=3)
  if mask is not None:
    mask = mask*edges
  else:
    mask = edges
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

  parser = arg.parser
  parser = argparse.ArgumentParser(description='Clean raw Manga scan image.')
  parser.add_argument('infile', help='Input (color) raw Manga scan image to clean.')
  parser.add_argument('-o','--output', dest='outfile', help='Output (color) cleaned raw manga scan image.')
  parser.add_argument('-m','--mask', dest='mask', default=None, help='Output (binary) mask for non-graphical regions.')
  parser.add_argument('-b','--binary', dest='binary', default=None, help='Binarized version of input file.')
  parser.add_argument('-v','--verbose', help='Verbose operation. Print status messages during processing', action="store_true")
  parser.add_argument('--display', help='Display output using OPENCV api and block program exit.', action="store_true")
  arg.value = parser.parse_args()

  infile = arg.string_value('infile')
  outfile = arg.string_value('outfile',default_value=infile + '.cleaned.png')
  binary_outfile = arg.string_value('binary',default_value=infile + '.binary.png')
  mask = arg.boolean_value('mask')

  if not os.path.isfile(infile):
    print('Please provide a regular existing input file. Use -h option for help.')
    sys.exit(-1)

  if arg.boolean_value('verbose'):
    print('\tProcessing file ' + infile)
    print('\tGenerating output ' + outfile)

  (binary,mask,cleaned) = clean_image_file(infile)

  cv2.imwrite(outfile,cleaned)
  if binary is not None:
    cv2.imwrite(binary_outfile, binary)
  
  if arg.boolean_value('display'):
    cv2.imshow('Binary',binary)
    cv2.imshow('Cleaned',cleaned)
    if arg.boolean_value('mask'):
      cv2.imshow('Mask',mask)
  if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
  cv2.destroyAllWindows()

