#!/usr/bin/python
 # vim: set ts=2 expandtab:
"""
Module: furigana.py
Desc: Estimate furigana in segmented raw manga scans
Author: John O'Neil
Email: oneil.john@gmail.com
DATE: Sunday, Sept 1st 2013

  Furigana is a major difficutly in running OCR
  on low resolution manga scans. This scipt attempts
  to estimate furigana sections of given (pre segmented)
  text areas. 
  
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
import clean_page as clean

parser = argparse.ArgumentParser(description='Estimate areas of furigana in segmented raw manga scan.')
parser.add_argument('infile', help='Input (color) raw Manga scan image to clean.')
parser.add_argument('segmentation_file', help='Input 3 channel segmentation of input image, with text areas in R channel.')
parser.add_argument('-o','--output', dest='outfile', help='Output (color) cleaned raw manga scan image.')
#parser.add_argument('-m','--mask', dest='mask', default=None, help='Output (binary) mask for non-graphical regions.')
#parser.add_argument('-b','--binary', dest='binary', default=None, help='Binarized version of input file.')
parser.add_argument('-v','--verbose', help='Verbose operation. Print status messages during processing', action="store_true")
parser.add_argument('--display', help='Display output using OPENCV api and block program exit.', action="store_true")
parser.add_argument('-d','--debug', help='Overlay input image into output.', action="store_true")
#parser.add_argument('--sigma', help='Std Dev of gaussian preprocesing filter.',type=float,default=None)
#parser.add_argument('--segment_threshold', help='Threshold for nonzero pixels to separete vert/horiz text lines.',type=int,default=1)
args = None

def binary_mask(mask):
  return np.array(mask!=0,'B')

def cc_center(component):
  x_center = component[1].start+(component[1].stop-component[1].start)/2
  y_center = component[0].start+(component[0].stop-component[0].start)/2
  return (x_center, y_center)

def is_in_component(row, col, component):
  return (row >= component[0].start and row <= component[0].stop
    and col >= component[1].start and col <= component[1].stop)

def cc_width(component):
  return component[1].stop-component[1].start

def intersects_other_component(row, col, component, components):
  for c in components:
    if c is component: continue
    if is_in_component(row, col, c):return c
  return None

def find_cc_to_left(component, components, max_dist=20):
  (c_col, c_row) = cc_center(component)
  left_col = c_col-int(max_dist)
  if left_col<0:left_col=0
  for col in reversed(range(left_col,c_col)):
    c = intersects_other_component(c_row, col, component, components)
    if c is not None:
      #print 'got hit from center ' + str(c_col) + ','+str(c_row) + 'at ' + str(col) + ',' + str(c_row)
      return c
  return None

def estimate_furigana(img, segmentation):
  (w,h)=img.shape[:2]

  if(args and args.verbose):
    print 'Estimateding furigana in ' + str(h) + 'x' + str(w) + ' image.'

  #gray = clean.grayscale(img)
  #text_areas = segmentation[:,:,2]
  text_areas = segmentation

  #form binary image from grayscale
  binary_threshold = 180
  if args and args.verbose:
    print 'binarizing images with threshold value of ' + str(binary_threshold)
  binary = clean.binarize(img,threshold=binary_threshold)
  #binary = binary[:,:,2]

  binary_average_size = cc.average_size(binary)
  if args and args.verbose:
    print 'average cc size for binaryized grayscale image is ' + str(binary_average_size)

  #apply mask and return images
  text_mask = binary_mask(text_areas)
  cleaned = cv2.bitwise_not(text_mask*binary)
  cleaned_average_size = cc.average_size(cleaned)
  if args and args.verbose:
    print 'average cc size for cleaned, binaryized grayscale image is ' + str(cleaned_average_size)

  columns = scipy.ndimage.filters.gaussian_filter(cleaned,(1.5*binary_average_size,0.0))
  columns = clean.binarize(columns,threshold=240)
  furigana = columns*text_mask

  #go through the columns in each text area, and:
  #1) Estimate the standard column width (it should be similar to the average connected component width)
  #2) Separate out those columns which are significantly thinner (>75%) than the standard width
  boxes = cc.get_connected_components(furigana)
  furigana_lines = []
  non_furigana_lines = []
  lines_general = []
  for box in boxes:
    line_width = cc_width(box)
    line_to_left = find_cc_to_left(box, boxes, max_dist=line_width*2.0)
    if line_to_left is None:
      non_furigana_lines.append(box)
      continue

    left_line_width = cc_width(line_to_left)
    #print 'found cc with width ' + str(line_width) + ' to the right of cc with width ' + str(left_line_width)
    if line_width < left_line_width * 0.65:
      furigana_lines.append(box)
    else:
      non_furigana_lines.append(box)

  furigana_mask = np.zeros(furigana.shape)
  for f in furigana_lines:
    furigana_mask[f]=1

  furigana = furigana * furigana_mask

  if args and args.debug:
    furigana = 0.25*(columns*text_mask) + 0.25*img + 0.5*furigana
    #cc.draw_bounding_boxes(furigana, boxes, color=255,line_size=1)

  return furigana

def estimate_furigana_from_files(filename, segmentation_filename):
  img = cv2.imread(filename)
  gray = clean.grayscale(img)
  seg = cv2.imread(segmentation_filename)
  segmentation = seg[:,:,2]
  return estimate_furigana(gray, segmentation)


if __name__ == '__main__':

  args = parser.parse_args()
  infile = args.infile
  segmentation_file = args.segmentation_file
  outfile = infile + '.furigana.png'
  if args.outfile is not None:
    outfile = args.outfile

  if not os.path.isfile(infile) or not os.path.isfile(segmentation_file):
    print 'Please provide a regular existing input file. Use -h option for help.'
    sys.exit(-1)

  if args.verbose:
    print '\tProcessing file ' + infile
    print '\tWith segmentation file ' + segmentation_file
    print '\tAnd generating output ' + outfile

  furigana = estimate_furigana_from_files(infile, segmentation_file)

  imsave(outfile,furigana)
  
  if args.display:
    cv2.imshow('Furigana', furigana)
    if cv2.waitKey(0) == 27:
      cv2.destroyAllWindows()
    cv2.destroyAllWindows()

