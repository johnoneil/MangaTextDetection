#!/usr/bin/python
# vim: set ts=2 expandtab:
"""
Module: ocr
Desc:
Author: John O'Neil
Email: oneil.john@gmail.com
DATE: Saturday, August 10th 2013
      Revision: Thursday, August 15th 2013

 Run OCR on some text bounding boxes.

"""
#import clean_page as clean
import connected_components as cc
import run_length_smoothing as rls
import segmentation
import clean_page as clean
import arg
import defaults
import argparse

import numpy as np
import cv2
import sys
import os
import scipy.ndimage
from pylab import zeros,amax,median

import pytesseract

class Blurb(object):
  def __init__(self, x, y, w, h, text, confidence=100.0):
    self.x=x
    self.y=y
    self.w=w
    self.h=h
    self.text = text
    self.confidence = confidence

def draw_2d_slices(img,slices,color=(0,0,255),line_size=1):
  for entry in slices:
    vert=entry[0]
    horiz=entry[1]
    cv2.rectangle(img,(horiz.start,vert.start),(horiz.stop,vert.stop),color,line_size)

def max_width_2d_slices(lines):
  max_width = 0
  for line in lines:
    width = line[1].stop - line[1].start
    if width>max_width:
      width = max_width
  return max_width

def estimate_furigana(lines):
  max_width = max_width_2d_slices(lines)
  furigana = []
  non_furigana = []
  for line in lines:
    width = line[1].stop - line[1].start
    if width < max_width*0.5:
      furigana.append(line)
    else:
      non_furigana.append(line)
  return (furigana, non_furigana)

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

def ocr_on_bounding_boxes(img, components):

  blurbs = []
  for component in components:
    (aspect, vertical, horizontal) = segment_into_lines(img, component)
    #if len(vertical)<2 and len(horizontal)<2:continue

    #attempt to separately process furigana
    #(furigana, non_furigana) = estimate_furigana(vertical)

    '''
      from http://code.google.com/p/tesseract-ocr/wiki/ControlParams
      Useful parameters for Japanese and Chinese

      Some Japanese tesseract user found these parameters helpful for increasing tesseract-ocr (3.02) accuracy for Japanese :

      Name 	Suggested value 	Description
      chop_enable 	T 	Chop enable.
      use_new_state_cost 	F 	Use new state cost heuristics for segmentation state evaluation
      segment_segcost_rating 	F 	Incorporate segmentation cost in word rating?
      enable_new_segsearch 	0 	Enable new segmentation search path.
      language_model_ngram_on 	0 	Turn on/off the use of character ngram model.
      textord_force_make_prop_words 	F 	Force proportional word segmentation on all rows.
    '''
    #now run OCR on this bounding box
    api = pytesseract.TessBaseAPI()
    api.Init(".","jpn",pytesseract.OEM_DEFAULT)
    #handle single column lines as "vertical align" and Auto segmentation otherwise
    if len(vertical)<2:
      api.SetPageSegMode(5)#pytesseract.PSM_VERTICAL_ALIGN)#PSM_AUTO)#PSM_SINGLECHAR)#
    else:
      api.SetPageSegMode(pytesseract.PSM_AUTO)#PSM_SINGLECHAR)#
    api.SetVariable('chop_enable','T')
    api.SetVariable('use_new_state_cost','F')
    api.SetVariable('segment_segcost_rating','F')
    api.SetVariable('enable_new_segsearch','0')
    api.SetVariable('language_model_ngram_on','0')
    api.SetVariable('textord_force_make_prop_words','F')
    api.SetVariable('tessedit_char_blacklist', '}><L')
    api.SetVariable('textord_debug_tabfind','0')

    x=component[1].start
    y=component[0].start
    w=component[1].stop-x
    h=component[0].stop-y
    roi = cv2.cv.CreateImage((w,h), 8, 1)
    sub = cv2.cv.GetSubRect(cv2.cv.fromarray(img), (x, y, w, h))
    cv2.cv.Copy(sub, roi)
    pytesseract.SetCvImage(roi, api)
    txt=api.GetUTF8Text()
    conf=api.MeanTextConf()
    if conf>0 and len(txt)>0:
      blurb = Blurb(x, y, w, h, txt, confidence=conf)
      blurbs.append(blurb)

    '''
    for line in non_furigana:
      x=line[1].start
      y=line[0].start
      w=line[1].stop-x
      h=line[0].stop-y
      roi = cv2.cv.CreateImage((w,h), 8, 1)
      sub = cv2.cv.GetSubRect(cv2.cv.fromarray(img), (x, y, w, h))
      cv2.cv.Copy(sub, roi)
      pytesseract.SetCvImage(roi, api)
      txt=api.GetUTF8Text()
      conf=api.MeanTextConf()
      if conf>0:
        blurb = Blurb(x, y, w, h, txt, confidence=conf)
        blurbs.append(blurb)
    '''
  return blurbs

def main():
  parser = arg.parser
  parser = argparse.ArgumentParser(description='Basic OCR on raw manga scan.')
  parser.add_argument('infile', help='Input (color) raw Manga scan image to clean.')
  parser.add_argument('-o','--output', dest='outfile', help='Output (color) cleaned raw manga scan image.')
  parser.add_argument('-v','--verbose', help='Verbose operation. Print status messages during processing', action="store_true")
  #parser.add_argument('-d','--debug', help='Overlay input image into output.', action="store_true")
  parser.add_argument('--sigma', help='Std Dev of gaussian preprocesing filter.',type=float,default=None)
  parser.add_argument('--binary_threshold', help='Binarization threshold value from 0 to 255.',type=int,default=defaults.BINARY_THRESHOLD)
  parser.add_argument('--furigana', help='Attempt to suppress furigana characters to improve OCR.', action="store_true")
  parser.add_argument('--segment_threshold', help='Threshold for nonzero pixels to separete vert/horiz text lines.',type=int,default=defaults.SEGMENTATION_THRESHOLD)

  arg.value = parser.parse_args()

  infile = arg.string_value('infile')
  outfile = arg.string_value('outfile', default_value=infile + '.html')

  if not os.path.isfile(infile):
    print('Please provide a regular existing input file. Use -h option for help.')
    sys.exit(-1)

  if arg.boolean_value('verbose'):
    print('\tProcessing file ' + infile)
    print('\tGenerating output ' + outfile)

  img = cv2.imread(infile)
  gray = clean.grayscale(img)
  binary = clean.binarize(gray)

  segmented = segmentation.segment_image_file(infile)

  components = cc.get_connected_components(segmented)

  #perhaps do more strict filtering of connected components because sections of characters
  #will not be dripped from run length smoothed areas? Yes. Results quite good.
  #filtered = cc.filter_by_size(img,components,average_size*100,average_size*1)

  blurbs = ocr_on_bounding_boxes(binary, components)
  for blurb in blurbs:
    print(str(blurb.x)+','+str(blurb.y)+' '+str(blurb.w)+'x'+str(blurb.h)+' '+ str(blurb.confidence)+'% :'+ blurb.text)

if __name__ == '__main__':
  main()
