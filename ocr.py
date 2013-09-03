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

import numpy as np
import cv2
import sys
import scipy.ndimage
from pylab import zeros,amax,median

import tesseract

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

def segment_into_lines(img,component, min_segment_threshold=2):
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
    #print str(count)
    if count<=min_segment_threshold or col==(xs.stop-1):
      #cv2.line(color_image, (col,ys.start), (col,ys.stop),(255,255,0),1)
      if start_col>=0:
        width = col-start_col
        #print 'width ' + str(width)
        #vertical.append(img[ys.start:ys.stop,start_col:col])
        vertical.append((slice(ys.start,ys.stop),slice(start_col,col)))
        start_col=-1
    else:
      if start_col<0:
        start_col=col

  #detect horizontal rows of non-zero pixels
  horizontal=[]
  start_row = ys.start
  for row in range(ys.start,ys.stop):
    count = np.count_nonzero(img[row,xs.start:xs.stop])
    if count<=min_segment_threshold or row==(ys.stop-1):
      if start_row>=0:
        height = row-start_row
        #print 'height ' + str(height)
        horizontal.append((slice(start_row,row),slice(xs.start,xs.stop)))
        start_row=-1
    else:
      if start_row<0:
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
    api = tesseract.TessBaseAPI()
    api.Init(".","jpn",tesseract.OEM_DEFAULT)
    #handle single column lines as "vertical align" and Auto segmentation otherwise
    if len(vertical)<2:
      api.SetPageSegMode(5)#tesseract.PSM_VERTICAL_ALIGN)#PSM_AUTO)#PSM_SINGLECHAR)#
    else:
      api.SetPageSegMode(tesseract.PSM_AUTO)#PSM_SINGLECHAR)#
    api.SetVariable('chop_enable','T')
    api.SetVariable('use_new_state_cost','F')
    api.SetVariable('segment_segcost_rating','F')
    api.SetVariable('enable_new_segsearch','0')
    api.SetVariable('language_model_ngram_on','0')
    api.SetVariable('textord_force_make_prop_words','F')
    api.SetVariable('tessedit_char_blacklist', '}><L')

    x=component[1].start
    y=component[0].start
    w=component[1].stop-x
    h=component[0].stop-y
    roi = cv2.cv.CreateImage((w,h), 8, 1)
    sub = cv2.cv.GetSubRect(cv2.cv.fromarray(img), (x, y, w, h))
    cv2.cv.Copy(sub, roi)
    tesseract.SetCvImage(roi, api)
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
      tesseract.SetCvImage(roi, api)
      txt=api.GetUTF8Text()
      conf=api.MeanTextConf()
      if conf>0:
        blurb = Blurb(x, y, w, h, txt, confidence=conf)
        blurbs.append(blurb)
    '''
  return blurbs


if __name__ == '__main__':

  #this experiment relies upon a single input argument
  if len(sys.argv)<2:
    print 'USAGE ocr_on_bounding_boxes.py <input image name>'
    sys.exit(-1)

  img = cv2.imread(sys.argv[1])
  (h,w,d)=img.shape

  #convert to single channel grayscale, and form scaled and unscaled binary images
  #we scale the binary image to have a copy with tones (zip-a-tones) removed
  #and we form a binary image that's unscaled for use in final masking
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  #scaled = cv2.pyrUp(cv2.pyrDown(gray,dstsize=(w/2, h/2)),dstsize=(w, h))
  #(binthresh,binary) = cv2.threshold(scaled, 190, 255, cv2.THRESH_BINARY_INV )
  (binthresh_gray,binary) = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV )
  
  #Draw out statistics on average connected component size in the rescaled, binary image
  components = cc.get_connected_components(binary)
  sorted_components = sorted(components,key=cc.area_bb)
  #sorted_components = sorted(components,key=lambda x:area_nz(x,binary))
  areas = zeros(binary.shape)
  for component in sorted_components:
    if amax(areas[component])>0: continue
    areas[component] = cc.area_bb(component)**0.5
    #areas[component]=area_nz(component,binary)
  average_size = median(areas[(areas>3)&(areas<100)])
  #average_size = median(areas[areas>3])
  #print 'Average area of component is: ' + str(average_size)

  #use multiple of average size as vertical threshold for run length smoothing
  vertical_smoothing_threshold = 0.75*average_size
  horizontal_smoothing_threshold = 0.75*average_size

  run_length_smoothed_or = rls.RLSO(binary, horizontal_smoothing_threshold, vertical_smoothing_threshold)

  components = cc.get_connected_components(run_length_smoothed_or)

  #perhaps do more strict filtering of connected components because sections of characters
  #will not be dripped from run length smoothed areas? Yes. Results quite good.
  filtered = cc.filter_by_size(img,components,average_size*100,average_size*1)

  #Attempt to segment CCs into vertical and horizontal lines
  #(horizontal_lines, vertical_lines, unk_lines) = segment_into_lines(binary,img,filtered,average_size)
  #draw_bounding_boxes(img,horizontal_lines,color=(0,0,255),line_size=2)
  #draw_2d_slices(img, horizontal_lines, color=(0,0,255))
  #draw_bounding_boxes(img,vertical_lines,color=(0,255,0),line_size=2)
  #draw_2d_slices(img,vertical_lines,color=(0,255,0))
  #draw_bounding_boxes(img,unk_lines,color=(255,0,0),line_size=2)
  #draw_2d_slices(img,unk_lines,color=(255,0,0))
  blurbs = ocr_on_bounding_boxes(binary, filtered)
  for blurb in blurbs:
    print str(blurb.confidence)+'% :'+ blurb.text
  

  #draw_bounding_boxes(img,filtered)
  cv2.imshow('img',img)
  cv2.imwrite('segmented.png',img)

  cv2.imshow('run_length_smoothed_or',run_length_smoothed_or)
  cv2.imwrite('run_length_smoothed.png',run_length_smoothed_or)
  #cv2.imwrite('cleaned.png',cleaned)

  if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
  cv2.destroyAllWindows()
