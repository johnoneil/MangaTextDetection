#!/usr/bin/python
# vim: set ts=2 expandtab:
"""
Module: run_length_smoothing.py
Desc:
Author: John O'Neil
Email: oneil.john@gmail.com
DATE: Thursday, August 1st 2013

  Experiment to use run length smoothing
  techniques to detect vertical or horizontal
  runs of characters in cleaned manga pages.
  
"""

import numpy as np
import cv2
import sys
import scipy.ndimage
from pylab import zeros,amax,median

def area_bb(a):
  return np.prod([max(x.stop-x.start,0) for x in a[:2]])
def area_nz(slice, image):
  return np.count_nonzero(image[slice])

def get_connected_components(image):
  s = scipy.ndimage.morphology.generate_binary_structure(2,2)
  labels,n = scipy.ndimage.measurements.label(image,structure=s)
  objects = scipy.ndimage.measurements.find_objects(labels)
  return objects  

def bounding_boxes(image,connected_components,max_size,min_size):
  mask = zeros(image.shape,'B')
  for component in connected_components:
    if area_bb(component)**.5<min_size: continue
    if area_bb(component)**.5>max_size: continue
    #a = area_nz(component,image)
    #if a<min_size: continue
    #if a>max_size: continue
    mask[component] = 1
  return mask

def cc_masks(image,connected_components,max_size,min_size):
  mask = zeros(image.shape,'B')
  for component in connected_components:
    if area_bb(component)**.5<min_size: continue
    if area_bb(component)**.5>max_size: continue
    #a = area_nz(component,image)
    #if a<min_size: continue
    #if a>max_size: continue
    mask[component] = image[component]>0
    #print str(mask[component])
  return mask

def draw_bounding_boxes(img,connected_components,color=(0,0,255),line_size=2):
  for component in connected_components:
    #if area_bb(component)**0.5<min_size: continue
    #if area_bb(component)**0.5>max_size: continue
    #a = area_nz(component,img)
    #if a<min_size: continue
    #if a>max_size: continue
    (ys,xs)=component[:2]
    cv2.rectangle(img,(xs.start,ys.start),(xs.stop,ys.stop),color,line_size)

if __name__ == '__main__':

  #this experiment relies upon a single input argument
  if len(sys.argv)<2:
    print 'USAGE run_length_encoding.py <input image name>'
    sys.exit(-1)

  img = cv2.imread(sys.argv[1])
  (h,w,d)=img.shape

  #convert to single channel grayscale, and form scaled and unscaled binary images
  #we scale the binary image to have a copy with tones (zip-a-tones) removed
  #and we form a binary image that's unscaled for use in final masking
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  #scaled = cv2.pyrUp(cv2.pyrDown(gray,dstsize=(w/2, h/2)),dstsize=(w, h))
  #(binthresh,binary) = cv2.threshold(scaled, 190, 255, cv2.THRESH_BINARY_INV )
  (binthresh_gray,binary_unscaled) = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV )
  
  #Draw out statistics on average connected component size in the rescaled, binary image
  components = get_connected_components(binary_unscaled)
  sorted_components = sorted(components,key=area_bb)
  #sorted_components = sorted(components,key=lambda x:area_nz(x,binary))
  areas = zeros(binary_unscaled.shape)
  for component in sorted_components:
    if amax(areas[component])>0: continue
    areas[component] = area_bb(component)**0.5
    #areas[component]=area_nz(component,binary)
  average_size = median(areas[(areas>3)&(areas<100)])
  #average_size = median(areas[areas>3])
  print 'Average area of component is: ' + str(average_size)

  #use multiple of average size as vertical threshold for run length smoothing
  smoothing_threshold = 1*average_size

  vertical = binary_unscaled.copy()
  (rows,cols)=vertical.shape
  print "total rows " + str(rows) + " total cols "+ str(cols)
  for row in xrange(rows):
    for col in xrange(cols):
      value = vertical.item(row,col)
      if value == 0:continue
      next_row = row+1
      while True:
        if next_row>=rows:break
        if vertical.item(next_row,col)>0 and next_row-row<=smoothing_threshold:
          for n in range(row,next_row):
            vertical.itemset(n,col,255)
          break
        if next_row-row>smoothing_threshold:break
        next_row = next_row+1

  horizontal = binary_unscaled.copy()
  (rows,cols)=horizontal.shape
  print "total rows " + str(rows) + " total cols "+ str(cols)
  for row in xrange(cols):
    for col in xrange(rows):
      value = horizontal.item(col,row)
      if value == 0:continue
      #print "row : " + str(row) + " col: " + str(col)
      next_row = row+1
      while True:
        if next_row>=cols:break
        if horizontal.item(col,next_row)>0 and next_row-row<=smoothing_threshold:
          for n in range(row,next_row):
            horizontal.itemset(col,n, 255)
            #horizontal[col,n]=255
          break
          #print 'setting white'
          #binary_unscaled[row,col]=255
        if next_row-row>smoothing_threshold:break
        next_row = next_row+1

  run_length_smoothed_or = cv2.bitwise_or(vertical,horizontal)

  components = get_connected_components(run_length_smoothed_or)
  draw_bounding_boxes(img,components)
  cv2.imshow('img',img)
  cv2.imwrite('segmented.png',img)

  cv2.imshow('run_length_smoothed_or',run_length_smoothed_or)
  cv2.imwrite('run_length_smoothed.png',run_length_smoothed_or)
  #cv2.imwrite('cleaned.png',cleaned)

  if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
  cv2.destroyAllWindows()
