#!/usr/bin/python
 # vim: set ts=2 expandtab:
"""
Module: connected_component_test.py
Desc:
Author: John O'Neil
Email: oneil.john@gmail.com
DATE: Friday, July 26th 2013

  Experiment to use basic scipy connected component methods
  to isolate text in raw manga sans. This leverages the
  commonality between the opencv and scipy/numpy packages
  (cv2 uses numpy arrays as a basic matrix implementation).

  The idea is to look over a given image for a typical
  (average) connected component size, and simply threshold
  on a fraction and multiple of this size. This ought to
  remove very, very small objects (tones) and very large
  or overly connected ones (figures, borders etc).
  Individual characters are generally pretty "small" and
  by definition must be disconnected with surrounding art.  
  
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

def draw_bounding_boxes(img,connected_components,max_size,min_size,color=(0,0,255),line_size=2):
  for component in connected_components:
    if area_bb(component)**0.5<min_size: continue
    if area_bb(component)**0.5>max_size: continue
    #a = area_nz(component,img)
    #if a<min_size: continue
    #if a>max_size: continue
    (ys,xs)=component[:2]
    cv2.rectangle(img,(xs.start,ys.start),(xs.stop,ys.stop),color,line_size)

if __name__ == '__main__':

  #this experiment relies upon a single input argument
  if len(sys.argv)<2:
    print 'USAGE connected_component_test.py <input image name>'
    sys.exit(-1)

  img = cv2.imread(sys.argv[1])
  (h,w,d)=img.shape

  #convert to single channel grayscale, and form scaled and unscaled binary images
  #we scale the binary image to have a copy with tones (zip-a-tones) removed
  #and we form a binary image that's unscaled for use in final masking
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  scaled = cv2.pyrUp(cv2.pyrDown(gray,dstsize=(w/2, h/2)),dstsize=(w, h))
  (binthresh,binary) = cv2.threshold(scaled, 190, 255, cv2.THRESH_BINARY_INV )
  (binthresh_gray,binary_unscaled) = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV )
  
  #scale = estimate_scale(binary)
  #Draw out statistics on average connected component size in the rescaled, binary image
  components = get_connected_components(binary)
  sorted_components = sorted(components,key=area_bb)
  #sorted_components = sorted(components,key=lambda x:area_nz(x,binary))
  areas = zeros(binary.shape)
  for component in sorted_components:
    if amax(areas[component])>0: continue
    areas[component] = area_bb(component)**0.5
    #areas[component]=area_nz(component,binary)
  average_size = median(areas[(areas>3)&(areas<100)])
  #average_size = median(areas[areas>3])
  print 'Average area of component is: ' + str(average_size)

  #find the bounding boxes of connected components of approx median size
  mask = bounding_boxes(binary,sorted_components,average_size*4,average_size*0.15)
  cleaned = cv2.bitwise_not(mask*binary)
  maskedgray = mask*gray

  #Form canny edges on initial (scaled) binary image. We do this as the contours
  #are a good discriminant for features we're interested in.
  scaled2 = scaled
  edges = cv2.Canny( scaled2, 128, 255, apertureSize=3)
  masked_edges = mask*edges
  contours,hierarchy = cv2.findContours(masked_edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

  #form a mask from our contours by computing their convex hulls.
  #then apply multiply this with our connected components areas of
  #interest.
  #The CCs are good at identifying characters of approprite size, but generally
  #leave in unwanted tone related artifacts and other noise
  #The edges also pick out characters well, but are less susceptible to tones.
  #The two masks multiplied together form an okay map of text only regions.
  temp_mask = np.zeros(cleaned.shape,np.uint8)
  for c in contours:
    #also draw detected contours into the original image in green
    cv2.drawContours(img,[c],0,(0,255,0),1)
    hull = cv2.convexHull(c)
    cv2.drawContours(temp_mask,[hull],0,255,-1)

  (btg,local_binary) = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV )
  
  #final_boxmap = compute_boxmap(temp_mask,average_size,dtype='B',threshold=(.15,4))
  final_mask = bounding_boxes(temp_mask,sorted_components,average_size*4,average_size*0.15)
  final = final_mask*local_binary
  cleaned = cv2.bitwise_not(mask*binary)

  #draw bouding boxes for all our CCs in the original image.
  temp_mask_components = get_connected_components(temp_mask)
  sorted_temp_components = sorted(temp_mask_components,key=area_bb)
  #sorted_temp_components = sorted(temp_mask_components,key=lambda x:area_nz(x,temp_mask))
  draw_bounding_boxes(img,sorted_temp_components,average_size*4,average_size*0.15)

  cv2.imshow('img',img)
  cv2.imwrite('output.jpg', img)

  if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
  cv2.destroyAllWindows()
