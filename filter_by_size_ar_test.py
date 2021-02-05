#!/usr/bin/env python
# vim: set ts=2 expandtab:
"""
Module: filter_by_size_ar_test.py
Desc:
Author: John O'Neil
Email: oneil.john@gmail.com
DATE: Thursday, June 19th 2014

  Step by step test of removing connected components first by size,
  and then by aspect ratio.
  
"""

import argparse
import os
from PIL import Image
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy
import scipy.ndimage
import scipy.stats
from pylab import zeros,amax,median
import sys

def area_bb(a):
  #return np.prod([max(x.stop-x.start,0) for x in a[:2]])
  return width_bb(a)*height_bb(a)

def width_bb(a):
  return a[1].stop-a[1].start

def height_bb(a):
  return a[0].stop-a[0].start

def area_nz(slice, image):
  return np.count_nonzero(image[slice])

def get_connected_components(image):
  s = scipy.ndimage.morphology.generate_binary_structure(2,2)
  labels,n = scipy.ndimage.measurements.label(image)#,structure=s)
  objects = scipy.ndimage.measurements.find_objects(labels)
  return objects

def generate_connected_components(image):
  s = scipy.ndimage.morphology.generate_binary_structure(2,2)
  labels, num_labels = scipy.ndimage.measurements.label(image)#,structure=s)
  slices = scipy.ndimage.measurements.find_objects(labels)
  return (labels, num_labels, slices) 

def bounding_boxes(image,connected_components,max_size,min_size):
  mask = np.zeros(image.shape,'B')#np.uint8)#'B')
  for component in connected_components:
    if area_bb(component)**.5<min_size: continue
    if area_bb(component)**.5>max_size: continue
    #a = area_nz(component,image)
    #if a<min_size: continue
    #if a>max_size: continue
    mask[component] = 1#255
  return mask

def masks(image,connected_components,max_size,min_size):
  mask = zeros(image.shape,np.uint8)#,'B')
  for component in connected_components:
    if area_bb(component)**.5<min_size: continue
    if area_bb(component)**.5>max_size: continue
    mask[component] = image[component]>0
  return mask


'''
def draw_bounding_boxes(img,connected_components,max_size=0,min_size=0,color=(0,0,255),line_size=2):
  for component in connected_components:
    if min_size > 0 and area_bb(component)**0.5<min_size: continue
    if max_size > 0 and area_bb(component)**0.5>max_size: continue
    #a = area_nz(component,img)
    #if a<min_size: continue
    #if a>max_size: continue
    (ys,xs)=component[:2]
    cv2.rectangle(img,(xs.start,ys.start),(xs.stop,ys.stop),color,line_size)
'''
def filter_by_size(image,connected_components,max_size,min_size):
  filtered = []
  for cc in connected_components:
    if area_bb(cc)<min_size: continue
    if area_bb(cc)>max_size: continue
    #if area_bb(cc)**0.5<min_size: continue
    #if area_bb(cc)**0.5>max_size: continue
    filtered.append(cc)
  return filtered

def filter_by_black_white_ratio(img,connected_components,maximum=1.0,minimum=0.0):
  filtered = []
  for component in connected_components:
    black = area_nz(component,img)
    a = area_bb(component) 
    percent_black = float(black)/float(a)
    if percent_black < minimum or percent_black > maximum:
      #print 'component removed for percent ' + str(percent_black)
      continue
    filtered.append(component)
  return filtered

def average_size(img, minimum_area=3, maximum_area=100):
  components = get_connected_components(img)
  sorted_components = sorted(components,key=area_bb)
  #sorted_components = sorted(components,key=lambda x:area_nz(x,binary))
  areas = zeros(img.shape)
  for component in sorted_components:
    #As the input components are sorted, we don't overwrite
    #a given area again (it will already have our max value)
    if amax(areas[component])>0: continue
    #take the sqrt of the area of the bounding box
    areas[component] = area_bb(component)**0.5
    #alternate implementation where we just use area of black pixels in cc
    #areas[component]=area_nz(component,binary)
  #we lastly take the median (middle value of sorted array) within the region of interest
  #region of interest is defaulted to those ccs between 3 and 100 pixels on a side (text sized)
  aoi = areas[(areas>minimum_area)&(areas<maximum_area)]
  if len(aoi)==0:
    return 0
  return np.median(aoi)

def mean_width(img, minimum=3, maximum=100):
  components = get_connected_components(img)
  sorted_components = sorted(components,key=area_bb)
  widths = zeros(img.shape)
  for c in sorted_components:
    if amax(widths[c])>0: continue
    widths[c]=width_bb(c)
  aoi = widths[(widths>minimum)&(widths<maximum)]
  if len(aoi)>0:
    return np.mean(aoi)
  return 0 

def mean_height(img, minimum=3, maximum=100):
  components = get_connected_components(img)
  sorted_components = sorted(components,key=area_bb)
  heights = zeros(img.shape)
  for c in sorted_components:
    if amax(heights[c])>0: continue
    heights[c]=height_bb(c)
  aoi = heights[(heights>minimum)&(heights<maximum)]
  if len(aoi)>0:
    return np.mean(aoi)
  return 0 

def form_mask(img, max_size, min_size):
  components = get_connected_components(img)
  sorted_components = sorted(components,key=area_bb)
  #mask = bounding_boxes(img,sorted_components,max_size,min_size)
  mask = masks(img,sorted_components,max_size,min_size)
  return mask

class AreaFilter(object):
  def __init__(self, min=10.0, max=100.0):
    self._min = min
    self._max = max
  def filter(self, component):
    if area_bb(component)**.5<self._min: return False
    if area_bb(component)**.5>self._max: return False
    return True

  def __call__(self, cc):
    return self.filter(cc)

class AspectRatioFilter(object):
  def __init__(self, min=0.9, max=1.1):
    self._min = min
    self._max = max
  def filter(self, component):
    width = width_bb(component)
    height = height_bb(component)
    if height == 0:
      return False
    aspect = float(width)/float(height)
    return aspect >= self._min and aspect <= self._max

  def __call__(self, cc):
    return self.filter(cc)

def generate_mask(image, filter):
  (labels, num_labels, components) = generate_connected_components(image)
  mask = zeros(image.shape,np.uint8)#,'B')
  for label in range(num_labels):
    two_d_slice = components[label]
    if not filter(two_d_slice):
      continue
    mask[two_d_slice] = labels[two_d_slice]==(label+1)
  return mask

def binarize(image, threshold=180):
  low_values = image <= threshold
  high_values = image > threshold
  binary = image
  binary[low_values] = 0
  binary[high_values] = 255
  return binary



def main():

  #proc_start_time = datetime.datetime.now()
  parser = argparse.ArgumentParser(description='Generate Statistics on connected components from manga scan.')
  parser.add_argument('infile', help='Input (color) raw Manga scan image to annoate.')
  args = parser.parse_args()

  infile = args.infile

  if not os.path.isfile(infile):
    print('Please provide a regular existing input file. Use -h option for help.')
    sys.exit(-1)

  image = Image.open(infile).convert("L")
  '''
  The necessary sigma needed for Gaussian filtering (to remove screentones and other noise) seems
  to be a function of the resolution the manga was scanned at (or original page size, I'm not sure).
  Assuming 'normal' page size for a phonebook style Manga is 17.5cmx11.5cm (6.8x4.5in).
  A scan of 300dpi will result in an image about 1900x1350, which requires a sigma of 1.5 to 1.8.
  I'm encountering many smaller images that may be nonstandard scanning dpi values or just smaller
  magazines. Haven't found hard info on this yet. They require sigma values of about 0.5 to 0.7.
  I'll therefore (for now) just calculate required (nonspecified) sigma as a linear function of vertical
  image resolution.
  '''
  (w,h) = image.size
  sigma = (0.8/676.0)*float(h)-0.9
  gaussian_filtered = scipy.ndimage.gaussian_filter(image, sigma=sigma)
  low_values = gaussian_filtered <= 180
  high_values = gaussian_filtered > 180
  binary = gaussian_filtered
  binary[low_values] = 0
  binary[high_values] = 255

  area_mask = generate_mask(np.invert(binary), AreaFilter(min=10.0, max=100.0))
  ar_mask = generate_mask(area_mask, AspectRatioFilter(min=0.8, max=1.2))
  clean_mask = np.invert(ar_mask)
  cleaned = np.invert(np.invert(image) * np.invert(clean_mask))

  plt.subplot(141)
  plt.imshow(image, cmap=cm.Greys_r)
  plt.subplot(142)
  plt.imshow(binary, cmap=cm.Greys_r)
  plt.subplot(143)
  plt.imshow(np.invert(area_mask), cmap=cm.Greys_r)
  plt.subplot(144)
  plt.imshow(cleaned, cmap=cm.Greys_r)

  plt.show()


if __name__ == '__main__':
  main()

