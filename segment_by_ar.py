#!/usr/bin/env python
# vim: set ts=2 expandtab:
"""
Module: segment_by_ar.py
Desc:
Author: John O'Neil
Email: oneil.john@gmail.com
DATE: Sunday, June 22nd 2014

  Attempt to segment character candidates via
  selecting them by ar (Japanese characters for the most
    part have an aspect ratio of about 1.0)
  
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
  components = get_connected_components(image)
  sorted_components = sorted(components,key=area_bb)
  mask = zeros(image.shape,np.uint8)#,'B')
  for component in sorted_components:
    if not filter(component):
      continue
    mask[component] = image[component]>0
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
  #parser.add_argument('-o','--output', dest='outfile', help='Output statistic file.')
  #parser.add_argument('-v','--verbose', help='Verbose operation. Print status messages during processing', action="store_true")
  #parser.add_argument('--display', help='Display output using OPENCV api and block program exit.', action="store_true")
  #parser.add_argument('--furigana', help='Attempt to suppress furigana characters which interfere with OCR.', action="store_true")
  #parser.add_argument('-d','--debug', help='Overlay input image into output.', action="store_true")
  #parser.add_argument('--sigma', help='Std Dev of gaussian preprocesing filter.',type=float,default=None)
  #parser.add_argument('--binary_threshold', help='Binarization threshold value from 0 to 255.',type=int,default=defaults.BINARY_THRESHOLD)
  #parser.add_argument('--segment_threshold', help='Threshold for nonzero pixels to separete vert/horiz text lines.',type=int,default=1)
  #parser.add_argument('--additional_filtering', help='Attempt to filter false text positives by histogram processing.', action="store_true")
  args = parser.parse_args()

  infile = args.infile

  if not os.path.isfile(infile):
    print 'Please provide a regular existing input file. Use -h option for help.'
    sys.exit(-1)

  #Load the image and make id suitable for analysis (filter and binarize)
  image = Image.open(infile).convert("L")
  gaussian_filtered = scipy.ndimage.gaussian_filter(image, sigma=1.5)
  binary = binarize(gaussian_filtered)

  #form masks via the area of connected components and their aspect ratio
  #this gives us a fairly good estimate of some subset of text candidates (>90% are text)
  area_mask = generate_mask(np.invert(binary), AreaFilter(min=10.0, max=100.0))
  ar_mask = generate_mask(area_mask, AspectRatioFilter(min=0.8, max=1.2))
  clean_mask = np.invert(ar_mask)

  #apply the size/ar filter to the original image, cleaning it
  cleaned = np.invert(np.invert(image) * np.invert(clean_mask))


  #show the steps we've followed.
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

