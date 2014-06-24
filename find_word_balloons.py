#!/usr/bin/env python
# vim: set ts=2 expandtab:
"""
Module: find_word_balloons.py
Desc:
Author: John O'Neil
Email: oneil.john@gmail.com
DATE: Sunday, June 22nd 2014
  
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

def generate_connected_components(image):
  s = scipy.ndimage.morphology.generate_binary_structure(2,2)
  labels, num_labels = scipy.ndimage.measurements.label(image)#,structure=s)
  slices = scipy.ndimage.measurements.find_objects(labels)
  return (labels, num_labels, slices) 

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
    if self._min and area_bb(component)**.5<self._min: return False
    if self._max and area_bb(component)**.5>self._max: return False
    #print str(area_bb(component))
    #if self._min and area_bb(component)<self._min: return False
    #if self._max and area_bb(component)>self._max: return False
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

def generate_mask(image, filter, include_contained=True):
  (labels, num_labels, components) = generate_connected_components(image)
  mask = zeros(image.shape,np.uint8)#,'B')
  for label in range(num_labels):
    two_d_slice = components[label]
    if not filter(two_d_slice):
      continue
    mask[two_d_slice] |= labels[two_d_slice]==(label+1)
    #also add nonzero pixels from all connected components ENTIRELY CONTAINED
    #by this cc's bounding box. This is an attempt to partially recover smaller
    #character components which might not be connected with the primary character
    #(i.e. marks and accent like forms)
    if include_contained:
      for l in range(num_labels):
        if l == label: continue
        other_slice = components[l]
        if contains(two_d_slice, other_slice):
          mask[other_slice] |= labels[other_slice]==(l+1)
  return mask

def binarize(image, threshold=180):
  low_values = image <= threshold
  high_values = image > threshold
  binary = image
  binary[low_values] = 0
  binary[high_values] = 255
  return binary


def contains(cc_a, cc_b):
  w = width_bb(cc_a)
  dw = w/5
  h = height_bb(cc_a)
  dh = h/5
  return cc_b[0].start>=(cc_a[0].start-dh) and cc_b[0].stop<=(cc_a[0].stop+dh) and \
    cc_b[1].start>=(cc_a[1].start-dw) and cc_b[1].stop<=(cc_a[1].stop+dw)

class ConnectedComponent(object):
  def __init__(self, index, bounding_box):
    self._index = index
    self._bounding_box = bounding_box
  @property
  def label(self):
    return self._index + 1

  @property  
  def index(self):
    return self._index

  @property
  def bounding_box(self):
    return self._bounding_box

  @staticmethod
  def area(cc):
    return area_bb(cc.bounding_box)


class BaloonCandidate(ConnectedComponent):
  def __init__(self, index, bounding_box):
    ConnectedComponent.__init__(self, index, bounding_box)
    self._characters = []

  def add_character(self, character):
    self._characters.append(character)

  @property
  def characters(self):
    return self._characters

def find_balloons(cleaned_binary, white_areas):
  balloons = []
  #sort white areas from smallest to largest. We want smallest "hits"
  (labels, num_labels, components) = generate_connected_components(white_areas)
  ccs = []
  for l in range(num_labels):
    ccs.append(ConnectedComponent(l, components[l]))
  sorted_white_areas = sorted(ccs, key=ConnectedComponent.area)
  (text_labels, text_num_labels, text_components) = generate_connected_components(np.invert(cleaned_binary))
  #sorted_white_areas = sorted(components, key=area_bb)

  for t in text_components:
    #for label in reversed(range(num_labels)):
    #print "..."
    for cc in sorted_white_areas:
      
      #for label, c in enumerate(sorted_white_areas):
      #print 'text size ' + str(area_bb(t))
      #print 'ballon size ' + str(area_bb(sorted_white_areas[label]))
      #print str(sorted_white_areas[label])
      if contains(cc._bounding_box, t):
        #print 'new '+str(sorted_white_areas[label])
        if not cc._index in [x.index for x in balloons]:
          new_balloon = BaloonCandidate(index=cc._index, bounding_box=cc._bounding_box)
          new_balloon.add_character(t)
          #print 'add char ' + str(t)
          balloons.append(new_balloon)
        else:
          old_balloon = [x for x in balloons if x.index==cc._index][0]
          #print 'add char ' + str(t) + ' to ' + str(old_balloon.bounding_box)
          old_balloon.add_character(t)
        break

  #form an image of just those white areas with text candidate hits
  mask = zeros(white_areas.shape,np.uint8)
  for b in balloons:
    #print str(len(b.characters))
    #if len(b.characters) > 1:
    #if True:
      two_d_slice = b.bounding_box
      #print b.index
      #print labels[two_d_slice]
      mask[two_d_slice] |= labels[two_d_slice]==(b.label)
  return (balloons, mask)


def main():

  #proc_start_time = datetime.datetime.now()
  parser = argparse.ArgumentParser(description='Generate Statistics on connected components from manga scan.')
  parser.add_argument('infile', help='Input (color) raw Manga scan image to annoate.')
  args = parser.parse_args()

  infile = args.infile

  if not os.path.isfile(infile):
    print 'Please provide a regular existing input file. Use -h option for help.'
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

  area_mask = generate_mask(np.invert(binary), AreaFilter(min=10.0, max=60.0))
  ar_mask = generate_mask(area_mask, AspectRatioFilter(min=0.75, max=1.25))
  clean_mask = np.invert(ar_mask)
  cleaned = np.invert(np.invert(image) * np.invert(clean_mask))

  #okay. we've got a cleaned image that has some subset of CCs, each with a high probability
  #of being a character or part of a character.
  #If there are many of these in an individual large white area, the likelihood of that area
  #being a word balloon or caption is very high.
  #1 Generate a binary image off the final cleaned mask
  cleaned_binary = np.invert(np.invert(binary) * np.invert(clean_mask))

  #2 find all ares of white (or just large ones?)
  white_areas = generate_mask(binary, AreaFilter(min=60.0, max=None), include_contained=False)

  #3 isolate white areas which have a significant number of character candidates witin them
  candidate_balloons, candidate_ballons_mask = find_balloons(cleaned_binary, white_areas)

  plt.subplot(141)
  plt.imshow(image, cmap=cm.Greys_r)
  plt.subplot(142)
  plt.imshow(cleaned_binary, cmap=cm.Greys_r)
  plt.subplot(143)
  plt.imshow(white_areas, cmap=cm.Greys_r)
  plt.subplot(144)
  plt.imshow(candidate_ballons_mask, cmap=cm.Greys_r)

  plt.show()


if __name__ == '__main__':
  main()

