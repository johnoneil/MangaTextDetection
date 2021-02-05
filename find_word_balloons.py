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
import sys
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
    #if self._min and area_bb(component)**.5<self._min: return False
    #if self._max and area_bb(component)**.5>self._max: return False
    #print str(area_bb(component))
    if self._min and area_bb(component)<self._min: return False
    if self._max and area_bb(component)>self._max: return False
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

class HoleFilter(object):
  def __init__(self, hole_candidate_ccs, min_holes=1,):
    self._hole_candidate_ccs = hole_candidate_ccs
    self._min_holes = min_holes

  def filter(self, component):
    hole_count=0
    for h in self._hole_candidate_ccs:
      if component.contains(h):
        hole_count+=1
        if hole_count >= self._min_holes:
          return True
    return False

    #return aspect >= self._min and aspect <= self._max

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


def contains(cc_a, cc_b, ddw=5, ddh=5):
  w = width_bb(cc_a)
  dw = 0
  if ddw>0:
    dw=w/ddw
  h = height_bb(cc_a)
  dh = 0
  if ddh>0:
    dh=h/ddh
  return cc_b[0].start>=(cc_a[0].start-dh) and cc_b[0].stop<=(cc_a[0].stop+dh) and \
    cc_b[1].start>=(cc_a[1].start-dw) and cc_b[1].stop<=(cc_a[1].stop+dw)

class ConnectedComponent(object):
  def __init__(self, index, bounding_box, labels):
    self._index = index
    self._bounding_box = bounding_box
    self._labels = labels
  @property
  def label(self):
    return self._index + 1

  @property  
  def index(self):
    return self._index

  @property
  def bounding_box(self):
    return self._bounding_box

  @property
  def labels(self):
    return self._labels

  @staticmethod
  def area(cc):
    return area_bb(cc.bounding_box)

  def contains(self, cc_b):
    w = width_bb(self.bounding_box)
    dw = w/5
    h = height_bb(self.bounding_box)
    dh = h/5
    return cc_b[0].start>=(self.bounding_box[0].start-dh) and cc_b[0].stop<=(self.bounding_box[0].stop+dh) and \
      cc_b[1].start>=(self.bounding_box[1].start-dw) and cc_b[1].stop<=(self.bounding_box[1].stop+dw)


class BalloonCandidate(ConnectedComponent):
  def __init__(self, index, bounding_box, labels):
    ConnectedComponent.__init__(self, index, bounding_box, labels)
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
    ccs.append(ConnectedComponent(l, components[l], labels))
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
      if cc.contains(t):
        #print 'new '+str(sorted_white_areas[label])
        if not cc._index in [x.index for x in balloons]:
          new_balloon = BalloonCandidate(index=cc._index, bounding_box=cc._bounding_box, labels=labels)
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
    #if len(b.characters) > 10:
    #if True:
      two_d_slice = b.bounding_box
      #print b.index
      #print labels[two_d_slice]
      #mask[two_d_slice] |= labels[two_d_slice]==(b.label)
      #TODO: Use stored labels in CC class
      mask[two_d_slice] |= b.labels[two_d_slice]==(b.label)
  return (balloons, mask)

def holes_mask(candidate_balloons, binary):
  #(labels, num_labels, holes) = generate_connected_components(binary)
  mask = zeros(binary.shape, np.uint8)
  for b in candidate_balloons:
    two_d_slice = b.bounding_box
    print(str(b.labels[two_d_slice]))
    mask[two_d_slice] |= b.labels[two_d_slice]==(b.label)
  return (candidate_balloons, mask)

def generate_holes_mask(candidate_balloons, binary):
  (labels, num_labels, holes) = generate_connected_components(binary)
  final_balloons = []
  for b in candidate_balloons:
    #print 'cb char: ' + str(len(candidate_balloons.characters))
    for t in b.characters:
      for h in range(num_labels):
        #if contains(t, h):
        #if contains(b.bounding_box, holes[h], ddw=0, ddh=0):
        if contains(t, holes[h], ddw=0, ddh=0):
          final_balloons.append(b)
          #final_balloons.append(ConnectedComponent(index=h, labels=labels, bounding_box=holes[h]))
  #form an image of just those white areas with text candidate hits
  mask = zeros(binary.shape,np.uint8)
  for b in final_balloons:
    #print str(len(b.characters))
    #if len(b.characters) > 10:
    #if True:
      two_d_slice = b.bounding_box
      #print b.index
      #print labels[two_d_slice]
      mask[two_d_slice] |= b.labels[two_d_slice]==(b.label)
      #mask[two_d_slice] |= b.labels[two_d_slice] #==(b.label)
      #mask[two_d_slice] |= b.labels[two_d_slice]!=(b.label)
      #mask[two_d_slice] |= b.labels[two_d_slice]==False
  return (final_balloons, mask)



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

  min_text_area = (float(h)/140.0)**2.0
  max_text_area = (float(h)/30.0)**2.0
  print("min " + str(min_text_area))
  print("max " + str(max_text_area))

  area_mask = generate_mask(np.invert(binary), AreaFilter(min=min_text_area, max=max_text_area))
  ar_mask = generate_mask(area_mask, AspectRatioFilter(min=0.75, max=1.15))
  clean_mask = np.invert(ar_mask)
  cleaned = np.invert(np.invert(image) * np.invert(clean_mask))

  #okay. we've got a cleaned image that has some subset of CCs, each with a high probability
  #of being a character or part of a character.
  #If there are many of these in an individual large white area, the likelihood of that area
  #being a word balloon or caption is very high.
  #1 Generate a binary image off the final cleaned mask
  cleaned_binary = np.invert(np.invert(binary) * np.invert(clean_mask))

  #2 find all ares of white (or just large ones?)
  white_areas = generate_mask(binary, AreaFilter(min=max_text_area, max=None), include_contained=False)

  #3 isolate white areas which have a significant number of character candidates witin them
  candidate_balloons, candidate_ballons_mask = find_balloons(cleaned_binary, white_areas)

  #4 Keep all balloon candidates whose contained possible characters posess "holes". A much
  #larger number of kanji candidates (and some kana) have holes. Few(er) noise strokes do.
  #balloons_with_holes, balloons_with_holes_mask = find_balloons_with_holes(candidate_balloons, image, binary)
  #balloons_with_holes_mask = generate_mask(candidate_ballons_mask, HolesFilter())
  
  (final_balloons, balloons_with_holes_mask) = generate_holes_mask(candidate_balloons, binary)
  #(final_balloons, balloons_with_holes_mask) = holes_mask(candidate_balloons, binary)

  #4 for each balloon candidate, attempt to find character candidates
  #characters, character_mask = find_characters_in_balloon_candidates(candidate_balloons)



  plt.subplot(241)
  plt.imshow(image, cmap=cm.Greys_r)
  plt.subplot(242)
  plt.imshow(binary, cmap=cm.Greys_r)
  plt.subplot(243)
  plt.imshow(cleaned_binary, cmap=cm.Greys_r)
  plt.subplot(244)
  plt.imshow(white_areas, cmap=cm.Greys_r)
  plt.subplot(245)
  plt.imshow(candidate_ballons_mask, cmap=cm.Greys_r)
  plt.subplot(246)
  plt.imshow(balloons_with_holes_mask, cmap=cm.Greys_r)

  plt.show()


if __name__ == '__main__':
  main()

