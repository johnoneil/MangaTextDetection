#!/usr/bin/python
# vim: set ts=2 expandtab:
"""
Module: ocr_on_bounding_boxes.py
Desc:
Author: John O'Neil
Email: oneil.john@gmail.com
DATE: Saturday, August 10th 2013

 Run OCR on some text bounding boxes.
  
"""

import numpy as np
import cv2
import sys
import scipy.ndimage
from pylab import zeros,amax,median

import tesseract

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

def cc_filter_by_size(image,connected_components,max_size,min_size):
  filtered = []
  for cc in connected_components:
    if area_bb(cc)**0.5<min_size: continue
    if area_bb(cc)**0.5>max_size: continue
    filtered.append(cc)
  return filtered

def draw_bounding_boxes(img,connected_components,color=(0,0,255),line_size=2):
  for component in connected_components:
    #if area_bb(component)**0.5<min_size: continue
    #if area_bb(component)**0.5>max_size: continue
    #a = area_nz(component,img)
    #if a<min_size: continue
    #if a>max_size: continue
    (ys,xs)=component[:2]
    cv2.rectangle(img,(xs.start,ys.start),(xs.stop,ys.stop),color,line_size)

def draw_2d_slices(img,slices,color=(0,0,255),line_size=2):
  for entry in slices:
    vert=entry[0]
    horiz=entry[1]
    cv2.rectangle(img,(horiz.start,vert.start),(horiz.stop,vert.stop),color,line_size)

def segment_into_lines(img,color_image,filtered,average_size):
  horizontal_lines = []
  vertical_lines = []
  unk_lines = []
  for cc in filtered:
    #horizontal and vertical histogram of nonzero pixels through each section
    #just look for completely white sections first.
    (ys,xs)=cc[:2]
    w=xs.stop-xs.start
    h=ys.stop-ys.start
    x = xs.start
    y = ys.start
    aspect = float(w)/float(h)

    #detect vertical columns of non-zero pixels
    vertical = []
    start_col = xs.start
    for col in range(xs.start,xs.stop):
      count = np.count_nonzero(img[ys.start:ys.stop,col])
      #print str(count)
      if count == 0 or col==(xs.stop-1):
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
      if count == 0 or row==(ys.stop-1):
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
    if len(vertical)<2 and len(horizontal)<2:continue
    if aspect<0.5:
      #assume vertical text
      vertical_lines.extend(vertical)
      #vertical_lines.extend(horizontal)
    elif aspect>2.0:
      #assume horizontal text
      horizontal_lines.extend(horizontal)
    else:
      #try to figure out if h or v
      unk_lines.extend(horizontal)
      unk_lines.extend(vertical)
      
  return (horizontal_lines, vertical_lines, unk_lines)

def ocr_on_bounding_boxes(img, components):
  #horizontal_lines = []
  #vertical_lines = []
  #unk_lines = []
  for cc in components:
    #horizontal and vertical histogram of nonzero pixels through each section
    #just look for completely white sections first.
    (ys,xs)=cc[:2]
    
    w=xs.stop-xs.start
    h=ys.stop-ys.start
    x = xs.start
    y = ys.start
    aspect = float(w)/float(h)
    print "..............."
    print " w:" + str(w) +" h:" +str(h)+ "at: " +str(x)+","+str(y)

    #detect vertical columns of non-zero pixels
    vertical = []
    start_col = xs.start
    for col in range(xs.start,xs.stop):
      count = np.count_nonzero(img[ys.start:ys.stop,col])
      #print str(count)
      if count == 0 or col==(xs.stop-1):
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
      if count == 0 or row==(ys.stop-1):
        if start_row>=0:
          height = row-start_row
          #print 'height ' + str(height)
          horizontal.append((slice(start_row,row),slice(xs.start,xs.stop)))
          start_row=-1
      else:
        if start_row<0:
          start_row=row

    if len(vertical)<2 and len(horizontal)<2:continue
    '''if aspect<0.5:
      #assume vertical text
      vertical_lines.extend(vertical)
      #vertical_lines.extend(horizontal)
    elif aspect>2.0:
      #assume horizontal text
      horizontal_lines.extend(horizontal)
    else:
      #try to figure out if h or v
      unk_lines.extend(horizontal)
      unk_lines.extend(vertical)
    '''
    #now run OCR on this bounding box
    api = tesseract.TessBaseAPI()
    api.Init(".","jpn",tesseract.OEM_DEFAULT)
    api.SetPageSegMode(tesseract.PSM_AUTO)#PSM_SINGLECHAR)#
    gray = cv2.cv.CreateImage((w,h), 8, 1)
    #cv2.cv.SetImageROI(binary,((x,y),(width,height))
    sub = cv2.cv.GetSubRect(cv2.cv.fromarray(img), (x, y, w, h))
    #cv2.cv.copy(sub,gray)
    cv2.cv.Copy(sub,gray)
    #cv2.cv.CvtColor(cv2.cv.fromarray(img), gray, cv2.cv.CV_BGR2GRAY)
    tesseract.SetCvImage(gray, api)
    #api.SetImage("image",binary)#,w,h,0)#channel1)#,channel1)
    txt=api.GetUTF8Text()
    conf=api.MeanTextConf()
    #cv2.putText(img, str(conf), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0))
    #image=None
    print "Ocred Text: %s"%txt
    print "Cofidence Level: %d %%"%conf
    
  

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

  #use multiple of average size as vertical threshold for run length smoothing
  smoothing_threshold = 1*average_size

  vertical = binary.copy()
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

  horizontal = binary.copy()
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
          #binary[row,col]=255
        if next_row-row>smoothing_threshold:break
        next_row = next_row+1

  run_length_smoothed_or = cv2.bitwise_or(vertical,horizontal)

  components = get_connected_components(run_length_smoothed_or)

  #perhaps do more strict filtering of connected components because sections of characters
  #will not be dripped from run length smoothed areas? Yes. Results quite good.
  filtered = cc_filter_by_size(img,components,average_size*100,average_size*1)

  #Attempt to segment CCs into vertical and horizontal lines
  #(horizontal_lines, vertical_lines, unk_lines) = segment_into_lines(binary,img,filtered,average_size)
  #draw_bounding_boxes(img,horizontal_lines,color=(0,0,255),line_size=2)
  #draw_2d_slices(img, horizontal_lines, color=(0,0,255))
  #draw_bounding_boxes(img,vertical_lines,color=(0,255,0),line_size=2)
  #draw_2d_slices(img,vertical_lines,color=(0,255,0))
  #draw_bounding_boxes(img,unk_lines,color=(255,0,0),line_size=2)
  #draw_2d_slices(img,unk_lines,color=(255,0,0))
  ocr_on_bounding_boxes(binary, filtered)
  

  #draw_bounding_boxes(img,filtered)
  cv2.imshow('img',img)
  cv2.imwrite('segmented.png',img)

  cv2.imshow('run_length_smoothed_or',run_length_smoothed_or)
  cv2.imwrite('run_length_smoothed.png',run_length_smoothed_or)
  #cv2.imwrite('cleaned.png',cleaned)

  if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
  cv2.destroyAllWindows()
