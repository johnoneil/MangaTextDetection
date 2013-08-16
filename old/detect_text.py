#!/usr/bin/python
 # vim: set ts=2 expandtab:
"""
Module: detect_text
Desc:
Author: John O'Neil
Email: oneil.john@gmail.com

  Experiment to localize text (extract bounding boxes)
  in typical raw manga scans.
  I'm focussing on juse standard font sizes at present
  to simplify things, and only text against white backgrounds.

  Results are still very primitive but better than I first
  thought possible.
  Basic morphological approach is taken from Bloomberg and Vincent:
  http://static.googleusercontent.com/external_content/untrusted_dlcp/research.google.com/en/us/pubs/archive/36668.pdf
  http://www.vincent-net.com/luc/papers/10wiley_morpho_DIAapps.pdf

  TODO:
  1. Prepare input scans by normalizing the background (very necessary)
  2. Implement image deskew to improve vertical/horizontal and gutter
  text detection.

  INFO:
  for a typical manga page of 1354x2029 pixels (a typical scan of 150/160ppi)
  character vertical and horizontal strokes will generally be about 4 px wide
  (that's for a normal font). for bolder fonts, they will be about 8 or 9 pixels.
  Normal font spacing (vertical dialog text) will be about 5px (maybe as large as 7px).
  Text can approach the borders of a panel quite close, but rarely less than 4px.
  Typically text is at least a full inter-character spacing from border (5-7px)
  'Typical' horizontal gutters will be about 15/16 pixels across
  'Typical' vertical gutters will be comparatively large, maybe more than 3x
   the vertical gutter width.
  
  The sheer variety of panel layouts, border styles, and borderless/balloonless
  techniques has discouraged me from trying to segment images into "panels"
  or "baloons." IMHO there are too many situations in which they don't exist.
  I'm focusing on just text localization here-- if there's text of a typical
  size on the given scan I want to find it and draw a bounding box.
  
"""

import numpy as np
import cv2
import sys

if __name__ == '__main__':

  #this experiment relies upon a single input argument
  if len(sys.argv)<2:
    print 'USAGE detect_text.py <input image name>'
    sys.exit(-1)

  #for a typical manga page of 1354x2029 pixels (a typical scan of 150/160ppi)
  #character vertical and horizontal strokes will generally be about 4 px wide
  #(that's for a normal font). for bolder fonts, they will be about 8 or 9 pixels

  #convert image to grayscale, and do simple thresholding
  # I've found so far that adaptive thresholding is less useful here. That could be wrong.
  img = cv2.imread(sys.argv[1])
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  #binary = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,0)
  (binthresh,binary) = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY )

  #get image geometry
  (height, width, depth) = img.shape
  print 'dealing with input image ' +str(width) +'px width and ' + str(height) + 'px high'

  #We can get a pretty good idea of typical text regions by doing a white top hat operation
  #shaped_k = [[0,0,1,0,0],[0,0,1,0,0],[0,1,0,1,0],[0,0,1,0,0],[0,0,1,0,0]]
  #shaped_kernel = np.array(shaped_k , dtype=np.uint8)
  tophat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
  tophat_white = cv2.morphologyEx(binary, cv2.MORPH_TOPHAT, tophat_kernel,iterations=4)
  whitehat_inv = cv2.bitwise_not(tophat_white)

  #black tophat operation for reference (to be used for ajusting background gradient?)
  tophat_black = cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT, tophat_kernel, iterations=5)
  blackhat_inv = cv2.bitwise_not(tophat_black)
  (bh_thresh,blackhat_inv_thresh) = cv2.threshold(blackhat_inv, 1, 255, cv2.THRESH_BINARY_INV )

 
  #cv2.imshow('whitehat',whitehat_inv)
  #cv2.imshow('blackhat',blackhat_inv)

  #Form gutter masks to disconnect text/image/panel areas in the final mask
  vertical_guttermask_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,200))
  vertical_guttermask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_guttermask_kernel,iterations=1)
  #cv2.imshow('vertical_guttermask',vertical_guttermask)
  horizontal_guttermask_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(500,1))
  horizontal_guttermask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_guttermask_kernel,iterations=1)
  #cv2.imshow('horizontal_guttermask',horizontal_guttermask)


  #Form a "whitemask" which should differentiate all image areas (text included in image)
  #from just pure white. We use the horizontal and vertical gutter masks to maintain 
  #area separation
  whitemask = cv2.add(binary, binary, mask=whitehat_inv)
  whitemask = vertical_guttermask + whitemask + horizontal_guttermask
  #cv2.imshow('whitemask',whitemask)

  #In reality I'd like to generate a mask precise enough to pick the text out of
  #the original image by a single multiplicaiton, but as yet I can't do that
  #so here I'm taking apart elements in the mask by contour geometry.
  #it's a property of the text portions of the mask that they are usually quite
  #long (aspect ratio > 2) and connected.
  #Also, associated disconnected characters can be inferred (I believe) by
  #groping contour centroids that form straight lines.
  contours,hierarchy = cv2.findContours(whitemask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

  for c in contours:
    moments = cv2.moments(c)
    if moments['m00']!=0:
      cx = int(moments['m10']/moments['m00'])
      cy = int(moments['m01']/moments['m00'])
      #moment_area = moments['m00']
      #contour_area = cv2.contourArea(c)

      #put points at all contour centroids
      cv2.circle(img,(cx,cy),2,(0,0,255),-1)

      #and if our contour aspect ratio is "long" we draw the bounding box
      #note that this just checks the vert/horiz direction at present.
      x,y,w,h = cv2.boundingRect(c)
      vert_aspect_ratio = float(h)/w
      horiz_aspect_ratio = float(w)/h
      if horiz_aspect_ratio > 2 or vert_aspect_ratio > 2:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    #draw in all contours to see how they fall
    cv2.drawContours(img,[c],0,(0,255,0),1)
    
  
  cv2.imshow('img',img)
  cv2.imwrite('detect_text_output.jpg', img)

  if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
  cv2.destroyAllWindows()
