#!/usr/bin/python
 # vim: set ts=2 expandtab:
"""
Module: kmeans_test1.py
Date: Sunday, 14th July 2013
Desc:
Author: John O'Neil
Email: oneil.john@gmail.com

  As most of the morphological approaches to isolate text from
  drawings in Manga are failing in some sense or another (i.e.
  perhaps detect kana well, but not kanji, or not applicable
  to all font sizes, or not to text that is diagonal etc)
  I'm moving to the conclusion that a simple clustering of
  contours might be the best. Such a clustering might
  be possible if we group according to several criteria:
  * grouped text will all generally be of the same height
  * grouped text will all generally be spaced uniformly
  * groupled text might generally have the same aspect ratio
  * grouped text would fall along (generally) linear paths
  across the page (vertical/horizontal or otherwise)

  I'm going to start doing some k-means analysis of some pages
  to see if this approach bears fruit.
  
"""

import numpy as np
import scipy.cluster.vq as clustering
import cv2
import sys
import random

if __name__ == '__main__':

  #this experiment relies upon a single input argument
  if len(sys.argv)<2:
    print 'USAGE kmeans_test1.py <input image name>'
    sys.exit(-1)

  #convert image to grayscale, and do simple thresholding
  # I've found so far that adaptive thresholding is less useful here. That could be wrong.
  img = cv2.imread(sys.argv[1])
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  #binary = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,0)
  

  #get image geometry
  (h, w, d) = img.shape
  print 'dealing with input image ' +str(w) +'px width and ' + str(h) + 'px high'

  #scle our image down and upwards (gaussian filter). This is helpful in removing unwanted zip-a-tone
  #patterns.
  scaled = cv2.pyrUp(cv2.pyrDown(gray,dstsize=(w/2, h/2)), dstsize=(w, h));

  #Perform edge detection and then threshold the results to a binary image
  edges = cv2.Canny(scaled, 128, 200, apertureSize=3)
  (edges_threshold,bin_edges) = cv2.threshold(edges, 32, 255, cv2.THRESH_BINARY )

  contours,hierarchy = cv2.findContours(bin_edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

  #Next build a vector of contour sizes, which we'll 'whiten' and try kmeans on
  contour_sizes=[]
  contour_lookup =[]
  print 'len of contours is ' + str(len(contours))
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
      #print 'x ' + str(x) + ' y ' + str(y) + ' w ' + str(w) + ' h ' + str(h)
      #if horiz_aspect_ratio > 2 or vert_aspect_ratio > 2:
      #cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
      #draw in all contours to see how they fall
      #contour_sizes.append([float(x)*4,float(y)*4,max(float(w),float(h))])#,horiz_aspect_ratio,vert_aspect_ratio])
      contour_sizes.append([cx*8.0,cy*8.0,max(float(w),float(h))/8.0])#,horiz_aspect_ratio,vert_aspect_ratio])
      contour_lookup.append(c)
      #contour_sizes.append([float(x),float(w),float(h)])#,horiz_aspect_ratio])
    #cv2.drawContours(img,[c],0,(0,255,0),1)
 
  whitened_contour_sizes = clustering.whiten(contour_sizes)
  #print str(contour_sizes)

  # let scipy do its magic (k==3 groups)
  centers,dist = clustering.kmeans(whitened_contour_sizes,75,iter=100)
  code, distance = clustering.vq(whitened_contour_sizes,centers)
  #print str(centroid)
  #print str(code)

  #print 'contours is ' + str(len(contour_sizes)) + ' and code is ' + str(len(code))

  colors = [( int(random.uniform(0, 255)),int(random.uniform(0, 255)),int(random.uniform(0, 255))) for i in code ]
  #print str(colors)
  for i, label in enumerate(code):
    color = colors[label]
    x,y,w,h = cv2.boundingRect(contour_lookup[i])
    #box = contour_sizes[i]
    #x=int(box[0])
    #y=int(box[1])
    #w=int(box[2])
    #h=int(box[3])
    #print 'x ' + str(x) + ' y ' + str(y) + ' w ' + str(w) + ' h ' + str(h)
    #color=(0,0,255)
    cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
    #cv2.rectangle(img,(100,100),(200,200),color,2)
    #cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    

  #for i in code:
    #label = contour_sizes[code==i]
    #print str(label)
    #for l in label:
    #box = contour_sizes[i]
    #x=int(box[0])
    #y=int(box[1])
    #w=int(box[2])
    #h=int(box[3])
    #print 'x ' + str(x) + ' y ' + str(y) + ' w ' + str(w) + ' h ' + str(h)
    #color = ( random.uniform(0, 255),random.uniform(0, 255),random.uniform(0, 255))
    #color=(0,0,255)
    #cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
    #cv2.rectangle(img,(100,100),(200,200),color,2)
    #cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

  #for i in label:
  #  #print 'idx is: ' + str(label)
  #  c = contour_sizes[i]
  #  x=int(c[0])
  #  y=int(c[1])
  #  w=int(c[2])
  #  h=int(c[3])
  #  #color = ( random.uniform(0, 255),random.uniform(0, 255),random.uniform(0, 255))
  #  color=(0,0,255)
  #  cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
  
  cv2.imshow('img',img)
  #cv2.imwrite('detect_text_output.jpg', img)

  if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
  cv2.destroyAllWindows()
