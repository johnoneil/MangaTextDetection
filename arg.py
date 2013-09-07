#!/usr/bin/python
 # vim: set ts=2 expandtab:
"""
Module: arg.py
Desc: Consolidated argument parser across manga detection modules
Author: John O'Neil
Email: oneil.john@gmail.com
DATE: Friday, Sept 6th 2013

  Import this argparser into modules, so they all can be driven
  off a 'global' argparser.
  
"""

import argparse


parser = None
value = None

def is_defined(arg_name):
  if not value:
    return False
  if not getattr(value, arg_name):
    return False
  return True

def string_value(arg_name, default_value=""):
  if not value:
    return default_value
  try:
    argument = getattr(value, arg_name)
  except AttributeError:
    return default_value
  if not argument:
    return default_value
  elif type(argument)!=str:
    return default_value
  else:
    return argument

def boolean_value(arg_name, default_value=False):
  if not value:
    return default_value
  try:
    argument = getattr(value, arg_name)
  except AttributeError:
    return default_value
  if not argument:
    return default_value
  elif type(argument)!=bool:
    return default_value
  else:
    return argument

def integer_value(arg_name, default_value=0):
  if not value:
    return default_value
  try:
    argument = getattr(value, arg_name)
  except AttributeError:
    return default_value
  if not argument:
    return default_value
  elif type(argument)!=int:
    return default_value
  else:
    return argument

def float_value(arg_name, default_value=0.0):
  if not value:
    return default_value
  try:
    argument = getattr(value, arg_name)
  except AttributeError:
    return default_value
  if not argument:
    return default_value
  elif type(argument)!=float:
    return default_value
  else:
    return argument

'''
parser = argparse.ArgumentParser(description='Segment raw Manga scan image.')
parser.add_argument('infile', help='Input (color) raw Manga scan image to clean.')
parser.add_argument('-o','--output', dest='outfile', help='Output (color) cleaned raw manga scan image.')
#parser.add_argument('-m','--mask', dest='mask', default=None, help='Output (binary) mask for non-graphical regions.')
#parser.add_argument('-b','--binary', dest='binary', default=None, help='Binarized version of input file.')
parser.add_argument('-v','--verbose', help='Verbose operation. Print status messages during processing', action="store_true")
parser.add_argument('--display', help='Display output using OPENCV api and block program exit.', action="store_true")
parser.add_argument('-d','--debug', help='Overlay input image into output.', action="store_true")
parser.add_argument('--sigma', help='Std Dev of gaussian preprocesing filter.',type=float,default=None)
parser.add_argument('--binary_threshold', help='Binarization threshold value from 0 to 255.',type=float,default=190)
parser.add_argument('--furigana', help='Attempt to suppress furigana characters to improve OCR.', action="store_true")
parser.add_argument('--segment_threshold', help='Threshold for nonzero pixels to separete vert/horiz text lines.',type=int,default=1)
'''
