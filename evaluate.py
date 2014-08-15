# -*- coding: utf-8 -*-

"""
Module: evaluate.py
Desc: Evaluate the ocr esults against the expected output and provide metrics on failures
Author: Barrie Treloar
Email: baerrach@gmail.com
DATE: 13th Aug 2014

  TODO
"""

import collections;

class Evaluation:
  def __init__(self):
    self.success = None;
    self.count = 0;
    self.failures = collections.defaultdict(list);

_newline = u"NL";

def _read(stream):
  """
  Read a single unicode character from a stream and ignore windows \r characters by reading the next character.
  \n is rewritten as NL so that mismatches are printable characters.
  """
  char = stream.read(1);
  while u"\r" == char:
    char = stream.read(1);

  if u"\n" == char:
    char = _newline;

  return char;

def _isnewline(char):
  return _newline == char;

def evaluate(actual, expected):
  """
  Evaluate the actual ocr results against the expected results and provide metrics on failures.

  :param actual: io.TextIOBase of the actual ocr results
  :param expected: io.TextIOBase of the expected ocr results
  """
  result = Evaluation();
  line = 1;
  column = 0;
  while True:
    expected_char = _read(expected);
    actual_char = _read(actual);
    if expected_char == "" or actual_char == "":
      if result.success == None:
        result.success = True;
      break;

    if _isnewline(expected_char) and _isnewline(actual_char):
      line += 1;
      column = 0;
    else:
      result.count += 1;
      column += 1;

    if expected_char != actual_char:
      result.success = False;
      result.failures[expected_char].append({ "actual" : actual_char, "line" : line, "position" : column});


  return result;


