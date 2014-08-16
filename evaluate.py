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

class EvaluationStream():
  """
  Wrap an io.TextIOBase to provide Evaluation support.

  :param stream: io.TextIOBase of the actual ocr results
  """
  def __init__(self, stream):
    self._stream = stream;
    self._newline = u"NL";
    self._eof = u"EOF";
    self._line = 1;
    self._position = 0;
    self.count = 0;

  def read(self):
    """
    As per io.TextIOBase.read(1), but also ignore windows \r characters by reading the next character.
    \n is rewritten as NL so that mismatches are printable characters.
    end of file is rewritten as EOF for printability.
    """
    char = self._stream.read(1);
    while u"\r" == char:
      char = self._stream.read(1);

    if u"" == char:
      char = self._eof;
    elif u"\n" == char:
      char = self._newline;
      self._line += 1;
      self._position = 0;
    else:
      self._position += 1;
      self.count += 1;

    return char;

  def isnewline(self, char):
    return self._newline == char;

  def iseof(self, char):
    return self._eof == char;

  def location(self):
    return "{0:d}:{1:d}".format(self._line, self._position);

def evaluate(actual, expected):
  """
  Evaluate the actual ocr results against the expected results and provide metrics on failures.

  :param actual: io.TextIOBase of the actual ocr results
  :param expected: io.TextIOBase of the expected ocr results
  """
  result = Evaluation();
  actual = EvaluationStream(actual);
  expected = EvaluationStream(expected);
  while True:
    expected_char = expected.read();
    actual_char = actual.read();
    if expected.iseof(expected_char) and actual.iseof(actual_char):
      if result.success == None:
        result.success = True;
      break;

    if expected_char != actual_char:
      result.success = False;
      result.failures[expected_char].append({ "actual" : actual_char, "actual_position" : actual.location(), "expected_position" : expected.location()});

    if expected.iseof(expected_char):
      result.success = False;
      break;

  result.count = expected.count;
  return result;


