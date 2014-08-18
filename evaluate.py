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
    self.successes = collections.defaultdict(list);

  def percentages(self):
    keys = set(self.successes.iterkeys()).union(self.failures.iterkeys());
    result = {};
    for key in keys:
      failure_count = len(self.failures[key]) if key in self.failures else 0
      success_count = len(self.successes[key]) if key in self.successes else 0;
      result[key] = success_count / float( failure_count + success_count );

    return result;

class EvaluationStream():
  """
  Wrap an io.TextIOBase to provide Evaluation support.

  :param stream: io.TextIOBase of the actual ocr results
  """

  _newline = u"NL";
  _eof = u"EOF";

  @staticmethod
  def isnewline(char):
    return EvaluationStream._newline == char;

  @staticmethod
  def iseof(char):
    return EvaluationStream._eof == char;

  def __init__(self, stream):
    self._stream = stream;
    self._line = 1;
    self._position = 0;
    self.count = 0;
    self._peek_buffer = collections.deque();
    self._max_peek_lookahead = 2;

  def _read_with_translations(self):
    """
    As per io.TextIOBase.read(1), but also ignore windows \r characters by reading the next character.
    \n is rewritten as NL so that mismatches are printable characters.
    end of file is rewritten as EOF for printability.
    """
    char = self._stream.read(1);
    while u"\r" == char:
      char = self._stream.read(1);

    if u"" == char:
      char = EvaluationStream._eof;
    elif u"\n" == char:
      char = EvaluationStream._newline;

    return char

  def _read_stream_or_peek_buffer(self):
    if self._peek_buffer:
      char = self._peek_buffer.popleft();
    else:
      char = self._read_with_translations();

    return char;

  def read(self):
    """
    As per io.TextIOBase.read(1), but also ignore windows \r characters by reading the next character.
    \n is rewritten as NL so that mismatches are printable characters.
    end of file is rewritten as EOF for printability.

    To support peek, an internal buffer is used and read from before re-reading from stream.
    """

    char = self._read_stream_or_peek_buffer();

    if self.iseof(char):
      pass; # EOF doesn't increment counts
    elif self.isnewline(char):
      self._line += 1;
      self._position = 0;
    else:
      self._position += 1;
      self.count += 1;

    return char;

  def location(self):
    return "{0:d}:{1:d}".format(self._line, self._position);

  def peek(self, n):
    """
    Peek ahead n characters in the input stream and return that character
    """

    current_peek_chars_available = len(self._peek_buffer);
    chars_needed = n - current_peek_chars_available;
    for _ in range(chars_needed):
      self._peek_buffer.append(self._read_with_translations());
    result = self._peek_buffer[n-1];
    return result;

  def resync(self, current_char, tostream):
    """
    Lookahead on the stream to see if re-syncing is required.
    If re-syncing is required the the extra characters will be consumed and returned appended to current_char

    :param current_char: the current failing character
    :param tostream: the evaluation stream to sync to
    """
    sync_to_char = tostream.peek(1);

    if self.iseof(sync_to_char):
      # Dont resync on EOF
      return current_char;

    resync_found_ahead_at = None;
    for i in range(1, self._max_peek_lookahead+1):
      candidate_sync_spot = self.peek(i);
      if sync_to_char == candidate_sync_spot:
        resync_found_ahead_at = i;

    if resync_found_ahead_at:
      while (resync_found_ahead_at > 1): # capture up to (but not including) the resync character
        resync_found_ahead_at -= 1;
        current_char += self.read();

    return current_char;


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
      failure_details = { "actual_location" : actual.location(), "expected_location" : expected.location()};
      actual_char = actual.resync(actual_char, expected);
      failure_details["actual"] = actual_char; # resync'ing changes location to the end of the sync, and we want the beginning
      result.failures[expected_char].append(failure_details);
    else:
      if not expected.isnewline(expected_char):
        result.successes[expected_char].append(expected.location());

    if expected.iseof(expected_char):
      result.success = False;
      break;


  result.count = expected.count;
  return result;


