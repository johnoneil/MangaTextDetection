#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Module: evaluate.py
Desc: Evaluate the ocr esults against the expected output and provide metrics on failures
Author: Barrie Treloar
Email: baerrach@gmail.com
DATE: 13th Aug 2014

  TODO
"""

import codecs;
import collections;
import argparse;
import arg;
import sys;
import os;
import json;
import logging

logger = logging.getLogger(__name__);

class IgnoreUnderscoreEncoder(json.JSONEncoder):
    def default(self, obj):
        attributes = {}
        obj_dict = obj.__dict__
        for key, value in obj_dict.iteritems():
          if key.startswith(u'_'):
              continue
          attributes[key] = value
        return attributes

class Evaluation:
  def __init__(self,expected_stream,actual_stream):
    self.success = None;
    self.count = 0;
    self.failures = collections.defaultdict(list);
    self.successes = collections.defaultdict(list);
    self._percentages = None;
    self._actual = EvaluationStream(actual_stream);
    self._expected = EvaluationStream(expected_stream);
    self._actual_char = None;
    self._expected_char = None;

  def readFromExpected(self):
    self._expected_char = self._expected.read();

  def readFromActual(self):
    self._actual_char = self._actual.read()

  def evaluate(self):
    """
    Evaluate the actual ocr results against the expected results and provide metrics on failures.
    """

    if logger.isEnabledFor(logging.DEBUG):
      sys.stdout.write("Debug Legend:\n");
      sys.stdout.write("  . = matched\n");
      sys.stdout.write("  X = failed\n");
      sys.stdout.write("  s = skipped\n");

    while True:
      self.readFromExpected();
      self.readFromActual();
      if EvaluationStream.iseof(self._expected_char) and EvaluationStream.iseof(self._actual_char):
        if self.success == None:
          self.success = True;
        break;

      up_to_count = self._expected.count;

      if self._expected_char != self._actual_char:
        self.success = False;
        failure_details = { u"actual_location" : self._actual.location(), u"expected_location" : self._expected.location()};
        if EvaluationStream.isnewline(self._expected_char):
          # Resync other stream to the next newline
          while not (EvaluationStream.isnewline(self._actual_char) or EvaluationStream.iseof(self._actual_char)):
            failure_details = { u"actual" : self._actual_char, u"actual_location" : self._actual.location(), u"expected_location" : self._expected.location()};
            self.failures[self._expected_char].append(failure_details);
            self._actual_char = self._actual.read();
        elif EvaluationStream.isnewline(self._actual_char):
          # Resync other stream to the next newline
          while not (EvaluationStream.isnewline(self._expected_char) or EvaluationStream.iseof(self._expected_char)):
            failure_details = { u"actual" : self._actual_char, u"actual_location" : self._actual.location(), u"expected_location" : self._expected.location()};
            self.failures[self._expected_char].append(failure_details);
            self._expected_char = self._expected.read();
        else:
          actual_char = self._actual.resync(self._actual_char, self._expected);
          failure_details[u"actual"] = actual_char; # resync'ing changes location to the end of the sync, and we want the beginning
          self.failures[self._expected_char].append(failure_details);
          if logger.isEnabledFor(logging.DEBUG):
            sys.stdout.write("X");
            if len(actual_char) > 1:
              sys.stdout.write("s" * (len(actual_char)-1));
      else:
        if not EvaluationStream.isnewline(self._expected_char):
          self.successes[self._expected_char].append(self._expected.location());
          if logger.isEnabledFor(logging.DEBUG):
            sys.stdout.write(".");
        else:
          if logger.isEnabledFor(logging.DEBUG):
            sys.stdout.write("\n");

      if EvaluationStream.iseof(self._expected_char):
        self.success = False;
        break;

    if logger.isEnabledFor(logging.DEBUG):
      sys.stdout.write("\n");
      sys.stdout.flush();
    self.count = self._expected.count;
    return self;

  def percentages(self):
    if not self._percentages:
      keys = set(self.successes.iterkeys()).union(self.failures.iterkeys());
      self._percentages = {};
      for key in keys:
        failure_count = len(self.failures[key]) if key in self.failures else 0
        success_count = len(self.successes[key]) if key in self.successes else 0;
        self._percentages[key] = success_count / float( failure_count + success_count );

    return self._percentages;

  def overall(self):
    values = self.percentages().values()
    return sum(values)/len(values);

  def __str__(self):
    return unicode(self).encode('utf-8');

  def __unicode__(self):
    result = [];
    result.append(u"success={0!s}".format(self.success));
    result.append(u"count={0:d}".format(self.count));
    result.append(u"failures={");
    for key, value in self.failures.iteritems():
      result.append(u"  '{0}' = {1},".format(key, unicode(value)));
    result.append(u"}");
    result.append(u"successes={");
    for key, value in self.successes.iteritems():
      result.append(u"  '{0}' = {1},".format(key, value));
    result.append(u"}");
    result.append(u"percentages={");
    for key, value in self.percentages().iteritems():
      result.append(u"  '{0}' = {1},".format(key, value));
    result.append(u"}");
    result.append(u"overall={0}".format(self.overall()));
    return u"\n".join(result);

  def summary(self):
    result = [];
    result.append(u"success={0!s}".format(self.success));
    result.append(u"count={0:d}".format(self.count));
    result.append(u"overall={0}".format(self.overall()));
    return u"\n".join(result);

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

    if EvaluationStream.iseof(char):
      pass; # EOF doesn't increment counts
    elif EvaluationStream.isnewline(char):
      self._line += 1;
      self._position = 0;
    else:
      self._position += 1;
      self.count += 1;

    return char;

  def location(self):
    return u"{0:d}:{1:d}".format(self._line, self._position);

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

    if EvaluationStream.iseof(sync_to_char):
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

def main():
  parser = argparse.ArgumentParser(description="Evaluate text against correct version.");
  parser.add_argument("-c", "--correct", dest="correct_file", help="File containing the correct text");
  parser.add_argument("-i", "--input", dest="input_file", required=True, help="File containing the text to compare against the correct version");
  parser.add_argument("-r", "--results", dest="results_file", help="File to write evaluation results to");
  parser.add_argument("-d", "--debug", action="store_true", help="Enable debug tracing");

  arg.value = parser.parse_args();
  correct_file = arg.string_value("correct_file", default_value="correct.txt");
  input_file = arg.string_value("input_file");
  results_file = arg.string_value("results_file", default_value=input_file+"-results.txt");
  if arg.boolean_value("debug"):
    logging.getLogger().setLevel(logging.DEBUG);

  if not os.path.isfile(input_file):
    print("Input file '{0}' does not exist. Use -h option for help".format(input_file));
    sys.exit(-1);

  if not os.path.isfile(correct_file):
    print("Correct file '{0}' does not exist. Use -h option for help".format(correct_file));
    sys.exit(-1);

  with codecs.open(correct_file, "rU", "utf-8") as c, codecs.open(input_file, "rU", "utf-8") as i:
    result = Evaluation(c, i);
    result.evaluate();

  with codecs.open(results_file, "wU", "utf-8") as w:
    json.dump(result, w, cls=IgnoreUnderscoreEncoder, ensure_ascii=False, indent=2, separators=(',', ': '), sort_keys=True);

  print(u"Summary of evaluation results:");
  print(u"results={0}".format(results_file));
  print(result.summary());


if __name__ == "__main__":
  logging.basicConfig(stream=sys.stderr, level=logging.INFO);

  UTF8Writer = codecs.getwriter('utf8');
  sys.stdout = UTF8Writer(sys.stdout);
  main();
