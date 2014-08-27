#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Module: evaluate.py
Desc: Evaluate the ocr results against the expected output and provide metrics on failures
Author: Barrie Treloar
Email: baerrach@gmail.com
DATE: 13th Aug 2014
"""

import codecs
import collections
import argparse
import arg
import sys
import os
import json
import logging

logger = logging.getLogger(__name__)
trace = logging.getLogger("trace")
trace.setLevel(logging.INFO)

class IgnoreUnderscoreEncoder(json.JSONEncoder):

    """
    This JSON Encoder ignores any keys that start with an underscore.

    Python uses underscore to indicate a private field. This JSON Encoder will ignore these
    private fields and return the public version of the data.
    """

    def default(self, obj):
        attributes = {}
        obj_dict = obj.__dict__
        for key, value in obj_dict.iteritems():
            if key.startswith(u'_'):
                continue
            attributes[key] = value
        return attributes

class Evaluation:

    u"""
    Evaluation takes an expected stream and an actual stream and evaluates them to determine how closely they match.

    self.success = True if they match completely, false otherwise.
    self.count = The count of the characters read from the actual stream
    self.failures = A dictionary of the failures, keyed by the failed character.
                    The value is a list of dictionaries that describe the failure locations.
                    e.g.
                    {
                     u"い": [{"actual": u"ぃ　", "actual_location": "1:1", "expected_location": "1:1"}],
                     u"る": [{"actual": u"ろ", "actual_location" : "1:5", "expected_location": "1:4"}]
                     }
    self.successes = A dictionary of the successes, keyed by the success character.
                     The value is a list of dictionaries that describe the success locations.
                     e.g.
                     {
                      u"あ": [{'actual_location': '1:3', 'expected_location': '1:2'}],
                      u"し": [{'actual_location': '1:4', 'expected_location': '1:3'}]
                      }
    self.percentages = A dictionary of characters and their percentages of successful matches.
    self.percentages_overall = A percentage of the successes over the entire stream.
    """

    def __init__(self, expected_stream, actual_stream):
        """
        Setup the Evaluation state to evaluate the two provided streams.

        Internally the streams are wrapped in EvaluationStream objects to provide additional features needed
        to track the stream; line and column positions, universal newline handling, peeking.

        :param expected_stream: Contains the expected text
        :type expected_stream: io.TextIOBase
        :param actual_stream: Contains the actual text
        :type actual_stream: io.TextIOBase
        """

        self.success = None
        self.count = 0
        self.failures = collections.defaultdict(list)
        self.successes = collections.defaultdict(list)
        self.percentages = None
        self.percentage_overall = None
        self._actual = EvaluationStream(actual_stream)
        self._expected = EvaluationStream(expected_stream)
        self._actual_char = None
        self._expected_char = None
        self._max_peek_lookahead = 3

    def readFromExpected(self):
        """Read from the expected stream and store the character read"""
        self._expected_char = self._expected.read()

    def readFromActual(self):
        """Read from the actual stream and store the character read"""
        self._actual_char = self._actual.read()

    def markFailure(self, actual_location=None):
        """
        Mark the current character as a failure.

        :param actual_location: The location where the failure occurred. If not provided the current location of the actual stream is used.
        """

        if not actual_location:
            actual_location = self._actual.location()
        failure_details = {u"actual":self._actual_char, u"actual_location":actual_location, u"expected_location":self._expected.location()}
        self.failures[self._expected_char].append(failure_details)
        if logger.isEnabledFor(logging.DEBUG):
            if EvaluationStream.iseof(self._expected_char):
                sys.stdout.write("E")
            elif EvaluationStream.iseof(self._actual_char):
                sys.stdout.write("e")
            elif EvaluationStream.isnewline(self._expected_char) or EvaluationStream.isnewline(self._actual_char):
                sys.stdout.write("$")
            elif EvaluationStream.isspace(self._actual_char):
                sys.stdout.write("_")
            else:
                sys.stdout.write("X")
                if len(self._actual_char) > 1:
                    sys.stdout.write("s" * (len(self._actual_char) - 1))
                trace.debug(u"expected='{0}' actual='{1}' expected_location={2} actual_location={3}".format(self._expected_char, self._actual_char, self._expected.location(), self._actual.location()))

    def resyncActual(self):
        """
        Lookahead on the stream to see if re-syncing is required.
        If re-syncing is required then the extra characters will be consumed and appended to self._actual_char.
        If characters are consumed then then position in the stream will also change. If you need to know the original
        position prior to resyncing then store the location prior to invoking this method.

        TODO: Add confidence by continuing past the resync point and determining how many characters still match.
              Note: For Japanese this isn't so useful since a kanji is an entire word and OCR may be failing
              on every other character and reducing confidence.

        TODO: Resync should stop at newlines. Tests indicate that this is not currently a problem.
        """

        sync_to_char = self._expected.peek(1)

        if EvaluationStream.iseof(sync_to_char):
            # Dont resync on EOF
            return

        resync_found_ahead_at = None
        for i in range(1, self._max_peek_lookahead + 1):
            candidate_sync_spot = self._actual.peek(i)
            if sync_to_char == candidate_sync_spot:
                resync_found_ahead_at = i
                break

        if resync_found_ahead_at:
            while (resync_found_ahead_at > 1):  # capture up to (but not including) the resync character
                resync_found_ahead_at -= 1
                self._actual_char += self._actual.read()

    def handleMismatch(self):
        """
        Handle a mismatch of the streams.

        This will mark the self.success as False indicating that the evaulation was not successful.

        If a newline is encountered on either stream then the other stream is consumed until a newline is found
        or the end of file is reached.

        If actual char is a space then peek ahead to see if that character matches what was expected.
        The space is marked as a failure and the expected char pushed back onto the stream to get back in sync.

        Otherwise the actual location is marked and a resync is attemped before marking the failure.
        """

        self.success = False
        if EvaluationStream.isnewline(self._expected_char):  # Resync actual stream to the next newline
            while not EvaluationStream.isnewline(self._actual_char) and not EvaluationStream.iseof(self._actual_char):
                self.markFailure()
                self.readFromActual()
        elif EvaluationStream.isnewline(self._actual_char):  # Resync expected stream to the next newline
            while not EvaluationStream.isnewline(self._expected_char) and not EvaluationStream.iseof(self._expected_char):
                self.markFailure()
                self.readFromExpected()
        elif EvaluationStream.isspace(self._actual_char) and self._expected_char == self._actual.peek(1):  # ignore whitespace if the next char matches
            self.markFailure()
            self._expected.push_back(self._expected_char)
        else:
            mark_failure_position = self._actual.location()
            self.resyncActual()
            self.markFailure(mark_failure_position)

    def handleMatch(self):
        """
        Handle a match of the streams.
        """

        self.successes[self._expected_char].append({"expected_location":self._expected.location(), "actual_location":self._actual.location()})
        if not EvaluationStream.isnewline(self._expected_char):
            if logger.isEnabledFor(logging.DEBUG):
                sys.stdout.write(".")
        elif logger.isEnabledFor(logging.DEBUG):
            sys.stdout.write("\n")

    def evaluate(self):
        """
        Evaluate the actual ocr results against the expected results and provide metrics on failures.
        """

        if logger.isEnabledFor(logging.DEBUG):
            sys.stdout.write("Debug Legend:\n")
            sys.stdout.write("  . = matched\n")
            sys.stdout.write("  X = failed\n")
            sys.stdout.write("  s = skipped\n")
            sys.stdout.write("  _ = skipped extra whitespace\n")
            sys.stdout.write("  $ = End of Line (expected or actual)")
            sys.stdout.write("  E = End of File (expected)\n")
            sys.stdout.write("  e = End of File (actual)\n")

        while True:
            self.readFromExpected()
            self.readFromActual()
            if EvaluationStream.iseof(self._expected_char) and EvaluationStream.iseof(self._actual_char):
                if self.success == None:
                    self.success = True
                break

            up_to_count = self._expected.count

            if self._expected_char != self._actual_char:
                self.handleMismatch()
            else:
                self.handleMatch()

            if EvaluationStream.iseof(self._expected_char):
                self.success = False
                break

        if logger.isEnabledFor(logging.DEBUG):
            sys.stdout.write("\n")
            sys.stdout.flush()

        self.count = self._expected.count

        self._calculate_percentages()
        return self

    def _calculate_percentages(self):
        """
        Calculate the percentages of successes to failures.
        """

        keys = set(self.successes.iterkeys()).union(self.failures.iterkeys())
        self.percentages = {}
        for key in keys:
            failure_count = len(self.failures[key]) if key in self.failures else 0
            success_count = len(self.successes[key]) if key in self.successes else 0
            self.percentages[key] = success_count / float(failure_count + success_count)

        values = self.percentages.values()
        if values:
            self.percentage_overall = sum(values) / len(values)

    def __str__(self):
        return unicode(self).encode('utf-8')

    def __unicode__(self):
        result = []
        result.append(u"success={0!s}".format(self.success))
        result.append(u"count={0:d}".format(self.count))
        result.append(u"failures={")
        for key, value in self.failures.iteritems():
            result.append(u"  '{0}' = {1},".format(key, unicode(value)))
        result.append(u"}")
        result.append(u"successes={")
        for key, value in self.successes.iteritems():
            result.append(u"  '{0}' = {1},".format(key, value))
        result.append(u"}")
        result.append(u"percentages={")
        for key, value in self.percentages().iteritems():
            result.append(u"  '{0}' = {1},".format(key, value))
        result.append(u"}")
        result.append(u"overall={0}".format(self.overall()))
        return u"\n".join(result)

    def summary(self):
        """
        Provide a summary version of __unicode__
        """

        result = []
        result.append(u"success={0!s}".format(self.success))
        result.append(u"count={0:d}".format(self.count))
        result.append(u"overall={0}".format(self.percentage_overall))
        return u"\n".join(result)


class EvaluationStream():

    """
    Wrap an io.TextIOBase to provide Evaluation support.

    self.count = How many characters have been read from the stream.
    """

    _newline = u"NL"
    _eof = u"EOF"

    @staticmethod
    def isnewline(char):
        """Check whether the char is a newline"""
        return EvaluationStream._newline == char

    @staticmethod
    def iseof(char):
        """Check whether the char is the end of file"""
        return EvaluationStream._eof == char

    @staticmethod
    def isspace(char):
        """Check whether the character is a space"""
        return u" " == char

    def __init__(self, stream):
        self._stream = stream
        self._line = 1
        self._position = 0
        self.count = 0
        self._peek_buffer = collections.deque()

    def _read_with_translations(self):
        """
        As per io.TextIOBase.read(1), but also ignore windows \r characters by reading the next character.
        \n is rewritten as NL so that mismatches are printable characters.
        end of file is rewritten as EOF for printability.

        Use self._read_stream_or_peek_buffer instead of this function directly.
        """

        char = self._stream.read(1)
        while u"\r" == char:
            char = self._stream.read(1)

        if u"" == char:
            char = EvaluationStream._eof
        elif u"\n" == char:
            char = EvaluationStream._newline

        return char

    def _read_stream_or_peek_buffer(self):
        """
        Reads a character from the peek buffer, if there is anything on it, or else directly fron the stream.
        """

        if self._peek_buffer:
            char = self._peek_buffer.popleft()
        else:
            char = self._read_with_translations()

        return char

    def read(self):
        """
        As per io.TextIOBase.read(1), but also ignore windows \r characters by reading the next character.
        \n is rewritten as NL so that mismatches are printable characters.
        end of file is rewritten as EOF for printability.

        To support peek, an internal buffer is used and read from before re-reading from stream.

        Internal counters are incrememented to track the current line and position, see self.location()
        """

        char = self._read_stream_or_peek_buffer()

        if EvaluationStream.iseof(char):
            pass    # EOF doesn't increment counts
        elif EvaluationStream.isnewline(char):
            self._line += 1
            self._position = 0
        else:
            self._position += 1
            self.count += 1

        return char

    def location(self):
        """Return a string description of the streams location, in the form of <line>:<position>"""
        return u"{0:d}:{1:d}".format(self._line, self._position)

    def peek(self, n):
        """
        Peek ahead n characters in the input stream and return that character
        """

        current_peek_chars_available = len(self._peek_buffer)
        chars_needed = n - current_peek_chars_available
        for _ in range(chars_needed):
            self._peek_buffer.append(self._read_with_translations())
        result = self._peek_buffer[n - 1]
        return result

    def push_back(self, char):
        """
        Push the provided character back onto the head of the stream.

        Newline and EOF are not supported.
        """
        assert not EvaluationStream.iseof(char)
        assert not EvaluationStream.isnewline(char)
        self._position -= 1
        self.count -= 1

        self._peek_buffer.appendleft(char)

def main():
    parser = argparse.ArgumentParser(description="Evaluate text against correct version.")
    parser.add_argument("-c", "--correct", dest="correct_file", help="File containing the correct text")
    parser.add_argument("-i", "--input", dest="input_file", required=True, help="File containing the text to compare against the correct version")
    parser.add_argument("-r", "--results", dest="results_file", help="File to write evaluation results to")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug tracing")
    parser.add_argument("-t", "--trace", action="store_true", help="Print out mismatches as they occur. Also enables debug")

    arg.value = parser.parse_args()
    correct_file = arg.string_value("correct_file", default_value="correct.txt")
    input_file = arg.string_value("input_file")
    results_file = arg.string_value("results_file", default_value=input_file + "-results.txt")
    if arg.boolean_value("debug") or arg.boolean_value("trace"):
        logging.getLogger().setLevel(logging.DEBUG)
    if arg.boolean_value("trace"):
        trace.setLevel(logging.DEBUG)

    if not os.path.isfile(input_file):
        print("Input file '{0}' does not exist. Use -h option for help".format(input_file))
        sys.exit(-1)

    if not os.path.isfile(correct_file):
        print("Correct file '{0}' does not exist. Use -h option for help".format(correct_file))
        sys.exit(-1)

    with codecs.open(correct_file, "rU", "utf-8") as c, codecs.open(input_file, "rU", "utf-8") as i:
        result = Evaluation(c, i)
        result.evaluate()

    with codecs.open(results_file, "wU", "utf-8") as w:
        json.dump(result, w, cls=IgnoreUnderscoreEncoder, ensure_ascii=False, indent=2, separators=(',', ': '), sort_keys=True)

    print(u"Summary of evaluation results:")
    print(u"results={0}".format(results_file))
    print(result.summary())


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)

    UTF8Writer = codecs.getwriter('utf8')
    sys.stdout = UTF8Writer(sys.stdout)
    main()
