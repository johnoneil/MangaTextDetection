# -*- coding: utf-8 -*-

import io;
import evaluate;

class TestEvaluate:

  def test_empty(self):
    actual = io.StringIO();
    expected = io.StringIO();
    result = evaluate.evaluate(actual, expected);
    assert result.success;
    assert result.count == 0;
    assert result.failures == {};

  def test_onecharacter(self):
    actual = io.StringIO(u"し",);
    expected = io.StringIO(u"し");
    result = evaluate.evaluate(actual, expected);
    assert result.success;
    assert result.count == 1;
    assert result.failures == {};

  def test_onecharacter_does_not_match(self):
    actual = io.StringIO(u"あ");
    expected = io.StringIO(u"し");
    result = evaluate.evaluate(actual, expected);
    assert result.success == False;
    assert result.count == 1;
    assert result.failures == { u"し" : [{ "actual" : u"あ", "line" : 1, "position" : 1}] };

  def test_endofline_unix_doesnot_increase_count(self):
    actual = io.StringIO(u"\n");
    expected = io.StringIO(u"\n");
    result = evaluate.evaluate(actual, expected);
    assert result.success;
    assert result.count == 0;
    assert result.failures == {};

  def test_endofline_windows_doesnot_increase_count(self):
    actual = io.StringIO(u"\r\n");
    expected = io.StringIO(u"\r\n");
    result = evaluate.evaluate(actual, expected);
    assert result.success;
    assert result.count == 0;
    assert result.failures == {};

  def test_endofline_mixed_unix_and_windows_doesnot_increase_count(self):
    actual = io.StringIO(u"\n");
    expected = io.StringIO(u"\r\n");
    result = evaluate.evaluate(actual, expected);
    assert result.success;
    assert result.count == 0;
    assert result.failures == {};

  def test_line_reported_in_failures(self):
    actual = io.StringIO(u"\r\nあ");
    expected = io.StringIO(u"\r\nし");
    result = evaluate.evaluate(actual, expected);
    assert result.success == False;
    assert result.count == 1;
    assert result.failures == { u"し" : [{ "actual" : u"あ", "line" : 2, "position" : 1}] };

