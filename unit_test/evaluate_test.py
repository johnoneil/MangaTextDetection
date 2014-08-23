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

  def test_one_character(self):
    actual = io.StringIO(u"し",);
    expected = io.StringIO(u"し");
    result = evaluate.evaluate(actual, expected);
    assert result.success;
    assert result.count == 1;
    assert result.failures == {};

  def test_one_character_does_not_match(self):
    actual = io.StringIO(u"あ");
    expected = io.StringIO(u"し");
    result = evaluate.evaluate(actual, expected);
    assert result.success == False;
    assert result.count == 1;
    assert result.failures == { u"し" : [{ "actual" : u"あ", "actual_location": "1:1", "expected_location": "1:1"}] };

  def test_endofline_unix_does_not_increase_count(self):
    actual = io.StringIO(u"\n");
    expected = io.StringIO(u"\n");
    result = evaluate.evaluate(actual, expected);
    assert result.success;
    assert result.count == 0;
    assert result.failures == {};

  def test_endofline_windows_does_not_increase_count(self):
    actual = io.StringIO(u"\r\n");
    expected = io.StringIO(u"\r\n");
    result = evaluate.evaluate(actual, expected);
    assert result.success;
    assert result.count == 0;
    assert result.failures == {};

  def test_endofline_mixed_unix_and_windows_does_not_increase_count(self):
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
    assert result.failures == { u"し" : [{ "actual" : u"あ", "actual_location": "2:1", "expected_location": "2:1"}] };

  def test_endoffile_mismatch_more_in_actual(self):
    actual = io.StringIO(u"あ\r\nし");
    expected = io.StringIO(u"あ\r\n");
    result = evaluate.evaluate(actual, expected);
    assert result.success == False;
    assert result.count == 1;
    assert result.failures == { u"EOF" : [{ "actual" : u"し", "actual_location": "2:1", "expected_location": "2:0"}] };

  def test_endoffile_mismatch_more_in_expected(self):
    actual = io.StringIO(u"あ\r\n");
    expected = io.StringIO(u"あ\r\nし");
    result = evaluate.evaluate(actual, expected);
    assert result.success == False;
    assert result.count == 2;
    assert result.failures == { u"し" : [{ "actual" : u"EOF", "actual_location": "2:0", "expected_location": "2:1"}] };

  def test_out_of_sync_stream(self):
    actual = io.StringIO(u"ぃ　あし\r\n");
    expected = io.StringIO(u"いあし\r\n");
    result = evaluate.evaluate(actual, expected);
    assert result.success == False;
    assert result.count == 3;
    assert result.failures == { u"い" : [{ "actual" : u"ぃ　", "actual_location": "1:1", "expected_location": "1:1"}] };

  def test_out_of_sync_stream_actual_new_lined_early(self):
    actual = io.StringIO(u"新しい\nしごと");
    expected = io.StringIO(u"新しいむすこ\nしごと\n");
    result = evaluate.evaluate(actual, expected);
    assert result.success == False;
    assert result.count == 9;
    assert result.failures == { u"む" : [{ "actual" : u"NL", "actual_location": "2:0", "expected_location": "1:4"}],
                                u"す" : [{ "actual" : u"NL", "actual_location": "2:0", "expected_location": "1:5"}],
                                u"こ" : [{ "actual" : u"NL", "actual_location": "2:0", "expected_location": "1:6"}],
                                };

  def test_out_of_sync_stream_expected_new_lined_early(self):
    actual = io.StringIO(u"新しいむすこ\nしごと\n");
    expected = io.StringIO(u"新しい\nしごと");
    result = evaluate.evaluate(actual, expected);
    assert result.success == False;
    assert result.count == 6;
    assert result.failures == { u"NL" : [{ "actual" : u"む", "actual_location": "1:4", "expected_location": "2:0"},
                                         { "actual" : u"す", "actual_location": "1:5", "expected_location": "2:0"},
                                         { "actual" : u"こ", "actual_location": "1:6", "expected_location": "2:0"}]
                              };

  def test_peek_when_empty(self):
    stream = io.StringIO();
    OUT = evaluate.EvaluationStream(stream);
    assert OUT.iseof(OUT.peek(1));
    assert OUT.iseof(OUT.peek(2));

  def test_peek(self):
    stream = io.StringIO(u"いあし\r\n");
    OUT = evaluate.EvaluationStream(stream);
    assert u"い" == OUT.peek(1);
    assert "1:0" == OUT.location();
    assert u"あ" == OUT.peek(2);
    assert "1:0" == OUT.location();
    assert u"し" == OUT.peek(3);
    assert "1:0" == OUT.location();
    assert OUT.isnewline(OUT.peek(4));
    assert "1:0" == OUT.location();
    assert OUT.iseof(OUT.peek(5));
    assert "1:0" == OUT.location();

  def test_success_statistics(self):
    actual = io.StringIO(u"ぃ　あしろろる\r\n");
    expected = io.StringIO(u"いあしるろる\r\n");
    result = evaluate.evaluate(actual, expected);
    assert result.success == False;
    assert result.count == 6;
    assert result.failures == { u"い" : [{ "actual" : u"ぃ　", "actual_location": "1:1", "expected_location": "1:1"}],
                                u"る" : [{ "actual" : u"ろ", "actual_location" : "1:5", "expected_location": "1:4"}]
                               };
    assert result.successes == {
                                u"あ" : ["1:2"],
                                u"し" : ["1:3"],
                                u"ろ" : ["1:5"],
                                u"る" : ["1:6"]
                                };
    assert result.percentages() == {
                                    u"い" : 0.0,
                                    u"あ" : 1.0,
                                    u"し" : 1.0,
                                    u"る" : 0.5,
                                    u"ろ" : 1.0
                                   };

