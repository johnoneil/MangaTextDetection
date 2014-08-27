# -*- coding: utf-8 -*-

import io
import json
from evaluate import Evaluation, EvaluationStream, IgnoreUnderscoreEncoder
import sys

class TestEvaluate:

    def test_empty(self):
        actual = io.StringIO()
        expected = io.StringIO()
        result = Evaluation(expected,actual)
        result.evaluate()
        assert result.success
        assert result.count == 0
        assert result.failures == {}

    def test_one_character(self):
        actual = io.StringIO(u"し",)
        expected = io.StringIO(u"し")
        result = Evaluation(expected,actual)
        result.evaluate()
        assert result.success
        assert result.count == 1
        assert result.failures == {}

    def test_one_character_does_not_match(self):
        actual = io.StringIO(u"あ")
        expected = io.StringIO(u"し")
        result = Evaluation(expected,actual)
        result.evaluate()
        assert result.success == False
        assert result.count == 1
        assert result.failures == { u"し" : [{ "actual" : u"あ", "actual_location": "1:1", "expected_location": "1:1"}] }

    def test_newline_unix_does_not_increase_count(self):
        actual = io.StringIO(u"\n")
        expected = io.StringIO(u"\n")
        result = Evaluation(expected,actual)
        result.evaluate()
        assert result.success
        assert result.count == 0
        assert result.failures == {}

    def test_newline_windows_does_not_increase_count(self):
        actual = io.StringIO(u"\r\n")
        expected = io.StringIO(u"\r\n")
        result = Evaluation(expected,actual)
        result.evaluate()
        assert result.success
        assert result.count == 0
        assert result.failures == {}

    def test_newline_mixed_unix_and_windows_does_not_increase_count(self):
        actual = io.StringIO(u"\n")
        expected = io.StringIO(u"\r\n")
        result = Evaluation(expected,actual)
        result.evaluate()
        assert result.success
        assert result.count == 0
        assert result.failures == {}

    def test_line_reported_in_failures(self):
        actual = io.StringIO(u"\r\nあ")
        expected = io.StringIO(u"\r\nし")
        result = Evaluation(expected,actual)
        result.evaluate()
        assert result.success == False
        assert result.count == 1
        assert result.failures == { u"し" : [{ "actual" : u"あ", "actual_location": "2:1", "expected_location": "2:1"}] }

    def test_endoffile_mismatch_more_in_actual(self):
        actual = io.StringIO(u"あ\r\nし")
        expected = io.StringIO(u"あ\r\n")
        result = Evaluation(expected,actual)
        result.evaluate()
        assert result.success == False
        assert result.count == 1
        assert result.failures == { u"EOF" : [{ "actual" : u"し", "actual_location": "2:1", "expected_location": "2:0"}] }

    def test_endoffile_mismatch_more_in_expected(self):
        actual = io.StringIO(u"あ\r\n")
        expected = io.StringIO(u"あ\r\nし")
        result = Evaluation(expected,actual)
        result.evaluate()
        assert result.success == False
        assert result.count == 2
        assert result.failures == { u"し" : [{ "actual" : u"EOF", "actual_location": "2:0", "expected_location": "2:1"}] }

    def test_mismatch_prior_to_newline(self):
        actual = io.StringIO(u"\"\nいあ")
        expected = io.StringIO(u"。\nいあ")
        result = Evaluation(expected,actual)
        result.evaluate()
        assert result.success == False
        assert result.count == 3
        assert result.failures == { u"。" : [{ "actual" : u"\"", "actual_location": "1:1", "expected_location": "1:1"}] }

    def test_mismatch_prior_to_newline_windows(self):
        actual = io.StringIO(u"\"\r\nいあ")
        expected = io.StringIO(u"。\r\nいあ")
        result = Evaluation(expected,actual)
        result.evaluate()
        assert result.success == False
        assert result.count == 3
        assert result.failures == { u"。" : [{ "actual" : u"\"", "actual_location": "1:1", "expected_location": "1:1"}] }

    def test_mismatch_prior_to_newline_followed_by_another_newline(self):
        actual = io.StringIO(u"\"\n\nいあ")
        expected = io.StringIO(u"。\n\nいあ")
        result = Evaluation(expected,actual)
        result.evaluate()
        assert result.success == False
        assert result.count == 3
        assert result.failures == { u"。" : [{ "actual" : u"\"", "actual_location": "1:1", "expected_location": "1:1"}] }

    def test_mismatch_prior_to_newline_followed_by_another_newline_windows(self):
        actual = io.StringIO(u"\"\r\n\r\nいあ")
        expected = io.StringIO(u"。\r\n\r\nいあ")
        result = Evaluation(expected,actual)
        result.evaluate()
        assert result.success == False
        assert result.count == 3
        assert result.failures == { u"。" : [{ "actual" : u"\"", "actual_location": "1:1", "expected_location": "1:1"}] }

    def test_out_of_sync_stream(self):
        actual = io.StringIO(u"ぃ　あし\r\n")
        expected = io.StringIO(u"いあし\r\n")
        result = Evaluation(expected,actual)
        result.evaluate()
        assert result.success == False
        assert result.count == 3
        assert result.failures == { u"い" : [{ "actual" : u"ぃ　", "actual_location": "1:1", "expected_location": "1:1"}] }

    def test_out_of_sync_stream_two_deep(self):
        actual = io.StringIO(u"ぃ　'あし\r\n")
        expected = io.StringIO(u"いあし\r\n")
        result = Evaluation(expected,actual)
        result.evaluate()
        assert result.success == False
        assert result.count == 3
        assert result.failures == { u"い" : [{ "actual" : u"ぃ　'", "actual_location": "1:1", "expected_location": "1:1"}] }

    def test_out_of_sync_stream_actual_new_lined_early(self):
        actual = io.StringIO(u"新しい\nしごと")
        expected = io.StringIO(u"新しいむすこ\nしごと\n")
        result = Evaluation(expected,actual)
        result.evaluate()
        assert result.success == False
        assert result.count == 9
        assert result.failures == { u"む" : [{ "actual" : u"NL", "actual_location": "2:0", "expected_location": "1:4"}],
                                    u"す" : [{ "actual" : u"NL", "actual_location": "2:0", "expected_location": "1:5"}],
                                    u"こ" : [{ "actual" : u"NL", "actual_location": "2:0", "expected_location": "1:6"}],
                                    }

    def test_out_of_sync_stream_expected_new_lined_early(self):
        actual = io.StringIO(u"新しいむすこ\nしごと\n")
        expected = io.StringIO(u"新しい\nしごと")
        result = Evaluation(expected,actual)
        result.evaluate()
        assert result.success == False
        assert result.count == 6
        assert result.failures == { u"NL" : [{ "actual" : u"む", "actual_location": "1:4", "expected_location": "2:0"},
                                             { "actual" : u"す", "actual_location": "1:5", "expected_location": "2:0"},
                                             { "actual" : u"こ", "actual_location": "1:6", "expected_location": "2:0"}]
                                   }

    def test_out_of_sync_stream_doesnt_sync_past_newline(self):
        actual = io.StringIO(u"新しいむすあ\nこしごと\n")
        expected = io.StringIO(u"新しいむすこ\nしごと\n")
        result = Evaluation(expected,actual)
        result.evaluate()
        assert result.success == False
        assert result.count == 9
        assert result.failures == { u"こ" : [{ "actual" : u"あ", "actual_location": "1:6", "expected_location": "1:6"}],
                                    u"し" : [{ "actual" : u"こし", "actual_location": "2:1", "expected_location": "2:1"}]
                                    }

    def test_peek_when_empty(self):
        stream = io.StringIO()
        OUT = EvaluationStream(stream)
        assert OUT.iseof(OUT.peek(1))
        assert OUT.iseof(OUT.peek(2))

    def test_peek(self):
        stream = io.StringIO(u"いあし\r\n")
        OUT = EvaluationStream(stream)
        assert u"い" == OUT.peek(1)
        assert "1:0" == OUT.location()
        assert u"あ" == OUT.peek(2)
        assert "1:0" == OUT.location()
        assert u"し" == OUT.peek(3)
        assert "1:0" == OUT.location()
        assert OUT.isnewline(OUT.peek(4))
        assert "1:0" == OUT.location()
        assert OUT.iseof(OUT.peek(5))
        assert "1:0" == OUT.location()

    def test_success_statistics(self):
        actual = io.StringIO(u"ぃ　あしろろる\r\n")
        expected = io.StringIO(u"いあしるろる\r\n")
        result = Evaluation(expected,actual)
        result.evaluate()
        assert result.success == False
        assert result.count == 6
        assert result.failures == { u"い" : [{ "actual" : u"ぃ　", "actual_location": "1:1", "expected_location": "1:1"}],
                                    u"る" : [{ "actual" : u"ろ", "actual_location" : "1:5", "expected_location": "1:4"}]
                                    }
        assert result.successes == {
                                    u"あ" : [{'actual_location': '1:3', 'expected_location': '1:2'}],
                                    u"し" : [{'actual_location': '1:4', 'expected_location': '1:3'}],
                                    u"ろ" : [{'actual_location': '1:6', 'expected_location': '1:5'}],
                                    u"る" : [{'actual_location': '1:7', 'expected_location': '1:6'}],
                                    u"NL" : [{'actual_location': '2:0', 'expected_location': '2:0'}]
                                    }
        result.calculate_percentages()
        print result.percentages
        assert result.percentages == {
                                      u"い" : 0.0,
                                      u"し" : 1.0,
                                      u"る" : 0.5,
                                      u"ろ" : 1.0,
                                      u"NL" : 1.0
                                      }
        assert result.percentage_overall == 0.75

    def test_extra_whitespace(self):
        actual = io.StringIO(u"新 し い むすこ\nし ご と")
        expected = io.StringIO(u"新しいむすこ\nしごと\n")
        result = Evaluation(expected,actual)
        result.evaluate()
        assert result.success == False
        assert result.count == 9
        json.dump(result.failures, sys.stdout, cls=IgnoreUnderscoreEncoder, ensure_ascii=False, indent=2, separators=(',', ': '))
        assert result.failures == { u"し" : [{ "actual" : u" ", "actual_location": "1:2", "expected_location": "1:2"}],
                                    u"い" : [{ "actual" : u" ", "actual_location": "1:4", "expected_location": "1:3"}],
                                    u"む" : [{ "actual" : u" ", "actual_location": "1:6", "expected_location": "1:4"}],
                                    u"ご" : [{ "actual" : u" ", "actual_location": "2:2", "expected_location": "2:2"}],
                                    u"と" : [{ "actual" : u" ", "actual_location": "2:4", "expected_location": "2:3"}],
                                    }
