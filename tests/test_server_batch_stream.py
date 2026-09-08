"""Per-stream detokenization and stop strings for the batched path.

BatchRequest.stop_ids is token ids; the client's `stop` is strings. Tokenizing
the strings to bridge them is wrong at merge boundaries, so matching happens
on decoded text -- and it must trim exactly where HF trims on the serial path,
or the same request returns different text depending on whether it happened to
be batched.
"""
from vaayu.server.engine import _BatchStream, _Detok


class CharTokenizer:
    """One id per character: id == ord(c). Decoding is concatenation, which is
    enough to exercise deltas, trimming and retirement."""

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


class PartialTokenizer:
    """Emits the replacement char until a pair of ids is complete -- the
    incomplete-multibyte case a per-token decode would leak to the client."""

    def decode(self, ids, skip_special_tokens=True):
        if len(ids) % 2:
            return "ok�"
        return "ok" + "!" * (len(ids) // 2)


def test_detok_emits_only_the_new_text():
    d = _Detok(CharTokenizer())
    assert d.feed(ord("a")) == "a"
    assert d.feed(ord("b")) == "b"
    assert d.feed(ord("c")) == "c"


def test_detok_holds_back_an_incomplete_character():
    d = _Detok(PartialTokenizer())
    assert d.feed(1) == "", "a replacement char must never reach the client"
    assert d.feed(2) == "ok!"


def test_stream_emits_deltas_and_records_ids():
    s = _BatchStream(job=None, tokenizer=CharTokenizer(), stops=[], cap=10)
    assert s.feed(ord("h")) == "h"
    assert s.feed(ord("i")) == "i"
    assert s.ids == [ord("h"), ord("i")]
    assert s.text == "hi"
    assert s.retired is False


def test_a_stop_string_trims_and_retires():
    s = _BatchStream(job=None, tokenizer=CharTokenizer(), stops=["END"], cap=50)
    assert s.feed(ord("a")) == "a"
    s.feed(ord("E"))
    s.feed(ord("N"))
    assert s.feed(ord("D")) == "", "nothing from the stop string may be sent"
    assert s.retired is True
    assert s.text == "a", "text is trimmed AT the stop, as HF does"


def test_a_stop_string_split_across_tokens_still_matches():
    s = _BatchStream(job=None, tokenizer=CharTokenizer(), stops=["<|end|>"], cap=50)
    for ch in "hi<|end":
        s.feed(ord(ch))
    assert s.retired is False
    s.feed(ord("|"))
    s.feed(ord(">"))
    assert s.retired is True
    assert s.text == "hi"


def test_the_earliest_stop_wins_when_several_match():
    s = _BatchStream(job=None, tokenizer=CharTokenizer(), stops=["cd", "bc"],
                     cap=50)
    for ch in "abcd":
        s.feed(ord(ch))
    assert s.text == "a"
    assert s.retired is True


def test_reaching_the_cap_retires_the_stream():
    s = _BatchStream(job=None, tokenizer=CharTokenizer(), stops=[], cap=3)
    for ch in "abc":
        s.feed(ord(ch))
    assert s.retired is True
    assert s.text == "abc"


def test_text_already_sent_is_never_re_emitted():
    s = _BatchStream(job=None, tokenizer=CharTokenizer(), stops=["zz"], cap=50)
    sent = "".join(s.feed(ord(ch)) for ch in "abcz")
    assert sent == "abcz"
    assert s.feed(ord("z")) == ""
    assert s.text == "abc", "the trim retracts the sent tail from the FINAL text"
