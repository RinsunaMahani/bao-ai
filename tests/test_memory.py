from bao.ai.memory import ConversationMemory


def test_empty_memory_has_no_context():
    memory = ConversationMemory()
    assert memory.as_context() == ""
    assert len(memory) == 0


def test_memory_records_turns():
    memory = ConversationMemory()
    memory.add("user", "hello", language="English")
    memory.add("assistant", "hi there", language="English")
    assert len(memory) == 2
    context = memory.as_context()
    assert "hello" in context
    assert "hi there" in context


def test_memory_window_is_bounded():
    memory = ConversationMemory(max_turns=2)
    for i in range(10):
        memory.add("user", f"message {i}")
        memory.add("assistant", f"reply {i}")
    # max_turns=2 means at most 4 stored turns (2 user+assistant pairs)
    assert len(memory) == 4
    context = memory.as_context()
    assert "message 9" in context  # most recent kept
    assert "message 0" not in context  # oldest evicted


def test_clear_resets_memory():
    memory = ConversationMemory()
    memory.add("user", "hello")
    memory.clear()
    assert len(memory) == 0


def test_one_long_reply_does_not_bloat_every_later_prompt():
    """The window bounded how MANY turns were carried and nothing bounded
    how LARGE each was - and it is the size that grows.

    The security guardrail caps user input at 500 characters, but a
    generated reply is uncapped; a measured one ran to 2,059. Six such
    exchanges prepended 12,656 characters, roughly 3,200 tokens, to every
    subsequent request - paid again on each turn against a small free-tier
    allowance, and delaying the first word, which is the part an audience
    watches.
    """
    memory = ConversationMemory()
    for _ in range(6):
        memory.add("user", "explain calculus in xitsonga")
        memory.add("assistant", "A" * 2059)

    context = memory.as_context()
    assert len(context) < 6000, f"context is {len(context)} chars"


def test_the_cut_is_visible_to_the_model():
    """A truncated turn must not read as a complete previous answer."""
    memory = ConversationMemory()
    memory.add("assistant", "A" * 2000)
    assert memory.turns[0].content.endswith("[…]")


def test_short_turns_are_kept_whole():
    """Most exchanges are well under the cap and must be untouched -
    resolving "and in Afrikaans?" depends on reading them exactly.
    """
    memory = ConversationMemory()
    memory.add("user", "Avuxeni")
    memory.add("assistant", "Avuxeni! Ndzi nga ku pfuna njhani?")
    assert memory.turns[0].content == "Avuxeni"
    assert memory.turns[1].content == "Avuxeni! Ndzi nga ku pfuna njhani?"


def test_the_per_turn_cap_can_be_switched_off():
    memory = ConversationMemory(max_chars_per_turn=0)
    memory.add("assistant", "A" * 2000)
    assert len(memory.turns[0].content) == 2000
