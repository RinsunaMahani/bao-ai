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
