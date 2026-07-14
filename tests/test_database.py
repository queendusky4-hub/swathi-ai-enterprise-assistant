from swathi_ai.database import ChatRepository

def test_save_load_delete(tmp_path):
    repo = ChatRepository(tmp_path / "chat.db")
    repo.save("s1", "user", "hello")
    assert repo.load("s1") == [("user", "hello")]
    repo.delete("s1")
    assert repo.load("s1") == []
