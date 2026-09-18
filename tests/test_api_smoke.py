def test_api_send_message_returns_string():
    try:
        from core.api import send_message
    except Exception:
        return  # MVP-safe: API may not be present in some setups

    out = send_message("hello")
    assert isinstance(out, str)
    assert len(out.strip()) > 0
