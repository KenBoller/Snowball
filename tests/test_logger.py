import logging

def test_logger_initializes_and_logs():
    from core.system.logger import SnowballLogger

    log = SnowballLogger(name="snowball_test_logger")
    log.info("test info")
    log.warning("test warning")
    log.error("test error")

    assert log is not None

def test_logger_no_duplicate_handlers_on_reinit():
    from core.system.logger import SnowballLogger

    a = SnowballLogger(name="snowball_test_dupe")
    handler_count_1 = len(a._logger.handlers)

    b = SnowballLogger(name="snowball_test_dupe")
    handler_count_2 = len(b._logger.handlers)

    # should not add new handlers
    assert handler_count_2 == handler_count_1

def test_env_log_level_respected(monkeypatch):
    from core.system.logger import SnowballLogger
    monkeypatch.setenv("SNOWBALL_LOG_LEVEL", "DEBUG")

    log = SnowballLogger(name="snowball_test_level")
    assert log._logger.level == logging.DEBUG
