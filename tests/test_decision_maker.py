def test_decision_maker_selects_best_response():
    from core.ai.decision_maker import DecisionMaker

    dm = DecisionMaker(logger=None)

    candidates = {
        "mistral:7b-instruct": "I don't know.",
        "deepseek-r1:14b": "Here are the steps:\n- Step one\n- Step two\nThis directly answers your question.",
    }

    chosen = dm.select_best_response(candidates, user_input="Give me steps to do X", query_type="Factual")
    assert "steps" in chosen.lower()
    assert "don't know" not in chosen.lower()


def test_decision_maker_handles_empty_candidates():
    from core.ai.decision_maker import DecisionMaker

    dm = DecisionMaker(logger=None)
    chosen = dm.select_best_response({}, user_input="hello", query_type="General")
    assert isinstance(chosen, str)
    assert len(chosen) > 0


def test_decision_maker_provider_order_tie_break():
    from core.ai.decision_maker import DecisionMaker

    dm = DecisionMaker(logger=None, provider_order=["a", "b"])

    # Same response text -> same score -> tie -> provider_order decides
    candidates = {"b": "Answer that is long enough to be considered valid and helpful.",
                  "a": "Answer that is long enough to be considered valid and helpful."}

    chosen = dm.select_best_response(candidates, user_input="question", query_type="General")
    assert chosen == candidates["a"]
