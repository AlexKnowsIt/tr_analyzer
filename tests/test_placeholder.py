from optimizer import parse_constraints


def test_parse_constraints():
    tickers = ["AAA", "BBB"]
    constraints = parse_constraints("AAA<=0.2,BBB>=0.1", tickers)
    assert constraints == [(0, "<=", 0.2), (1, ">=", 0.1)]
