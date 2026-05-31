def test_evaluation_uses_non_interactive_matplotlib_backend() -> None:
    import src.evaluate  # noqa: F401
    import matplotlib

    assert matplotlib.get_backend().lower() == "agg"
