import pathlib


def test_readme_mentions_scored_evals():
    text = pathlib.Path("README.md").read_text()
    assert "scored evals" in text.lower()
