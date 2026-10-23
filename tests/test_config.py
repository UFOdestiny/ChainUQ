"""``.env`` loading: file values fill unset variables, the environment wins."""

import os

from chainuq.config import load_env


def test_env_file_precedence(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nCHAINUQ_TEST_A=from_file\nexport CHAINUQ_TEST_B='quoted'\nCHAINUQ_TEST_C=${CHAINUQ_TEST_A}/x\n")
    monkeypatch.setenv("CHAINUQ_TEST_B", "from_env")
    monkeypatch.delenv("CHAINUQ_TEST_A", raising=False)
    monkeypatch.delenv("CHAINUQ_TEST_C", raising=False)
    load_env(env)
    assert os.environ["CHAINUQ_TEST_A"] == "from_file"
    assert os.environ["CHAINUQ_TEST_B"] == "from_env"
    assert os.environ["CHAINUQ_TEST_C"] == "from_file/x"
    for k in ("CHAINUQ_TEST_A", "CHAINUQ_TEST_C"):
        monkeypatch.delenv(k)
