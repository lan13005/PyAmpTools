import os


def test_environ_is_set():
    assert os.environ["REPO_HOME"] != ""
    assert os.environ["AMPTOOLS_HOME"] != ""
    assert os.environ["AMPPLOTTER"] != ""
    assert os.environ["AMPTOOLS"] != ""
    assert os.environ["FSROOT"] != ""
    assert os.environ["ROOTSYS"] != ""
