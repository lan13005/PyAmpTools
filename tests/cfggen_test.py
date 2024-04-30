import os


def test_cfgGenerator():
    PYAMPTOOLS_HOME = os.environ["PYAMPTOOLS_HOME"]
    SCRIPTS = f"{PYAMPTOOLS_HOME}/scripts"
    cmd = [f"python {SCRIPTS}/cfggen.py"]
    output_file = "EtaPi.cfg"
    cmd = " ".join(cmd)
    print(cmd)
    os.system(cmd)
    assert os.path.exists(output_file) and os.path.getsize(output_file) > 0
    os.system(f"rm {output_file}")
