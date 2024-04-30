import os


def test_plotgen():
    PYAMPTOOLS_HOME = os.environ["PYAMPTOOLS_HOME"]
    fit_results = f"{PYAMPTOOLS_HOME}/tests/samples/SIMPLE_EXAMPLE/result.fit"
    output_root_file = f"{PYAMPTOOLS_HOME}/tests/plotgen_test.root"
    cmd = f"python {PYAMPTOOLS_HOME}/scripts/plotgen.py {fit_results} -o {output_root_file}"
    print(cmd)
    os.system(cmd)
    assert os.path.exists(output_root_file) and os.path.getsize(output_root_file) > 1000
    os.system(f"rm {output_root_file}")
