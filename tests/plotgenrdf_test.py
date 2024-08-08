import os


def test_plotrdf():
    PYAMPTOOLS_HOME = os.environ["PYAMPTOOLS_HOME"]
    fit_results = f"{PYAMPTOOLS_HOME}/tests/samples/SIMPLE_EXAMPLE/result.fit"
    output_file = f"{PYAMPTOOLS_HOME}/tests/plotgenrdf_test"  # leave out ftype as program appends .pdf type
    cmd = f"python {PYAMPTOOLS_HOME}/scripts/plotgenrdf.py {fit_results} -o {output_file}"
    print(cmd)
    os.system(cmd)
    assert os.path.exists(f"{output_file}_all.pdf"), f"scripts/plotgenrdf.py failed to generate {output_file}_all.pdf"
    os.system(f"rm {output_file}_all.pdf")
