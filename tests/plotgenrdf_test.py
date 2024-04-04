import os


def test_plotrdf():
    REPO_HOME = os.environ["REPO_HOME"]
    fit_results = f"{REPO_HOME}/tests/samples/SIMPLE_EXAMPLE/result.fit"
    output_file = f"{REPO_HOME}/tests/plotgenrdf_test"  # leave out ftype as program appends .pdf type
    cmd = f"python {REPO_HOME}/scripts/plotgenrdf.py {fit_results} -o {output_file}"
    print(cmd)
    os.system(cmd)
    assert os.path.exists(f"{output_file}_all.png"), f"scripts/plotgenrdf.py failed to generate {output_file}_all.png"
    os.system(f"rm {output_file}_all.png")
