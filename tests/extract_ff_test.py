import os

PYAMPTOOLS_HOME = os.environ["PYAMPTOOLS_HOME"]
SRC = f"{PYAMPTOOLS_HOME}/src/pyamptools"
fit_results = f"{PYAMPTOOLS_HOME}/tests/samples/SIMPLE_EXAMPLE/result.fit"
output_file = f"{PYAMPTOOLS_HOME}/tests/ff.txt"


def test_extract_ff():
    cmd = [
        "pa fitfrac",
        fit_results,
        "--outputfileName",
        output_file,
        "--regex_merge",
        "'.*::(.*)::.*~>\\1'",
        "'.*(.)$~>\\1'",
    ]
    cmd = " ".join(cmd)
    print(cmd)
    os.system(cmd)
    with open(output_file, "r") as f:
        print(f.readlines())
    assert os.path.exists(output_file) and os.path.getsize(output_file) > 0
    os.system(f"rm {output_file}")
