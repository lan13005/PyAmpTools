import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize the contents of an iftpwa results pickle file")
    parser.add_argument("pkl_file", type=str, help="Path to the pickle file")
    args = parser.parse_args()
    pkl_file = args.pkl_file
    
    os.system(f"iftPwaSummarize {pkl_file}")
