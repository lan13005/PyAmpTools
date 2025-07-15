import argparse
import pickle as pkl
from omegaconf.dictconfig import DictConfig

# NOTE: This script will break if there are repeated keys in the pkl file. Even if the parent keys are different.
# to be more robust I think the user has to specify the key path in the pkl file and directly replace that value

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("result_pkl", type=str, help="Path to the NIFTy result pickle file")
    parser.add_argument("key_to_find", type=str, help="Key to replace in the pickle file")
    parser.add_argument("update_value", type=str, help="Value to replace the key with")
    parser.add_argument("-o", "--output_pkl", type=str, default=None, help="Path to the output pickle file, if not provided, the input file will be overwritten")
    args = parser.parse_args()

    result_pkl = args.result_pkl
    key_to_find = args.key_to_find
    update_value = args.update_value
    output_pkl = args.output_pkl

    with open(result_pkl, "rb") as f:
        print(f"Loading pickle file from: {result_pkl}")
        data = pkl.load(f)

    def find_nested_key(dictionary, key_to_find, update_value):
        """
        Recursively search for all occurrences of key_to_find in a nested dictionary
        and update their values with update_value.
        """
        def recursive_search(d, key_chain):
            if isinstance(d, (dict, DictConfig)):  # Ensure it's a dictionary-like object
                for key, value in list(d.items()):  # Use list() to avoid modification issues
                    new_key_chain = key_chain + [key]  # Track full key path
                    
                    if key == key_to_find:
                        keys_to_update.append(new_key_chain)  # Store full key path
                    elif isinstance(value, (dict, DictConfig)):  # If value is a nested dictionary, recurse
                        recursive_search(value, new_key_chain)

        keys_to_update = []  # Stores paths to keys that need updating

        recursive_search(dictionary, [])  # Start recursion

        # Apply updates based on stored key paths
        for key_chain in keys_to_update:
            d = dictionary  # Start from root dictionary
            for key in key_chain[:-1]:  # Traverse down to the second last key
                d = d[key]  # Access next level of dictionary
            d[key_chain[-1]] = update_value  # Update the final key

    print(f"  Recursively searching for {key_to_find} in {result_pkl}")
    find_nested_key(data, key_to_find, update_value)

    if output_pkl is None:
        output_pkl = result_pkl

    with open(output_pkl, "wb") as f:
        print(f"  Dumping updated pickle file to: {output_pkl}")
        pkl.dump(data, f)
