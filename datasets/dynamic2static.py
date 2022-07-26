import os
import sys
import pickle as pkl
from argparse import ArgumentParser, RawTextHelpFormatter

########################################################################################################################

parser = ArgumentParser(description="Examples: \n", formatter_class=RawTextHelpFormatter)
parser.add_argument(
    '--input_folder', type=str, required=True, help='Path of the dataset folder'
)
parser.add_argument(
    '--output_path', type=str, required=True, help='Path of the output edge list file'
)
args = parser.parse_args()
input_folder = args.input_folder
output_path = args.output_path

########################################################################################################################

pairs_file_path = os.path.join(input_folder, 'pairs.pkl')
with open(pairs_file_path, 'rb') as f:
    pairs = pkl.load(f)

with open(output_path, 'w') as f:
    for pair in pairs:
        f.write(f"{int(pair[0])} {int(pair[1])}\n")