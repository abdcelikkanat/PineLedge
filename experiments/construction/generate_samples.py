import os
import pickle
import utils
import numpy as np
from argparse import ArgumentParser, RawTextHelpFormatter
from multiprocessing import Pool

########################################################################################################################

seed = None
r = None
nodes_num = None
all_events = None

########################################################################################################################

parser = ArgumentParser(description="Examples: \n", formatter_class=RawTextHelpFormatter)
parser.add_argument(
    '--dataset_folder', type=str, required=True, help='Path of the dataset folder'
)
parser.add_argument(
    '--samples_file_folder', type=str, required=True, help='Path of the output folder storing sample files'
)
parser.add_argument(
    '--radius', type=float, default=0.001, required=False, help='The half length of the interval'
)
parser.add_argument(
    '--threads', type=int, required=False, default=64, help='Number of threads'
)
parser.add_argument(
    '--seed', type=int, default=19, required=False, help='Seed value'
)

########################################################################################################################

args = parser.parse_args()
dataset_folder = args.dataset_folder
samples_file_folder = args.samples_file_folder
r = args.radius
threads_num = args.threads
seed = args.seed
utils.set_seed(seed=seed)

if __name__ == '__main__':

    ####################################################################################################################

    # Load the dataset
    all_events = utils.load_dataset(dataset_folder, seed=seed)
    nodes_num = all_events.number_of_nodes()

    ####################################################################################################################

    with Pool(threads_num, initializer=utils.init_worker, initargs=(r, nodes_num, all_events)) as p:
        output = p.map(utils.generate_pos_samples, zip(all_events.get_pairs(), all_events.get_events()))
    pos_samples = [value for sublist in output for value in sublist]

    with Pool(threads_num, initializer=utils.init_worker, initargs=(r, nodes_num, all_events)) as p:
        output = p.map(utils.generate_neg_samples, np.random.uniform(size=(len(pos_samples),)).tolist())
    neg_samples = output

    ####################################################################################################################

    # If folder does not exits
    if not os.path.exists(samples_file_folder):
        os.makedirs(samples_file_folder)

    pos_samples_file_path = os.path.join(samples_file_folder, "pos.samples")
    with open(pos_samples_file_path, 'wb', ) as f:
        pickle.dump(pos_samples, f, pickle.HIGHEST_PROTOCOL)

    neg_samples_file_path = os.path.join(samples_file_folder, "neg.samples")
    with open(neg_samples_file_path, 'wb', ) as f:
        pickle.dump(neg_samples, f, pickle.HIGHEST_PROTOCOL)
