import os
import numpy as np
import math
import pickle
import utils
from sklearn.utils import shuffle
from src.events import Events
from argparse import ArgumentParser, RawTextHelpFormatter
from multiprocessing import Pool
import time


def generate_numbers(id):

    for _ in range(4):
        nums = np.random.randint(low=0, high=100, size=(10, )).tolist()
        print(id, nums)

########################################################################################################################

r = None
nodes_num = None
all_events = None

########################################################################################################################

parser = ArgumentParser(description="Examples: \n", formatter_class=RawTextHelpFormatter)
parser.add_argument(
    '--dataset_folder', type=str, required=True, help='Path of the original dataset'
)
parser.add_argument(
    '--output_folder', type=str, required=True, help='Output folder to store samples and residual network'
)
parser.add_argument(
    '--radius', type=float, required=False, default=0.001, help='The half length of the interval'
)
parser.add_argument(
    '--train_ratio', type=float, required=False, default=0.95, help='Training set ratio'
)
parser.add_argument(
    '--split_ratio', type=float, required=False, default=0.9, help='Training set ratio'
)
parser.add_argument(
    '--threads', type=int, required=False, default=64, help='Number of threads'
)
parser.add_argument(
    '--seed', type=int, default=19, required=False, help='Seed value'
)
########################################################################################################################

# Set some parameters
args = parser.parse_args()
dataset_folder = args.dataset_folder
output_folder = args.output_folder
r = args.radius
train_ratio = args.train_ratio
split_ratio = args.split_ratio
threads_num = args.threads
seed = args.seed
utils.set_seed(seed=seed)
info_path = os.path.join(output_folder, "info.txt")

if __name__ == '__main__':

    ####################################################################################################################

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    ####################################################################################################################
    # Load the dataset
    all_events = utils.load_dataset(dataset_folder, seed=seed)
    nodes_num = all_events.number_of_nodes()

    pairs = np.asarray(all_events.get_pairs(), dtype=int)
    pair_events = np.asarray(all_events.get_events())

    ####################################################################################################################

    # Split the dataset
    all_event_times = [e for pair_events in all_events.get_events() for e in pair_events]
    all_event_times.sort()
    split_time = all_event_times[math.floor(all_events.number_of_total_events() * split_ratio)]

    first_part_pairs, first_part_events = [], []
    second_part_pairs, second_part_events = [], []
    for pair in all_events.get_pairs():

        first_part_pairs.append(pair)
        first_part_events.append([])
        second_part_pairs.append(pair)
        second_part_events.append([])

        for e in all_events[pair][1]:
            if e < split_time:
                first_part_events[-1].append(e)
            else:
                second_part_events[-1].append(e)

        if len(first_part_events[-1]) == 0:
            first_part_pairs.pop()
            first_part_events.pop()

        if len(second_part_events[-1]) == 0:
            second_part_pairs.pop()
            second_part_events.pop()

    assert len(np.unique(first_part_pairs)) == nodes_num, "In the first set, there are less number of nodes!"

    with open(info_path, 'w') as f:
        f.write("+ Residual network has less number of events than the original network.\n")
        f.write("\t- All pairs having at least one event: {}\n".format(all_events.number_of_event_pairs()))
        f.write("\t- Pairs having at least one event in the first set: {}\n".format(len(first_part_pairs)))
        f.write("\t- Pairs having at least one event in the second set: {}\n".format(len(second_part_events)))

    ####################################################################################################################

    # Compute the number of training/testing/validation samples
    event_pairs_num = len(first_part_pairs)
    train_samples_num = int(event_pairs_num * train_ratio)
    train_samples_num = train_samples_num if train_samples_num % 2 == 0 else train_samples_num - 1
    test_valid_event_pairs_num = event_pairs_num - train_samples_num

    # Train samples
    first_part_pairs, first_part_events = shuffle(first_part_pairs, first_part_events, random_state=seed)
    train_pairs = first_part_pairs[:train_samples_num]
    # Keep the number of nodes same
    while len(np.unique(train_pairs)) != nodes_num:
        first_part_pairs, first_part_events = shuffle(first_part_pairs, first_part_events, random_state=seed)
        train_pairs = first_part_pairs[:train_samples_num]
    train_events = first_part_events[:train_samples_num]

    # Residual samples
    test_valid_pairs = pairs[train_samples_num:]
    test_valid_events = pair_events[train_samples_num:]

    with open(info_path, 'a+') as f:
        f.write("+ Validation and testing set generation from the first set.\n")
        f.write("\t- The number of pairs in the training set: {}\n".format(len(train_pairs)))
        f.write("\t- The number of pairs in the valid + test set: {}\n".format(len(test_valid_pairs)))

    ####################################################################################################################

    # Remove the valid/test pairs existing in the prediction
    # The prediction set must not contain the event pair
    with open(info_path, 'a+') as f:
        f.write("+ Removing extra pairs in the second set.\n".format(len(second_part_pairs)))
    # print(pred_test_pairs)
    for pair in test_valid_pairs:

        idx = 0
        while idx < len(second_part_pairs):
            if second_part_pairs[idx][0] == pair[0] and second_part_pairs[idx][1] == pair[1]:
                second_part_pairs.pop(idx)
                second_part_events.pop(idx)
                break
            idx += 1

    with open(info_path, 'a+') as f:
        f.write("\t- {} pairs have remained since test/valid pairs considered in the first set were deleted.\n".format(
            len(second_part_pairs))
        )

    ####################################################################################################################

    residual_events = Events(data=(first_part_events, first_part_pairs, range(nodes_num)))
    # Generate samples for the testing set
    with Pool(threads_num, initializer=utils.init_worker, initargs=(r, nodes_num, residual_events)) as p:
        output = p.map(utils.generate_pos_samples, zip(first_part_pairs, first_part_events))
    pos_samples = [value for sublist in output for value in sublist]

    with Pool(threads_num, initializer=utils.init_worker, initargs=(r, nodes_num, residual_events)) as p:
        output = p.starmap(
            utils.generate_neg_samples,
            zip(
                np.random.uniform(low=0.0, high=split_time, size=(len(pos_samples),)).tolist(),
                [0.] * len(pos_samples), [split_time] * len(pos_samples)
            )
        )
    neg_samples = output

    with open(info_path, 'a+') as f:
        f.write(f"+ Total number of events in the first set: {len(first_part_pairs)}\n")
        f.write(f"\t- Positive samples in the first set: {len(pos_samples)}\n")
        f.write(f"\t- Negative samples in the first set: {len(neg_samples)}\n")

    ####################################################################################################################

    # Save the pairs and events
    pairs_file_path = os.path.join(output_folder, 'pairs.pkl')
    with open(pairs_file_path, 'wb') as f:
        pickle.dump(train_pairs, f, protocol=pickle.HIGHEST_PROTOCOL)

    events_file_path = os.path.join(output_folder, 'events.pkl')
    with open(events_file_path, 'wb') as f:
        pickle.dump(train_events, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save the samples
    l = len(pos_samples) // 2

    # Validation samples
    pos_sample_file_path = os.path.join(output_folder, 'valid_pos.samples')
    with open(pos_sample_file_path, 'wb') as f:
        pickle.dump(pos_samples[:l], f, protocol=pickle.HIGHEST_PROTOCOL)
    neg_sample_file_path = os.path.join(output_folder, 'valid_neg.samples')
    with open(neg_sample_file_path, 'wb') as f:
        pickle.dump(neg_samples[:l], f, protocol=pickle.HIGHEST_PROTOCOL)

    # Testing samples
    pos_sample_file_path = os.path.join(output_folder, 'test_pos.samples')
    with open(pos_sample_file_path, 'wb') as f:
        pickle.dump(pos_samples[l:], f, protocol=pickle.HIGHEST_PROTOCOL)
    neg_sample_file_path = os.path.join(output_folder, 'test_neg.samples')
    with open(neg_sample_file_path, 'wb') as f:
        pickle.dump(neg_samples[l:], f, protocol=pickle.HIGHEST_PROTOCOL)

    # For prediction
    prediction_folder_path = os.path.join(output_folder, "prediction")
    if not os.path.exists(prediction_folder_path):
        os.makedirs(prediction_folder_path)

    pred_pairs_file_path = os.path.join(prediction_folder_path, "pairs.pkl")
    with open(pred_pairs_file_path, 'wb') as f:
        pickle.dump(second_part_pairs, f, protocol=pickle.HIGHEST_PROTOCOL)

    pred_events_file_path = os.path.join(prediction_folder_path, "events.pkl")
    with open(pred_events_file_path, 'wb') as f:
        pickle.dump(second_part_events, f, protocol=pickle.HIGHEST_PROTOCOL)
