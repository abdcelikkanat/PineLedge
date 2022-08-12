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
    events = utils.load_dataset(dataset_folder, seed=seed)

    pairs = np.asarray(events.get_pairs(), dtype=int)
    pair_events = np.asarray(events.get_events())

    with open(info_path, 'w') as f:
        f.write("+ The given network statistics.\n")
        f.write(f"\t It contains {events.number_of_nodes()} nodes.\n")
        f.write(f"\t It contains {events.number_of_event_pairs()} pairs having at least one link.\n")
        f.write(f"\t It contains {events.number_of_total_events()} links in total.\n")

    ####################################################################################################################
    with open(info_path, 'w') as f:
        f.write("+ The network is being split into two parts.\n")
    # Split the dataset
    all_event_times = [e for pair_events in events.get_events() for e in pair_events]
    all_event_times.sort()
    split_time = split_ratio  #all_event_times[math.floor(all_events.number_of_total_events() * split_ratio)]

    first_part_pairs, first_part_events = [], []
    second_part_pairs, second_part_events = [], []
    for pair in events.get_pairs():

        first_part_pairs.append(pair)
        first_part_events.append([])
        second_part_pairs.append(pair)
        second_part_events.append([])

        for e in events[pair][1]:
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

    with open(info_path, 'a+') as f:
        f.write(f"\t- The first set contains {len(np.unique(first_part_pairs))} unique nodes.\n")
        f.write(f"\t- The first set has {len(first_part_pairs)} pairs having events.\n")
        f.write(f"\t- The second set contains {len(np.unique(second_part_pairs))} unique nodes.\n")
        f.write(f"\t- The second set has {len(second_part_pairs)} pairs having events.\n")

    if len(np.unique(pairs)) != len(np.unique(first_part_pairs)):
        with open(info_path, 'a+') as f:
            f.write("+ The first set contains less number of nodes than the original network.\n")
            f.write(f"\t- Extra nodes are being removed and relabeling is being done.\n")

        selected_nodes = np.unique(first_part_pairs)
        node2newlabel = dict(zip(selected_nodes, range(len(selected_nodes))))

        # Relabel the nodes and remove the nodes in the second set which do not appear in the first set
        idx = 0
        while idx < len(first_part_pairs):
            pair = first_part_pairs[idx]
            first_part_pairs[idx][0], first_part_pairs[idx][1] = node2newlabel[pair[0]], node2newlabel[pair[1]]
            idx += 1
        idx = 0
        while idx < len(second_part_pairs):

            if second_part_pairs[idx][0] not in selected_nodes or second_part_pairs[idx][1] not in selected_nodes:
                second_part_pairs.pop(idx)
                second_part_events.pop(idx)

            else:
                pair = second_part_pairs[idx]
                second_part_pairs[idx][0], second_part_pairs[idx][1] = node2newlabel[pair[0]], node2newlabel[pair[1]]
                idx += 1
    nodes_num = len(np.unique(first_part_pairs))
    ####################################################################################################################
    # Training/testing/validation sample computation in the first set
    with open(info_path, 'a+') as f:
        f.write("+ The training/testing/validation sets construction has started out from the first set.\n")

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
    test_valid_pairs = first_part_pairs[train_samples_num:]
    test_valid_events = first_part_events[train_samples_num:]

    with open(info_path, 'a+') as f:
        f.write("\t- The number of pairs in the training set: {}\n".format(len(train_pairs)))
        f.write("\t- It contains {} events in total.\n".format(sum([len(el) for el in train_events])))
        f.write("\t- The number of pairs in the valid + test set: {}\n".format(len(test_valid_pairs)))
        f.write("\t- It contains {} events in total.\n".format(sum([len(el) for el in test_valid_events])))
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
    # Generate samples for the testing set of the network reconstruction experiment
    first_part_pairs, first_part_events = shuffle(first_part_pairs, first_part_events)
    train_pairs, train_events = shuffle(train_pairs, train_events)

    first_part = Events(data=(first_part_events, first_part_pairs, range(nodes_num)))
    train = Events(data=(train_events, train_pairs, range(nodes_num)))
    test_valid = Events(data=(test_valid_events, test_valid_pairs, range(nodes_num)))

    # Generate positive samples for network reconstruction
    with Pool(threads_num, initializer=utils.init_worker, initargs=(r, nodes_num, train)) as p:
        output = p.starmap(
            utils.generate_pos_samples,
            zip(
                np.random.randint(0, 1e4, size=(len(train_pairs), )),
                first_part_pairs,
                first_part_events,
                [0.] * len(train_pairs), [split_time] * len(train_pairs)
            )
        )
    train_pos_samples = [value for sublist in output for value in sublist]

    # Generate positive samples for test/valid reconstruction
    with Pool(threads_num, initializer=utils.init_worker, initargs=(r, nodes_num, test_valid)) as p:
        output = p.starmap(
            utils.generate_pos_samples,
            zip(
                np.random.randint(0, 1e4, size=(len(test_valid_pairs),)),
                first_part_pairs,
                first_part_events,
                [0.] * len(test_valid_pairs), [split_time] * len(test_valid_pairs)
            )
        )
    test_pos_samples = [value for sublist in output[:len(output)//2] for value in sublist]
    valid_pos_samples = [value for sublist in output[len(output)//2:] for value in sublist]

    # Generate negative samples for all train/test/valid reconstruction
    with Pool(threads_num, initializer=utils.init_worker, initargs=(r, nodes_num, first_part)) as p:
        output = p.starmap(
            utils.generate_neg_samples,
            zip(
                np.random.randint(0, 1e4, size=(len(train_pos_samples)+len(test_pos_samples)+len(valid_pos_samples),)),
                np.random.uniform(low=0.0, high=split_time, size=(len(train_pos_samples)+len(test_pos_samples)+len(valid_pos_samples),)).tolist(),
                [0.] * (len(train_pos_samples)++len(test_pos_samples)+len(valid_pos_samples)),
                [split_time] * (len(train_pos_samples)+len(test_pos_samples)+len(valid_pos_samples))
            )
        )
    train_neg_samples, test_valid_neg_samples = output[:len(train_pos_samples)], output[len(train_pos_samples):]
    test_neg_samples, valid_neg_samples = test_valid_neg_samples[:len(test_pos_samples)], test_valid_neg_samples[len(test_pos_samples):]

    with open(info_path, 'a+') as f:
        f.write(f"+ Generated samples\n")
        f.write(f"\t- Positive samples in the train set: {len(train_pos_samples)}\n")
        f.write(f"\t- Positive samples in the test set: {len(test_pos_samples)}\n")
        f.write(f"\t- Positive samples in the valid set: {len(valid_pos_samples)}\n")
        f.write(f"\t- Negative samples in the train set: {len(train_neg_samples)}\n")
        f.write(f"\t- Negative samples in the test set: {len(test_neg_samples)}\n")
        f.write(f"\t- Negative samples in the valid set: {len(valid_neg_samples)}\n")

    ####################################################################################################################

    # Save the pairs and events
    pairs_file_path = os.path.join(output_folder, 'pairs.pkl')
    with open(pairs_file_path, 'wb') as f:
        pickle.dump(train_pairs, f, protocol=pickle.HIGHEST_PROTOCOL)

    events_file_path = os.path.join(output_folder, 'events.pkl')
    with open(events_file_path, 'wb') as f:
        pickle.dump(train_events, f, protocol=pickle.HIGHEST_PROTOCOL)

    # For Reconstruction
    reconstruction_folder_path = os.path.join(output_folder, "reconstruction")
    if not os.path.exists(reconstruction_folder_path):
        os.makedirs(reconstruction_folder_path)

    pos_sample_file_path = os.path.join(reconstruction_folder_path, 'pos.samples')
    with open(pos_sample_file_path, 'wb') as f:
        pickle.dump(train_pos_samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    neg_sample_file_path = os.path.join(reconstruction_folder_path, 'neg.samples')
    with open(neg_sample_file_path, 'wb') as f:
        pickle.dump(train_neg_samples, f, protocol=pickle.HIGHEST_PROTOCOL)

    # For Completion
    completion_folder_path = os.path.join(output_folder, "completion")
    if not os.path.exists(completion_folder_path):
        os.makedirs(completion_folder_path)

    # Validation samples
    pos_sample_file_path = os.path.join(completion_folder_path, 'valid_pos.samples')
    with open(pos_sample_file_path, 'wb') as f:
        pickle.dump(valid_pos_samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    neg_sample_file_path = os.path.join(completion_folder_path, 'valid_neg.samples')
    with open(neg_sample_file_path, 'wb') as f:
        pickle.dump(valid_neg_samples, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Testing samples
    pos_sample_file_path = os.path.join(completion_folder_path, 'test_pos.samples')
    with open(pos_sample_file_path, 'wb') as f:
        pickle.dump(test_pos_samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    neg_sample_file_path = os.path.join(completion_folder_path, 'test_neg.samples')
    with open(neg_sample_file_path, 'wb') as f:
        pickle.dump(test_neg_samples, f, protocol=pickle.HIGHEST_PROTOCOL)

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
