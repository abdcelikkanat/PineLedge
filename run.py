import pickle
import torch
from argparse import ArgumentParser, RawTextHelpFormatter
from src.learning import LearningModel
from src.events import Events

# Global control for device
CUDA = True
device = "cuda:0" if torch.cuda.is_available() else "cpu"
if (CUDA) and (device == "cuda:0"):
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def parse_arguments():
    parser = ArgumentParser(description="Examples: \n",
                            formatter_class=RawTextHelpFormatter)

    parser.add_argument(
        '--dataset', type=str, required=True, help='Path of the dataset'
    )
    parser.add_argument(
        '--model_path', type=str, required=True, help='Path of the model'
    )
    parser.add_argument(
        '--log', type=str, required=False, default=None, help='Path of the log file'
    )
    parser.add_argument(
        '--bins_num', type=int, default=100, required=False, help='Number of bins'
    )
    parser.add_argument(
        '--dim', type=int, default=2, required=False, help='Dimension size'
    )
    parser.add_argument(
        '--k', type=int, default=10, required=False, help='Latent dimension size of the prior element'
    )
    parser.add_argument(
        '--prior_lambda', type=float, default=1e5, required=False, help='Scaling coefficient of the covariance'
    )
    parser.add_argument(
        '--epoch_num', type=int, default=100, required=False, help='Number of epochs'
    )
    parser.add_argument(
        '--spe', type=int, default=1, required=False, help='Number of steps per epoch'
    )
    parser.add_argument(
        '--batch_size', type=int, default=0, required=False, help='Batch size'
    )
    parser.add_argument(
        '--lr', type=float, default=0.1, required=False, help='Learning rate'
    )
    parser.add_argument(
        '--seed', type=int, default=19, required=False, help='Seed value to control the randomization'
    )
    parser.add_argument(
        '--verbose', type=bool, default=1, required=False, help='Verbose'
    )

    return parser.parse_args()


def process(args):

    dataset_path = args.dataset
    model_path = args.model_path
    log_file_path = args.log

    bins_num = args.bins_num
    dim = args.dim
    K = args.k
    prior_lambda = args.prior_lambda
    epoch_num = args.epoch_num
    steps_per_epoch = args.spe
    batch_size = args.batch_size
    learning_rate = args.lr

    seed = args.seed
    verbose = args.verbose

    # Load the dataset
    all_events = Events(seed=seed)
    all_events.read(dataset_path)
    if batch_size <= 0:
        batch_size = all_events.number_of_nodes()

    # Normalize the events
    all_events.normalize(init_time=0, last_time=1.0)

    # Get the number of nodes
    nodes_num = all_events.number_of_nodes()
    data = all_events.get_pairs(), all_events.get_events()

    if verbose:
        print(f"- The dataset statistics:")
        print(f"\t+ The number of nodes: {all_events.number_of_nodes()}")
        print(f"\t+ The total number of edges: {all_events.number_of_total_events()}")
        print(f"\t+ The number of pairs having events: {all_events.number_of_event_pairs()}")
        print(f"\t+ Average number of events: {all_events.number_of_total_events()/all_events.number_of_event_pairs()}")

    if verbose:
        print(f"- The active device is {device}.")

    # Run the model
    lm = LearningModel(
        data=data, nodes_num=nodes_num, bins_num=bins_num, dim=dim, last_time=1., batch_size=batch_size,
        prior_k=K, prior_lambda=prior_lambda,
        learning_rate=learning_rate, epoch_num=epoch_num, steps_per_epoch=steps_per_epoch,
        verbose=verbose, seed=seed, device=torch.device(device)
    )
    lm.learn(loss_file_path=log_file_path)

    if verbose:
        print(f"- Model file is saving.")
        print(f"\t+ Target path: {model_path}")

    torch.set_default_tensor_type('torch.FloatTensor')
    lm = lm.cpu()
    with open(model_path, 'wb') as f:
        pickle.dump(lm.cpu(), f, pickle.HIGHEST_PROTOCOL)
    if verbose:
        print(f"\t+ Completed.")


if __name__ == "__main__":
    args = parse_arguments()
    process(args)