import sys
import math
import torch
import numpy as np
from src.nhpp import NHPP
from src.base import BaseModel
from utils.constants import const
from torch.utils.tensorboard import SummaryWriter


class LearningModel(BaseModel, torch.nn.Module):

    def __init__(self, data_loader, nodes_num, bins_num, dim, last_time: float,
                 learning_rate, epochs_num, verbose,
                 seed: int = 0):

        super(LearningModel, self).__init__(
            x0=torch.nn.Parameter(2 * torch.rand(size=(nodes_num, dim)) - 1, requires_grad=False),
            v=torch.nn.Parameter(2 * torch.rand(size=(bins_num, nodes_num, dim)) - 1, requires_grad=False),
            beta=torch.nn.Parameter(2 * torch.rand(size=(nodes_num, )) - 1, requires_grad=False),
            gamma=torch.nn.Parameter(torch.rand(size=(bins_num,)), requires_grad=False),
            bins_width=bins_num,
            last_time=last_time,
            seed=seed
        )

        self.__lr = learning_rate
        self.__epochs_num = epochs_num
        self.__data_loader = data_loader
        self.__verbose = verbose

        self.__optimizer = torch.optim.Adam(self.parameters(), lr=self.__lr)
        self.__writer = SummaryWriter("../logs/")

        # Order matters for sequential learning
        self.__param_names = ["x0", "v", "beta","gamma"]

    def get_dataset_size(self):

        return len(self.__data_loader)

    def learn(self, learning_type="seq"):

        # Learns the parameters sequentially
        if learning_type == "seq":

            self.__sequential_learning()

        else:

            raise NotImplementedError("Non-sequential learning not implemented!")

        self.__writer.close()

    def __sequential_learning(self):

        epoch_num_per_var = int(self.__epochs_num / len(self.__param_names))

        for param_idx, param_name in enumerate(self.__param_names):

            # Set the gradients
            self.__set_gradients(**{f"{param_name}_grad": True})

            for epoch in range(param_idx*epoch_num_per_var, min(self.__epochs_num, (param_idx+1)*epoch_num_per_var)):
                self.__train_one_epoch(epoch=epoch, correction=self.__correction(centering=True, rotation=True))

    def __train_one_epoch(self, epoch, correction=None):

        self.train()

        epoch_loss = 0
        for batch_node_pairs, batch_times_list, in self.__data_loader:

            # Forward pass
            batch_loss_sum = self.forward(
                time_seq_list=batch_times_list, node_pairs=batch_node_pairs
            )

            batch_loss_value = batch_loss_sum.item()

            if not math.isfinite(batch_loss_value):
                print(f"Batch loss is {batch_loss_value}, stopping training")
                sys.exit(1)

            # Set the gradients to 0
            self.__optimizer.zero_grad()

            # Backward pass
            batch_loss_sum.backward()

            # Perform a step
            self.__optimizer.step()

            if correction is not None:
                correction()

            epoch_loss += batch_loss_value

        average_epoch_loss = epoch_loss / float(self.get_dataset_size())

        self.__writer.add_scalar(tag="Loss/train", scalar_value=average_epoch_loss, global_step=epoch)

        if self.__verbose and (epoch % 10 == 0 or epoch == self.__epochs_num):
            print(f"| Epoch = {epoch} | Loss/train: {average_epoch_loss} |")

    def forward(self, time_seq_list, node_pairs):

        return self.get_negative_log_likelihood(time_seq_list, node_pairs)

    def __set_gradients(self, beta_grad=None, x0_grad=None, v_grad=None,gamma_grad=None):

        if beta_grad is not None:
            self._beta.requires_grad = beta_grad

        if x0_grad is not None:
            self._x0.requires_grad = x0_grad

        if v_grad is not None:
            self._v.requires_grad = v_grad

        if gamma_grad is not None:
            self._gamma.requires_grad = gamma_grad

    def __correction(self, centering=None, rotation=None):

        with torch.no_grad():

            if centering:
                x0_m = torch.mean(self._x0, dim=0, keepdim=True)
                self._x0 -= x0_m

                v_m = torch.mean(self._v, dim=1, keepdim=True)
                self._v -= v_m

            if rotation:

                border = self._x0.shape[0]
                U, S, _ = torch.linalg.svd(
                    torch.vstack((self._x0, self._v.view(-1, self._v.shape[2]))),
                    full_matrices=False
                )

                temp = U @ torch.diag_embed(S)

                self._x0.data = temp[:border, :]
                self._v.data = temp[border:, :].view(self._v.shape[0], self._v.shape[1], self._v.shape[2])