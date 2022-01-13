import torch
import math


class Constants:
    def __init__(self):

        self.eps = 1e-6
        self.inf = 1e+6
        self.pi = torch.tensor([math.pi])


const = Constants()