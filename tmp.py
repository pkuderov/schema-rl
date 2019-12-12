import torch
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.functional import softmax
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random


def tnp():
    return [2, 3], [4, 5]
if __name__ == '__main__':
    X = tnp()
    print(*X)


