import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l


n_train, n_test = 100, 100
features = np.random.normal(size=((n_train + n_test), 5))
true_w = np.ones(5) * 0.01
labels = np.dot(features, true_w) + 5
labels += np.random.normal(scale=0.1, size=labels.shape)
print(labels)