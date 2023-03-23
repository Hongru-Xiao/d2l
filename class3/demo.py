import random
import torch
from d2l import torch as d2l


# def synthetic_data(w, b, num_examples):  # @save
#     """⽣成y=Xw+b+噪声"""
#     X = torch.normal(0, 1, (num_examples, len(w)))
#     y = torch.matmul(X, w) + b
#     y += torch.normal(0, 0.01, y.shape)
#     return X, y.reshape((-1, 1))
#
#
# true_w = torch.tensor([2, -3.4])
# true_b = 4.2
# features, labels = synthetic_data(true_w, true_b, 1000)
#
#
# def data_iter(batch_size, features, labels):
#     num_examples = len(features)
#     indices = list(range(num_examples))
#     # 这些样本是随机读取的，没有特定的顺序
#     random.shuffle(indices)
#     for i in range(0, num_examples, batch_size):
#         batch_indices = torch.tensor(
#             indices[i: min(i + batch_size, num_examples)])
#         yield features[batch_indices], labels[batch_indices]
#
#
# batch_size = 10
#
#
# for X, y in data_iter(batch_size, features, labels):
#     print(X, '\n', y)

y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])


def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])


def accuracy(y_hat, y): #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


print(accuracy(y_hat, y)/len(y))

