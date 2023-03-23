import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision  # 计算机视觉相关库
from torch.utils import data


def get_dataloader_workers():  # @save
    """使用4个进程来读取数据，可以不写"""
    return 4  # 并行


# 读取函数没什么好说的，就加了一个选择resize类，这个地方dataloader后是输出两个返回值，第一个为四维张量，表示为([bs，c，w，h])；
# 第二个为一维张量标签，表示为([bs]),里面的数为对应各个图片的类别标签
# 重写load_data_fashion_mnist函数，更改读取图片的地址
def load_data_fashion_mnist(batch_size, resize=None):  # @save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
        # transforms.Resize：调整PILImage对象的尺寸。transforms.Resize([h, w])或transforms.Resize(x)等比例缩放
    trans = transforms.Compose(trans)  # 串联多个图片变换的操作
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=False)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=False)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))


# 累加工具，已经讲过
class Accumulator:
    """For accumulating sums over `n` variables."""

    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 计算预测正确的个数工具，y_hat不经过softmax没影响，因为exp是单增的，这里argmax是返回一张图片中最大预测类别的位置，也就是返回对应的类别数字
# cmp是一个bool列表，包含0(False),1(True)的张量，在此为[False, True...],通过累加得到浮点数；'=='运算符对type很敏感，故还要统一成y的type
def accuracy(y_hat, y):  # @save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


# 计算指定数据集上的精度，这里data_iter后续输入为test集，经过net(X)后得到y_hat,再通过上面的accuracy得到正确预测的个数
def evaluate_accuracy(net, data_iter):  # @save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


# train_epoch_ch3函数最终返回的是训练损失值的平均数和精确度（正确预测/总），这个.numel()表示计算张量的所有元素个数
# 到最后这个返回的第一个值是计算出所有的loss后，除train图片的总个数，这才应该是整个epoch的损失值
# 如果改成float(l.mean()),然后不除metric[2],表示的是计算每一个batch的loss平均数，然后累加，不太科学，因为loss前面大后面小。
# loss经过交叉熵返回的是一个一维张量(bs),每张图片对应一个loss值。其输入的y_hat是（bs,cls),y为（bs）
def train_epoch_ch3_1(net, train_iter, loss, updater):
    """The training loop defined in Chapter 3.

    Defined in :numref:`sec_softmax_scratch`"""
    # Set the model to training mode
    if isinstance(net, torch.nn.Module):
        net.train()
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)
    for X, y in train_iter:
        # Compute gradients and update parameters
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # Using PyTorch in-built optimizer & loss criterion
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # Using custom built optimizer & loss criterion
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # Return training loss and training accuracy
    return metric[0] / metric[2], metric[1] / metric[2]


# 训练过程，返回数字,真正修改的就这一个，把画图那些省掉了
def train_ch3_1(net, train_iter, test_iter, loss, num_epochs, updater):
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3_1(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        print(f'训练损失{train_metrics[0]:.5f},精确度{train_metrics[1]:.5f},测试集{test_acc:.5f}')


if __name__ == '__main__':
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    # 初始化模型参数
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))  # 线性层之前使用展平层调整输入的形状


    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.1)


    net.apply(init_weights)

    # 损失
    loss = nn.CrossEntropyLoss(reduction='none')

    # 优化算法
    trainer = torch.optim.SGD(net.parameters(), lr=0.1)

    # 训练
    num_epoch = 10
    train_ch3_1(net, train_iter, test_iter, loss, num_epoch, trainer)

    # # 预测
    # d2l.predict_ch3(net, test_iter)
    # plt.show()
