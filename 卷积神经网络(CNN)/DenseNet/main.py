import torch
from torch import nn 
from d2l import torch as d2l
from matplotlib import pyplot as plt

'''
实现一个DenseNet网络
'''

# 卷积块
def conv_block(input_channels, num_channels):
    return nn.Sequential(nn.BatchNorm2d(input_channels), nn.ReLU(),
                         nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))

# 过渡层, 用来控制模型的复杂度
# 通过1x1卷积层来减⼩通道数, 并使用步幅为2的平均汇聚层减半⾼和宽, 从而进一步降低模型复杂度
def transition_block(input_channels, num_channels):
    return nn.Sequential(nn.BatchNorm2d(input_channels), nn.ReLU(),
                         nn.Conv2d(input_channels, num_channels, kernel_size=1),
                         nn.AvgPool2d(kernel_size=2, stride=2))

# 稠密块, ⼀个稠密块由多个卷积块组成，每个卷积块使⽤相同数量的输出通道
class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []

        for i in range(num_convs):
            layer.append(conv_block(num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)
        return X

'''
# 测试
blk = DenseBlock(2, 3, 10)
X = torch.randn(4, 3, 8, 8)
Y = blk(X)
print(Y.shape)

blk = transition_block(23, 10)
print(blk(Y).shape)
'''

# 构造模型
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

num_channels, growth_rate = 64, 32  # 当前通道数, 增长率
num_convs_in_dense_blocks = [4, 4, 4, 4]  # 使用4个稠密块, 每个稠密块设置4个卷积层
blks = []

for i, num_convs in enumerate(num_convs_in_dense_blocks):
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
    # 上一个稠密块的输出通道数
    num_channels += num_convs * growth_rate
    # 在稠密块之间添加一个转换层，使得通道减半
    if i != len(num_convs_in_dense_blocks) - 1:
        blks.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2

# 最后接上全局汇聚层和全连接层来输出结果
net = nn.Sequential(b1, *blks,
                    nn.BatchNorm2d(num_channels), nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(), nn.Linear(num_channels, 10))

# 训练 
lr, num_epochs, batch_size = 0.1, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
plt.show()
