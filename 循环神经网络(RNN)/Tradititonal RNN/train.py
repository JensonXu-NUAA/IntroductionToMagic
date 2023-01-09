import model
import math
import torch
import data_loader
from torch import nn
from d2l import torch as d2l
from matplotlib import pyplot as plt

'''
训练网络
'''

# 训练⽹络⼀个迭代周期
def train_epoch(net, train_iter, loss, updater, device, use_random_iter):
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 训练损失之和以及词元数量
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第⼀次迭代或使⽤随机抽样时初始化state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_()
        
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()

        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            model.grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            model.grad_clipping(net, 1)
            updater(batch_size=1)

        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

# 训练模型
def train(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity', legend=['train'], xlim=[10, num_epochs])

    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict =  lambda prefix: model.predict(prefix, 50, net, vocab, device)

    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch(net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])

    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))

# 开始训练
num_hiddens = 512
num_epochs, lr = 500, 1
batch_size, num_steps = 32, 35
train_iter, vocab = data_loader.load_data_time_machine(batch_size, num_steps)
net = model.RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), model.get_params, model.init_rnn_state, model.rnn)
train(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())
plt.show()
