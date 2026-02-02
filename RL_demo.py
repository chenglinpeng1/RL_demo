
#/Users/ruben/miniconda3/bin/python /Users/ruben/ML_pytorch/huashu.py

import os
import time
#!pipinstallpandas

#import pandas as pd
import torch
import numpy as np
# import numpy as np
#from matplotlib_inline import backend_inline
import matplotlib


'''
#os.makedirs(os.path.join('..', 'data'), exist_ok=True)
#data_file = os.path.join('..', 'data', 'house_tiny.csv')
home_dir = os.path.expanduser('~')#very useful function to adapt the current path to the actual pad directory
data_dir = os.path.join(home_dir, 'my_data')
os.makedirs(data_dir, exist_ok=True)

#data_file = os.path.join('..', 'data', 'house_tiny.csv')
data_file = os.path.join(data_dir, 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms, Alley, price\n') # column name
    f.write('NA, Pave, 127500\n') # every column represent a data sample
    f.write('2, NA, 106000\n') 
    f.write('4, NA, 178100\n')
    f.write('NA, NA, 140000\n')

data = pd.read_csv(data_file)
print(data)


X = torch.arange(24).reshape(2,3,4)
print(X)
#X


A = torch.arange(20, dtype=torch.float32).reshape(5,4)
B = A.clone()
print(A)
print(A+B)
print(A * B)


a = 2
X = torch.arange(24).reshape(2, 3, 4)
print(a + X)
print((a * X).shape)


x = torch.arange(4, dtype=torch.float32)
print(x)
print(x.sum())


# A = torch.arange(20, dtype=torch.float32).reshape(5,4)
# B = A.clone()
# print(A)
# print(A+B)
# print(A * B)
# print(A.shape)
# print(A.sum())

# A_sum_axis0 = A.sum(axis=0)
# print(A_sum_axis0)
# print(A_sum_axis0.shape)

# A_sum_axis1 = A.sum(axis=1)
# print(A_sum_axis1)
# print(A_sum_axis1.shape)

# print(A.sum(axis=[0,1]))

# print(A.sum(), A.mean(), A.sum()/A.numel())
# print(A.mean(axis=0), A.sum(axis=0)/A.shape[0])

# sum_A = A.sum(axis=1, keepdims=True)
# print(sum_A)
# print(A / sum_A)
# print(A.cumsum(axis=0))

# x = torch.arange(4, dtype=torch.float32)
# y = torch.ones(4, dtype=torch.float32)
# print(x, y, torch.dot(x, y))
#print(A.shape, x.shape, torch.mv(A, x)) #矩阵向量积用mv函数
# B = torch.ones(4, 3)
# print(torch.mm(A, B))

# #向量的范数
# u = torch.tensor([3.0, -4.0])
# print(torch.norm(u))#L2范数
# #L1范数
# print(torch.abs(u).sum())
# #Frobenius Norm
# m = torch.norm(torch.ones(4,9))
# print(m)

#from matplotlib_inline import backend_inline
#from d2l import torch as d2l

import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 3 * x ** 2 - 4 * x

def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h

h = 0.1
for i in range(10):
    print(f'h={h: .10f}, numerical limit={numerical_lim(f, 1, h): .10f}')
    h *= 0.1

#%matplotlib inline
#from matplotlib_inline import backend_inline
import matplotlib.pyplot as plt

def use_svg_display(): #@save
    #这个函数的作用是设置Matplotlib的输出格式为SVG(Scalable Vector Graphics可缩放矢量图形)
    #backend_inline.set_matplotlib_formats('svg')
    plt.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)): #@save
    #use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

#@save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()
#绘图函数plot
#@save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    # 如果X有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    plt.show()

#x = np.arange(0, 3, 0.1)
#plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])




# x = torch.arange(4.0)
# print(x)

# x.requires_grad_(True)
# x.grad
# y = 2 * torch.dot(x , x)
# y , x, x.grad
# y.backward()
# print(x.grad)
# print(x.grad == 4 * x)

# x.grad.zero_()
# y = x * x
# y.sum().backward()
# print(x.grad)

#分离式计算

# x.grad.zero_()
# y = x * x
# u = y.detach()
# z = u * x

# z.sum().backward()
# print(x.grad == u)

# x.grad.zero_()
# y.sum().backward()
# print(x.grad == 2 * x)

# def f(a):
#     b = a * 2
#     while b.norm() < 10000:
#         b = b * 2
#     if b.sum() > 0:
#         c = b
#     else:
#         c = 100 * b
#     return c

# a = torch.randn(size=(), requires_grad=True)
# d = f(a)
# d.backward()

# print(a.grad == d/a)
import torch
import matplotlib.pyplot as plt
#set_figsize function
def set_figsize(figsize=(3.5, 2.5)): #@save
    #use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

#概率probability
from torch.distributions import multinomial
import torch

# fair_probs = torch.ones([6])/6
# fair_probs
# cnt = multinomial.Multinomial(100000, fair_probs).sample()
# cnt/100000


# counts = multinomial.Multinomial(10, fair_probs).sample((500,))
# cum_counts = counts.cumsum(dim=0)
# estimates = cum_counts / cum_counts.sum(dim = 1, keepdims=True)

# set_figsize((6, 4.5))
# for i in range(6):
#     plt.plot(estimates[:, i].numpy(),
#             label=("P(die=" + str( i + 1) + ")"))
# plt.axhline(y=0.167, color='red', linestyle='dashed')
# plt.gca().set_xlabel('Group of experiments')
# plt.gca().set_xlabel('Estimated probability')
# plt.legend();
# plt.show();

########################
##第3章 线性神经网络######
##come on!GOGOGO!######
########################

import math
import time
import numpy as np
import torch
#from d2l import torch as d2l

n = 10000
a = torch.ones(n)
b = torch.ones(n)

class Timer: #@save
    """record n run time"""
    def __init__(self):
        self.times = []
        self.start()
    
    def start(self):
        """start the timer"""
        self.tik = time.time()

    def stop(self):
        """stop the timer and record the time in list"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """return the avg time"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """return the sum of the time"""
        return sum(self.times)

    def cumsum(self):
        """return the cumulate time"""
        return np.arange(self.times).cumsum().tolist()

# c = torch.zeros(n)
# timer = Timer()
# for i in range(n):
#     c[i] = a[i] + b[i]
# print(f'{timer.stop():.5f} sec')

# timer.start()
# d = a + b
# print(f'{timer.stop():.5f} sec')


# def normal(x, mu, sigma):
#     p = 1 / math.sqrt(2 * math.pi * sigma**2)
#     return p *np.exp(-0.5 / sigma**2*(x-mu)**2)

# x = np.arange(-7, 7, 0.01)

# params = [(0, 1), (0, 2), (3, 1)]
# plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
#     ylabel='p(x)', figsize=(4.5, 2.5),
#     legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])

import random
import torch

#generate Dateset
def synthetic_data(w, b, num_examples): #@save
    """generate y = wx + b + Norise"""
    X = torch.normal(0, 1, (num_examples, len(w)))#normal function :形状为(1000,2),里面的元素全是来自正态分布
    y = torch.matmul(X, w) + b #y's shape is (1000, 1)
    y += torch.normal(0, 0.01, y.shape)#给y加上误差，形状是y的形状(1000,1)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4]) #set w
true_b = 4.2 #set b
features, labels = synthetic_data(true_w, true_b, 1000) #features is x,labels is y

print('features:', features[0], '\nlabel:', labels[0])

set_figsize()#
#plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(),1);#点的大小设置为1
#plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(),s=5, c='blue', alpha=0.6,marker='*', label='样本点')
#plt.show();

# read the DataSet
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))#生成样本索引列表[0, 1, 2, 3, ..., 999]
    #随机读取样本，没有特定的顺序
    random.shuffle(indices)#对索引进行随机打乱
    for i in range(0, num_examples, batch_size):#遍历所有样本，每次跳过batch_size个数据。比如
        #如果batch_size=10,会循环1=0，10，20，...990
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])#取出当前小批量的样本索引：
            #indices[i: i + batch_size]是当前这批样本的位置
            #torch.tensor是把它们变成pytorch张量
            #min(i + batch_size, num_examples)为了防止越界
        yield features[batch_indices], labels[batch_indices]#使用yield返回当前批次的特征和标签，yield是为了创建一个生成器
        #它可以懒加载地逐个返回数据项，而不是一次性返回整个数据集


#read no1 small batch data
batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    #print(num_examples)
    break#只读取第一批就跳出循环

#initilize the model's paragram
w = torch.normal(0,0.01, size=(2,1), requires_grad=True)#init it use 正态分布
b = torch.zeros(1, requires_grad=True)#init it as 0

#define the model
def linreg(X, w, b): #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b

#define the loss function
def squared_loss(y_hat, y): #@save y_hat是预测值，y是真实值
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

#define optimize algorithm
def sgd(params, lr, batch_size):#params待优化的参数，比如w/b
    """小批量随机梯度下降"""
    with torch.no_grad():#告诉pytorch下面这段代码不参与自动求导机制，因为我们是手动更新参数，这样可以节省内存和计算
        for param in params:#这些参数都是torch.nn.Parameter或者有.grad属性的张量
            param -= lr * param.grad / batch_size 
# 这个是核心SGD的更新公式：这是核心的 SGD 更新公式：
# \theta = \theta - \eta \cdot \frac 1/B * nabla_\theta J
# 	•	param.grad：该参数的梯度
# 	•	/ batch_size：因为我们通常累积了整个 batch 的梯度，所以要对梯度进行标准化（平均）
# 	•	lr * grad / batch_size：本次要更新的“步长”
# 	•	param -= ...：更新参数值（向负梯度方向前进一步）
            param.grad.zero_()#将当前参数的梯度清零，以便下一轮训练时不累积之前的梯度

# train
# 理解此段代码非常重要，因为从事深度学习后，相同的训练过程几乎一遍又一遍的出现。
# 在每次迭代中，我们读取小批量训练样本，并通过我们的模型来获得一组预测
# 计算完损失后， 我们开始反向传播，存储每个参数(w1,w2,...wn)的梯度
# 最后，我们调用优化算法来更新模型的参数
# 概括一下，我们将执行以下迭代：1 初始化参数 2 重复计算梯度g <--- d/d(w,b) 1/|B| Sum(l(x(i),y(i),w,b))
# 3 更新参数(w,b) <--- (w,b)-lr*g
# 在每一轮(epoch)中，我们使用函数data_iter遍历整个训练数据集。这里的论数和学习率时超参数。

lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y) #X和y的小批量损失，net(X, w, b)线性模型的前向计算，预测值y_hat
        #因为l的形状是(batch_size, 1),而不是一个标量
        
        l.sum().backward()#l中所有的元素被加到一起，并以此计算损失关于[w,b]的梯度，并存到w.grad and b.grad
        sgd([w, b], lr, batch_size)#使用参数的梯度更行参数
    with torch.no_grad():#禁止梯度追踪 也就是清零
        #no_grad函数临时关闭自动求导机制。
        #它是pytorch的上下文管理器，常常用于推理阶段或者手动更新参数时
        #不追踪requires_grad=True的张量计算图，默认情况下，pytorch会为你创建计算图，以便之后反向传播，在某些场合
        #下，你不需要自动求导，这时就可以用此函数关闭
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

#线性回归的简洁实现
import numpy as np
import torch
from torch.utils import data

true_w = torch.tensor([2, -3.4]) #set w
true_b = 4.2 #set b
features, labels = synthetic_data(true_w, true_b, 1000)

#read dataSet
def load_array(data_arrays, batch_size, is_train=True): #@save
    """generate a data iter of pytorch"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

print(next(iter(data_iter)))

#define a model
#we use class sequential

#nn
from torch import nn

net = nn.Sequential(nn.Linear(2, 1))#linear is a class in pytorch. 因为全连接层在linear类中定义 

#init the param of the model
#我们可以用net[0]直接访问net第一层网络里的参数w和b，用normal和fill生成和置数
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
print(net[0].weight.data,net[0].bias.data)

#define the loss function
# in MSELoss class
loss = nn.MSELoss()

#define the optimize algrithm
#小批量梯度下降算法在optim模块中，当我们实例化一个SDG算法时，我们需要制定优化算法的参数
#我们可以通过net.parameters()从我们的模型中获取 、制动算法需要的超参数，这里只要指定学习率lr
trainer = torch.optim.SGD(net.parameters(), lr = 0.03)

#train
#step：1通过调用net(X)生成预测并计算损失l（前向传播）
#2通过反向传播在计算梯度
#3通过调用优化器来更新模型参数
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()#no no_grad
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l: f}')

#比较我们生成数据集的真实参数和通过有限数据训练获得的模型参数
#我们从net访问所需的层，然后在去相应的参数
w = net[0].weight.data
print('w的估计误差:', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差:', true_b - b)


#图像分类数据集
import torch
import torchvision
from torch.utils import data
from torchvision import transforms

import matplotlib.pyplot as plt

def use_svg_display(): #@save
    #这个函数的作用是设置Matplotlib的输出格式为SVG(Scalable Vector Graphics可缩放矢量图形)
    plt.rcParams['savefig.format'] = 'svg'

use_svg_display()

# read the DataSet
# read and load the Fashion MNIST to the memory using inside Function in Pytorch

# 通过ToTensor实例将图像数据从PIL类型转换成32位浮点数形式
# 并除以255使得所有像素的数值均为0-1
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="/Users/ruben/Downloads/fashionmnist", train=True, transform=trans, download=False)
mnist_test = torchvision.datasets.FashionMNIST(
    root="/Users/ruben/Downloads/fashionmnist", train=False, transform=trans, download=False)

print(len(mnist_train), len(mnist_test))
print(mnist_train[0][0].shape)

#数字标签索引和文本名称转换
def get_fashion_mnist_labels(labels):
    """返回fashionmnist数据集文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels] #使用列表推导式返回一个新列表

#创建一个函数来可视化这些样本
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5): #@save
    #接受“图像列表、行数、列数、标题列表和缩放比例”
    """绘制图像列表"""
    figsize = (num_rows * scale, num_cols * scale)#计算图表大小
    #一般写成fig,axes = plt.subplots()接受图对象fig和子图数组，这里的"__,"是一个占位符，意思是我们不关心这个值
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)#创建子图，它返回两个值，我们只关心第二个值
    axes = axes.flatten()#展平为一维数组

    for i, (ax, img) in enumerate(zip(axes, imgs)):#遍历图像和轴，将图像显示在对应的轴上，并隐藏坐标轴
        if torch.is_tensor(img):
            #图像张量
            ax.imshow(img.numpy())
        else:
            # PIL图像
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])#如果提供title，为每个图像设置标题
    return axes#返回轴对象

#show the first sample in the train dataset
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
#show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y));#将X重塑为18个28X28的图像
#plt.show()

batch_size = 256
def get_dataloader_workers(): #@save
    """use 4 processes to read data"""
    return 4
    #macOS默认对Pytorch的多进程支持不好，尤其是pytorch+多worker时，容易崩溃
    #python的multiprocessing在非主函数入口容易出错

# if __name__ == '__main__':
#     train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
#                                 num_workers=get_dataloader_workers())

#     timer = Timer()
#     for X, y in train_iter:
#         continue
#     print(f'{timer.stop(): .2f} sec')


#整合所有组件
def load_data_fashion_minist(batch_size, resize=None): #@save
    """下载Fashion-MNIST数据集,然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)#Compose把你输入的函数一个个串接起来执行，用于数据增强或者预处理
    mnist_train = torchvision.datasets.FashionMNIST(
        root="/Users/ruben/Downloads/fashionmnist", train=True, transform=trans, download=False)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="/Users/ruben/Downloads/fashionmnist", train=False, transform=trans, download=False)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))

# if __name__ == '__main__':
#     train_iter, test_iter = load_data_fashion_minist(32, resize=64)
#     for X, y in train_iter:
#         print(X.shape, X.dtype, y.shape, y.dtype)
#         break

#softmax回归的从零开始实现
import torch
from IPython import display

class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


if __name__ == '__main__': #chapter 3

    batch_size = 256
    train_iter, test_iter = load_data_fashion_minist(batch_size)
    # for X, y in train_iter:
    #     print(X.shape, X.dtype, y.shape, y.dtype)
    #     break
    #init the parameter
    num_inputs = 784
    num_outputs = 10
    
    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)
    print(w,b)

    #define softmax operations
    X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(X.sum(0, keepdim=True), X.sum(1, keepdim=True))
    #keepdim=True作用是在你求和、求均值等操作时，保留原来的维度数量，便于后续计算或广播

    def softmax(X):
        X_exp = torch.exp(X)
        partition = X_exp.sum(1, keepdim=True)
        return X_exp / partition #这里应用了广播机制

    X = torch.normal(0, 1, (2, 5))
    X_prob = softmax(X)
    print(X_prob, X_prob.sum(1))

    #define the model
    def net(X):
        return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) +b)#将X重新塑形，-1表示自动计算该维度的大小
        #W.shape[0]是权重矩阵W的第一维大小，确保X的最后一维与W的第一维匹配

    #define the loss function
    y = torch.tensor([0, 2])#y提供真实标签索引，y_hat提供预测概率。这里y表示真实标签分别为类别0和类别2
    y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
    print(y_hat[[0, 1], y])#选择第一个和第二个样本，y提供索引

    def cross_entropy(y_hat, y):
        return -torch.log(y_hat[range(len(y_hat)), y])

    print(cross_entropy(y_hat, y))

    #分类精度
    def accuracy(y_hat, y):
        """计算预测正确的数量"""
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:#如果y_hat的维数大于1且第二维（类别数）大于1
            y_hat = y_hat.argmax(axis=1)#获取每行最大元素的索引 沿轴1通常是类别的维度
        cmp = y_hat.type(y.dtype) == y#将y_hat转换为与y相同的数据类型，然后与y进行比较，生成一个布尔张量cmp，
                                    #其中ture表示预测正确，false表示预测错误
        return float(cmp.type(y.dtype).sum())#将cmp转换为y的类型，然后计算true的总数

    print(accuracy(y_hat, y) / len(y))

    def evaluate_accuracy(net, data_iter): #@save
        """计算指定数据集上模型的精度"""
        if isinstance(net, torch.nn.Module):#检查net是否是pytorch的nn.Module的实例
            net.eval() #将模型设为评估模式 如果是nn Module的实例，调用eval()，将模型切换为评估模式，
                        #禁用dropout和batch normalization等训练特定操作
        metric = Accumulator(2) #初始化两个预测器，一个用于正确的预测的样本数 一个用于总预测样本数
        with torch.no_grad():#禁用梯度计算 减少内存使用 因为评估阶段不需要反向传播
            for X, y in data_iter:
                metric.add(accuracy(net(X), y), y.numel())
        return metric[0] / metric[1]

    class Accumulator: #@save 用于累加多个计数器
        """在n个变量上累加"""
        def __init__(self, n):#接受参数n-计数器数量为参数，初始化self data为长度n的零列表【0.0】*n
            self.data = [0.0] * n

        def add(self, *args):#添加方法add，接受任意数量的参数*args，将self.data中的每一个元素与对应args相加并转换为浮点数
            self.data = [a + float(b) for a, b in zip(self.data, args)]
        
        def reset(self):
            self.data = [0.0] * len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    print(evaluate_accuracy(net, test_iter))

    #train
    def train_epoch_ch3(net, train_iter, loss, updater): #@save update是更新模型参数的常用函数
        """训练模型一轮"""
        #将模型设制为训练模式
        if isinstance(net, torch.nn.Module):
            net.train()
        
        #训练损失总和、训练准确度总和、样本数
        metric = Accumulator(3)
        for X, y in train_iter:
            #计算梯度并更新参数
            y_hat = net(X)#使用模型net对X进行前向传播得到预测值y_hat
            l = loss(y_hat, y)
            if isinstance(updater, torch.optim.Optimizer):
                #使用Pytorch内置的优化器和损失函数
                updater.zero_grad()#清零优化器中的梯度缓冲区，防止梯度累积
                l.mean().backward()#计算损失均值 然后执行反向传播，计算梯度?如果不是均值反向传播，计算每个值的反向传播是不是更准？
                #逐个样本反向传播可以把batch_size=1
                # for i in range(len(y)):
                #     l[i].backward(retain_graph=True)#每次保留计算图
                updater.step()#根据梯度更新模型参数
            else:
                #使用定制的优化器和损失函数
                l.sum().backward()
                updater(X.shape[0])#传入批次大小batch_size:X.shape[0]进行参数更新
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
        #返回训练损失和训练精度
        return metric[0] / metric[2], metric[1] / metric[2]


    #多伦训练函数
    def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater): #@save
        """train model"""
        animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.1, 1.0],
                            legend=['train loss', 'train acc', 'test acc'])#图例包括legend
        for epoch in range(num_epochs):
            train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
            test_acc = evaluate_accuracy(net, test_iter)
            animator.add(epoch + 1, train_metrics + (test_acc,))#将本轮训练/测试结果添加到指标中（loss，acc） + （test_acc,)拼成三元组
        train_loss, train_acc = train_metrics#拆开训练指标结果
        #用断言来验证模型是否成功
        assert train_loss < 0.5, train_loss #简单验证模型训练是否成功 损失小于0，5 准确率在70%-100%之间
        assert train_acc <= 1 and train_acc > 0.7, train_acc
        assert test_acc <= 1 and test_acc > 0.7, test_acc
    
    lr = 0.1
    def updater(batch_size):
        return sgd([W, b], lr, batch_size)
    num_epochs = 10
    #train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
    #plt.show()

    #predict

    def predict_ch3(net, test_iter, n=6): #@save
        """predict frag"""
        for X, y in test_iter:#只取1组，break跳出
            break
        trues = get_fashion_mnist_labels(y)#将真实标签y转换为人类可读的类名
        preds = get_fashion_mnist_labels(net(X).argmax(axis=1))#用net(X)取得预测概率，取最大概率对应的类别再转换成字符串类名
        titles = [true + '\n' + pred for true, pred in zip(trues, preds)]#创建图像标题：真实类别在上，预测类别在下，用换行符分隔
        show_images(
            X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])#显示前n张图像 每行显示1张图像 共显示n张 加上刚刚构造的的标签标题
    
    # predict_ch3(net,test_iter)
    # plt.show()

    #API 模式
    batch_size = 256
    train_iter, test_iter = load_data_fashion_minist(batch_size)
    # init model's parameter
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    #Sequential是一个容器，以顺序的方式容纳其它模块 nn.Flatten()是第一层， nn.Linear是一个全连接（密集）层
    #flatten函数：将输入的多维张量展平为1维张量，以便输入到全连接层
    #nn.Flatten()是个无参数的层，仅改变张量的形状，不会引入额外的计算开销
    #与torch.flatten()的区别：
    #nn.Flatten()是个模块，适合在nn.Module定义网络结构时使用
    #torch.flatten()是个函数，直接多张量进行操作，适合在数据处理或非模型定义的场景中使用
    #Sequential

    def init_weights(m):#m表示nn的一个模块或层
        if type(m) == nn.Linear:#m是否是全连接层的实例，权重W的初始化通常用于可学习的层如线性层
            nn.init.normal_(m.weight, std=0.01)#初始化 从正态分布中抽取值执行原地初始化
    
    net.apply(init_weights);#将init weights函数应用于net顺序模型中的所有模块（层）。apply方法递归地将函数应用于每个子模块以及self

    # loss function
    loss = nn.CrossEntropyLoss(reduction='none')#none此参数指定单独返回批次中每个样本的损失，不进行平均或者求和，如果需要每个
    #样本进一步分析或者加权平均，这可能很有用

    # optimize the algorithm
    trainer = torch.optim.SGD(net.parameters(), lr = 0.1)#net.parameters()这为优化器提供了net模型中所有需要训练期间更新的
    #可学习参数（W b）
    #学习率决定了在每次迭代中朝损失函数最小值移动的步长

    # train
    num_epochs = 10#训练周期 一个周期是指完整遍历整个训练数据集一次
    #train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
    #plt.show()


#if __name__ == '__main__': #chapter 4

    # class Accumulator: #@save 用于累加多个计数器
    #     """在n个变量上累加"""
    #     def __init__(self, n):#接受参数n-计数器数量为参数，初始化self data为长度n的零列表【0.0】*n
    #         self.data = [0.0] * n

    #     def add(self, *args):#添加方法add，接受任意数量的参数*args，将self.data中的每一个元素与对应args相加并转换为浮点数
    #         self.data = [a + float(b) for a, b in zip(self.data, args)]
        
    #     def reset(self):
    #         self.data = [0.0] * len(self.data)

    #     def __getitem__(self, idx):
    #         return self.data[idx]
    #train
    # def train_epoch_ch3(net, train_iter, loss, updater): #@save update是更新模型参数的常用函数
    #     """训练模型一轮"""
    #     #将模型设制为训练模式
    #     if isinstance(net, torch.nn.Module):
    #         net.train()
        
    #     #训练损失总和、训练准确度总和、样本数
    #     metric = Accumulator(3)
    #     for X, y in train_iter:
    #         #计算梯度并更新参数
    #         y_hat = net(X)#使用模型net对X进行前向传播得到预测值y_hat
    #         l = loss(y_hat, y)
    #         if isinstance(updater, torch.optim.Optimizer):
    #             #使用Pytorch内置的优化器和损失函数
    #             updater.zero_grad()#清零优化器中的梯度缓冲区，防止梯度累积
    #             l.mean().backward()#计算损失均值 然后执行反向传播，计算梯度?如果不是均值反向传播，计算每个值的反向传播是不是更准？
    #             #逐个样本反向传播可以把batch_size=1
    #             # for i in range(len(y)):
    #             #     l[i].backward(retain_graph=True)#每次保留计算图
    #             updater.step()#根据梯度更新模型参数
    #         else:
    #         #使用定制的优化器和损失函数
    #             l.sum().backward()
    #             updater(X.shape[0])#传入批次大小batch_size:X.shape[0]进行参数更新
    #         metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    #     #返回训练损失和训练精度
    #     return metric[0] / metric[2], metric[1] / metric[2]

    #多伦训练函数
    # def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater): #@save
    #     """train model"""
    #     animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.1, 1.0],
    #                         legend=['train loss', 'train acc', 'test acc'])#图例包括legend
    #     for epoch in range(num_epochs):
    #         train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
    #         test_acc = evaluate_accuracy(net, test_iter)
    #         animator.add(epoch + 1, train_metrics + (test_acc,))#将本轮训练/测试结果添加到指标中（loss，acc） + （test_acc,)拼成三元组
    #     train_loss, train_acc = train_metrics#拆开训练指标结果
    #     #用断言来验证模型是否成功
    #     assert train_loss < 0.5, train_loss #简单验证模型训练是否成功 损失小于0，5 准确率在70%-100%之间
    #     assert train_acc <= 1 and train_acc > 0.7, train_acc
    #     assert test_acc <= 1 and test_acc > 0.7, test_acc

    x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)#建一个从-8到8，步长为0.1，相当于一个等差数列。requires_grad=True需要计算这个张量X的梯度

    y = torch.relu(x)#应用ReLU修正线性单元到张量X上
    #detach函数是返回一个新的张量，与原张量xy具有相同的数据，但不再与计算图关联。这意味着后续的梯度计算中，xy不会被考虑在内。在这里使用
    #detach通常是为了绘图。
    #plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
    #plt.show()

    #绘制ReLU函数的导数图像
    #这是pytorch的关键一步，用于计算梯度。当对一个标量计算梯度时，直接调用backward函数。
    #然而y这里是一个张量，在对张量计算梯度时，我们需要传入一个与y形状相同的张量作为梯度，通常称为“梯度向量”或“权重”（weights）
    #ones_like会创建一个与x形状相同，且所有元素都为1的张量，这意味着我们正在计算y对x的导数即梯度，并且每个输出维度都对相应的输入维度有等同贡献
    #retain_graph=True 这个参数告诉pytorch在这次反向传播之后保留计算图。一般计算完我们会释放，以节省内存。但是，如果我们需要多次反向传播
    #例如在同一个计算图上计算不同部分的梯度）就需要这样设置。接下来我们还需要绘制梯度的图像，所以需要保留。
    y.backward(torch.ones_like(x), retain_graph=True)
    #plot(x.detach(), x.grad,  'x', 'grad of relu', figsize=(5, 2.5))#x.grad这个是关键，在调用y.backward之后，x张量的grad
    #属性会存储y对x的梯度

    #sigmoid激活函数
    y = torch.sigmoid(x)
    #plot(x.detach(), y.detach(),  'x', 'sigmoid(x)', figsize=(5, 2.5))

    #sigmoid函数的导数
    x.grad.data.zero_()
    y.backward(torch.ones_like(x), retain_graph=True)
    #plot(x.detach(), x.grad,  'x', 'grad of sigmoid', figsize=(5, 2.5))

    #tanh函数
    #在0处是关于0的中心对称
    y = torch.tanh(x)
    #plot(x.detach(), y.detach(),  'x', 'tanh(x)', figsize=(5, 2.5))

    #tanh函数的导数
    x.grad.data.zero_()
    y.backward(torch.ones_like(x), retain_graph=True)
    #plot(x.detach(), x.grad,  'x', 'grad of tanh', figsize=(5, 2.5))  


    #MLP的从零开始实现 Chatpter 4
    import torch
    from torch import nn
    batch_size = 256
    train_iter, test_iter = load_data_fashion_minist(batch_size)

    #init parameter of model
    # num_hiddens 隐藏层 神经网络数
    num_inputs, num_outputs, num_hiddens = 784, 10, 256
    # 初始化第一个全连接层（从输入到隐藏层）的权重矩阵 W1。
    # torch.randn(num_inputs, num_hiddens) 创建一个形状为 (784, 256) 的张量，并用标准正态分布随机初始化其元素。
    # * 0.01 将随机初始化的值乘以 0.01，这是一种常见的权重初始化策略，有助于防止训练初期梯度爆炸或消失。
    # * nn.Parameter(...) 将这个张量包装成 PyTorch 的可学习参数。这意味着在反向传播时，PyTorch 会自动计算并更新 W1 的梯度。
    # * requires_grad=True 明确指示 PyTorch 为此参数计算梯度（默认对于 nn.Parameter 就是 True）。
    W1 = nn.Parameter(torch.randn(
        num_inputs, num_hiddens, requires_grad=True) * 0.01)
    # 初始化第一个隐藏层的偏置向量 b1。
    # torch.zeros(num_hiddens) 创建一个形状为 (256) 的零张量。
    # 同样，nn.Parameter(...) 将其标记为可学习参数。
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
    # 初始化第二个全连接层（从隐藏层到输出层）的权重矩阵 W2。形状为 (256, 10)。
    W2 = nn.Parameter(torch.randn(
        num_hiddens, num_outputs, requires_grad=True) * 0.01)
    # 初始化输出层的偏置向量 b2。形状为 (10)
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
    # 将所有可学习参数放入一个列表中。这通常是为了方便后续将这些参数传递给优化器。
    params = [W1, b1, W2, b2]

    #激活函数
    # 定义了一个名为 relu 的函数，它接受一个输入张量 X。ReLU（Rectified Linear Unit）是一种常用的激活函数
    def relu(X):
        # 创建一个与输入张量 X 形状相同、数据类型相同且所有元素都为零的张量 a
        a = torch.zeros_like(X)
        return torch.max(X, a)#返回 X 和 a 之间按元素取最大值的结果。对于 ReLU 函数，这意味着所有小于零的值都将被替换为零，而正值保持不变
    
    #net
    def net(X):#定义了神经网络的前向传播逻辑。它接受一个输入张量 X
        # -1 表示 PyTorch 会自动计算该维度的大小，以确保总元素数量不变？
        # num_inputs (784) 表示每个样本的特征数量。
        # 例如，如果 X 的原始形状是 (batch_size, 1, 28, 28)，这一行会将其重塑为 (batch_size, 784)，将每个 28x28 的图像展平为一维向量
        X = X.reshape(-1, num_inputs)#将输入张量 X 重塑
        # 计算隐藏层的输出 H。
        # * X@W1 执行矩阵乘法（@ 运算符在 Python 3.5+ 中代表矩阵乘法）。将输入 X 与第一个权重矩阵 W1 相乘。
        # * + b1 将偏置向量 b1 加到乘积结果上（PyTorch 会自动进行广播）。
        # * relu(...) 将 ReLU 激活函数应用于结果。
        H = relu(X@W1 + b1) #@ represent matrix muliply
        # 计算输出层的输出。
        # * H@W2 执行矩阵乘法，将隐藏层输出 H 与第二个权重矩阵 W2 相乘。
        # * + b2 将偏置向量 b2 加到乘积结果上。
        # * 这是模型的最终预测（通常称为 logits），在分类任务中，这些值会传递给损失函数
        return (H@W2 + b2)
    
    #loss
    # 定义了交叉熵损失函数。
    # * nn.CrossEntropyLoss 适用于多类别分类问题，它结合了 Softmax 和负对数似然损失。
    # * reduction='none' 表示不对损失进行平均或求和，而是返回每个样本的损失值。这在某些情况下很有用，例如需要对每个样本的损失进行加权时。
    # * 注意： 这一行单独存在，它创建了一个损失函数的实例，但没有将其赋值给任何变量，也没有直接使用它。在实际的训练循环中，您需要将它赋值给一个变量，
    # 比如 loss_fn = nn.CrossEntropyLoss()，然后在计算损失时调用 loss_fn(predictions, targets)。根据您提供的第二张图片中的错误信息，
    # 很可能在 train_ch3 函数内部正确地定义并使用了 loss 变量。
    # nn.CrossEntropyLoss(reduction='none')
    #train
    # * num_epochs = 10: 训练的总轮数。模型将完整地遍历训练数据集 10 次。
    # * lr = 0.1: 学习率（learning rate）。这是一个控制模型在每次迭代中更新其参数步长的超参数
    num_epochs, lr = 10, 0.1#定义了训练过程的超参数
    # 初始化了一个随机梯度下降（SGD）优化器。
    # * torch.optim.SGD 是 PyTorch 中 SGD 优化器的类。
    # * params (之前定义的 [W1, b1, W2, b2]) 告诉优化器需要更新哪些模型参数。
    # * lr=lr 设置了优化器的学习率。
    updater = torch.optim.SGD(params, lr=lr)
    # 这很可能是实际的训练循环函数，它封装了训练和评估模型的过程。
    # * net: 您的神经网络模型（这里是 net 函数）。
    # * train_iter: 训练数据迭代器
    # * test_iter: 测试数据迭代器
    # * loss: 损失函数（在 PyTorch 中，通常这里会传递一个 nn.CrossEntropyLoss 的实例，而不是直接写 loss，
    #   但根据上下文，可能在 train_ch3 函数内部定义了 loss 变量）。
    # * num_epochs: 训练轮数
    # * updater: 优化器
    #train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
    #plt.show()

    #predict
    #predict_ch3(net, test_iter)  
    #plt.show()

    #MLP API 
    #这是PyTorch中一个常用的容器，它会将传入的模块按顺序连接起来，形成一个神经网络。
    # * nn.Flatten(): 展平层。当输入是多维数据（例如图像，通常是 [批大小, 频道, 高, 宽]）时，这个层会将其展平为一维向量。对于Fashion MNIST图像（28x28像素），784 是 28 * 28 的结果，表示将每张图像展平为784个特征。
    # * nn.Linear(784,256): 这是一个全连接层（也称为密集层或线性层）。它将输入（784个特征）映射到256个输出特征。这是网络的第一隐藏层。
    # * nn.ReLU(): 这是一个ReLU（Rectified Linear Unit）激活函数。它对线性层的输出进行非线性变换，引入非线性能力，使得网络能够学习更复杂的模式。
    # * nn.Linear(256,10): 这是网络的输出层。它将前一个隐藏层的256个特征映射到10个输出。这10个输出通常对应于分类问题的10个类别（例如，Fashion MNIST数据集有10种衣物类别）。
    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(784,256),
                        nn.ReLU(),
                        nn.Linear(256,10))
    
    # * 这是一个函数定义，用于初始化神经网络的权重
    def init_weights(m):
        #这行检查传入的模块 m 是否是 nn.Linear（全连接层）。我们通常只对全连接层或卷积层的权重进行初始化
        if type(m) == nn.Linear:
            #如果 m 是一个线性层，这行代码会使用正态分布来初始化其权重 m.weight。std=0.01 表示正态分布的标准差为0.01，
            #即权重会被初始化为接近0的小随机数。这种初始化有助于训练的稳定性
            nn.init.normal_(m.weight, std=0.01)
    # 这是PyTorch nn.Module 的一个方法，它会将传入的函数递归地应用到模块的所有子模块上。
    # 这里，它会遍历 net 中的所有层（nn.Flatten、nn.Linear、nn.ReLU 等），并将 init_weights 函数应用于每个层。这样，所有 nn.Linear 层的权重都会被初始化
    net.apply(init_weights)

    # 这是定义训练超参数
    # * batch_size = 256: 每次训练迭代（或梯度更新）时处理的样本数量
    # * lr = 0.1: 学习率（learning rate），控制模型在每次迭代中更新权重的步长
    # * num_epochs = 10: 训练的总轮数，即整个数据集将被模型遍历和学习10次
    batch_size, lr, num_epochs = 256, 0.1, 10
    # 这是PyTorch中常用的用于多分类问题的损失函数。它结合了Softmax（将模型的原始输出转换为概率分布）和负对数似然损失
    # * reduction='none': 这意味着损失函数不会对每个样本的损失进行求平均或求和，而是返回一个与输入批次大小相同的损失向量，
    # 每个元素对应一个样本的损失。在后续计算梯度时，通常会手动对这些损失求平均
    loss = nn.CrossEntropyLoss(reduction='none')
    # 这是随机梯度下降（SGD）优化器。优化器的作用是根据计算出的损失梯度来更新模型的权重。
    # * net.parameters(): 这会返回神经网络 net 中所有可学习的参数（即权重和偏置）。优化器将使用这些参数进行更新。
    # * lr=lr: 将前面定义的学习率传递给优化器
    updater = torch.optim.SGD(net.parameters(), lr=lr)
    # * 这行代码调用了一个名为 load_data_fashion_mnist 的函数，用于加载Fashion MNIST数据集。
    # * 它会将数据集分成训练集和测试集，并使用 batch_size 参数来创建数据迭代器 (train_iter 和 test_iter)，
    # 以便在训练和测试时可以按批次获取数据
    train_iter, test_iter = load_data_fashion_minist(batch_size)
    # * 这很可能是一个自定义的训练函数，负责执行神经网络的训练循环。
    # * 它接收的参数包括：
    # * net: 待训练的神经网络模型。
    # * train_iter: 训练数据迭代器。
    # * test_iter: 测试数据迭代器（用于在训练过程中评估模型性能）。
    # * loss: 损失函数。
    # * num_epochs: 训练的总轮数。
    # * updater: 优化器。
    # * 这个函数内部会包含每个 epoch 的循环，以及每个 batch 的前向传播、损失计算、反向传播和参数更新
    #train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
    #plt.show()

    # overfitting and regularization
    # 多项式回归举例
    import torch
    import numpy as np
    import math
    from torch import nn
    # generate 100 train sample and test sample
    max_degree = 20 # 多项式最大的阶数
    n_train, n_test = 100, 100
    true_w = np.zeros(max_degree)# 分配大量的空间 numpy数组存储多项式系数
    true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

    features = np.random.normal(size=(n_train + n_test, 1)) #生成200个x的值(形状（200，1）
    np.random.shuffle(features)
    #np.arange(max_degree)生成一个从0-max_degree-1的数组，即[0,1,2,3,....19]。reshape将其重塑为行向量[[0,1,2,...19]]
    poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))#pow函数时逐元素的x的y次幂,对featrues中的每个元素分别计算0次幂，1次幂。。。
    #构建多项式特征矩阵
    #对于每个样本x， 生成X^0,X^1,X^2....X^19,最终的形状是（200，20）每行是一个样本的多项式展开

    #所有的多项式除以对应阶数的阶乘
    #实际上这是做泰勒展开标准化（用于数值稳定性，尤其是在正则化时）
    for i in range(max_degree):
        poly_features[:, i] /= math.gamma(i + 1) #gamma(n) = (n-1)!
    #labels的维度：(n_train + n_test)
    # 用真实的W和多项式特征计算出“目标值”
    # 添加随机高斯噪声，模拟真实数据的不确定性
    labels = np.dot(poly_features, true_w)#点积矩阵乘法
    labels += np.random.normal(scale=0.1, size=labels.shape)#生成与label形状相同一组服从正态分布的随机数，其标准差是0.1

    #查看前两个样本
    # NumPy ndarray 转换为tensor
    true_w, features, poly_features, labels = [torch.tensor(x, dtype=
        torch.float32) for x in [true_w, features, poly_features, labels]]#列表推导式样
    #打印前两个样本的原始输入，对应的多项式特征、标签（y值）
    print(features[:2], poly_features[:2, :], labels[:2])

    # 评估函数
    def evaluate_loss(net, data_iter, loss): #@save
        """评估给定数据集上的损失"""
        metric = Accumulator(2) #损失的总和 样本的数量
        for X, y in data_iter:
            out = net(X)
            y = y.reshape(out.shape)#在处理不同数据的维度时常见
            l = loss(out, y)
            metric.add(l.sum(),l.numel())
        return metric[0] / metric[1]#函数返回平均损失
    
    # train function
    def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
        loss = nn.MSELoss(reduction='none')#none表示不对每个样本的损失进行求和或求平均，而是返回一个与输入形状相同的损失张量（每个样本一个损失值）
        input_shape = train_features.shape[-1]#获取训练特征的最后一个维度的大小 通常代表输入特征的数量
        #不设置偏置，因为已经在多项式中实现了它 暗示数据通过某种多项式变换包括了bias项
        net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))#input_shape输入特征的数量，1输出特征的数量（回归任务输出1个值
        batch_size = min(10, train_labels.shape[0])#确保批处理大小不会超过总样本数 train_labels.shape[0]代表训练标签数量
        #load_array用于将数据封装成可迭代的批次
        #train_labels.reshape(-1,1)将训练标签转换成列向量 以适应模型输出形状
        #-1是个占位符，表示pytorch或者numpy会根据张量的总元素数量和另一个指定的维度自动计算当前维度的长度，简单说就是你告诉pytorch
        #把这个维度的大小留给我自己计算，只要确保总元素数量不变，并且另一个维度是指定的长度就行
        #1的含义表示你希望张量的第二维度（列）的长度是1
        #将一个张量重新塑形为一个拥有任意行数但只有一列的二维张量
        train_iter = load_array((train_features, train_labels.reshape(-1,1)), batch_size)
        test_iter = load_array((test_features, test_labels.reshape(-1,1)), batch_size, is_train=False)
        updater = torch.optim.SGD(net.parameters(), lr=0.01)
        animator = Animator(xlabel='epoch', ylabel='loss', yscale='log', 
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])#yscale='log' y轴使用对数刻度
        
        for epoch in range(num_epochs):
            train_epoch_ch3(net, train_iter, loss, updater)
            if epoch == 0 or (epoch + 1) % 20 == 0:#每隔20个周期执行一些操作
                animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                         evaluate_loss(net, test_iter, loss)))
        print('Weiget:', net[0].weight.data.numpy())#weight.data.numpy()将张量转换为数组便于打印

    #三阶多项式拟合正常
    #从多项式中选择前4个维度，即1、x、x^2/2!, x^3/3!
    #poly_features[:n_train, :4] 从训练集中选择前n_train个样本和前4个特征
    # train(poly_features[:n_train, :4], poly_features[n_train:, :4],
    #         labels[:n_train], labels[n_train:])
    #weight: [[4.997884 1.2145151 -3.4133592 5.5992494]]
    #plt.show()

    #线性函数拟合（欠拟合）
    #从多项式特征中选择前两个维度即 1和x
    # train(poly_features[:n_train, :2], poly_features[n_train:, :2],
    #         labels[:n_train], labels[n_train:])
    #plt.show() 

    #高阶多项式函数拟合（过拟合）
    #从多项式特征中选取所有维度
    #num_epochs=1500长时间的训练容易导致过拟合
    # train(poly_features[:n_train, :], poly_features[n_train:, :],
    #         labels[:n_train], labels[n_train:], num_epochs=1500)
    #plt.show()

    #权重衰减
    n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
    #true_w是一个所有元素都是0.01的num_inputs * 1的张量
    true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
    #用true_w, true_b, n_train生成训练数据
    train_data = synthetic_data(true_w, true_b, n_train)
    train_iter = load_array(train_data, batch_size)
    test_data = synthetic_data(true_w, true_b, n_test)#将训练数据包装成一个数据迭代器
    test_iter = load_array(test_data, batch_size, is_train=False)

    #init the parameters of the model
    def init_params():
        w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        return [w, b]
    #define L2 范数
    def l2_penalty(w):
        return torch.sum(w.pow(2)) / 2   
    #define train function
    def train(lambd):
        w, b = init_params()
        #这是个匿名函数 它接受输入X，并调用linreg函数进行线性回归计算，使用当前的w和b 
        net, loss = lambda X: linreg(X, w, b), squared_loss#squared_loss表面模型正在进行回归任务，且使用的是MSE
        num_epochs, lr = 100, 0.003
        animator = Animator(xlabel='epoch', ylabel='loss', yscale='log', 
                            xlim=[5, num_epochs], 
                            legend=['train', 'test'])
        for epoch in range(num_epochs):
            for X, y in train_iter:
                #增加L2范数惩罚项
                #广播机制使用l2_penalty(w)成为一个长度为batech_size的向量
                l = loss(net(X), y) + lambd * l2_penalty(w)
                l.sum().backward()
                sgd([w, b], lr, batch_size)
            if (epoch + 1) % 5 == 0:#每隔5个epoch执行一次代码
                animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                         evaluate_loss(net, test_iter, loss)))
        print('w的L2范数是: ', torch.norm(w).item())

    #ignore the L2 and train the model derectly
    # train(lambd=0)
    # plt.show()

    # #使用权重衰减
    # train(lambd=3)
    # plt.show()

    # API concise implement code
    def train_concise(wd):
        net = nn.Sequential(nn.Linear(num_inputs, 1))
        #遍历模型中所有可学习的参数进行原地正态分布初始化
        for param in net.parameters():
            param.data.normal_()
        loss = nn.MSELoss(reduction='none')
        num_epochs, lr = 100, 0.003
        #net0指第一层网路即Linear wd为权重设置了权重衰减 L2正则化 pytorch的优化器会在反向传播时将权重衰减项添加到梯度中
        trainer = torch.optim.SGD([
            {"params":net[0].weight, 'weight_decay':wd},
            {"params":net[0].bias}], lr=lr)
        animator = Animator(xlabel='epoch', ylabel='loss', yscale='log', 
                            xlim=[5, num_epochs], 
                            legend=['train', 'test'])
        
        for epoch in range(num_epochs):#表示迭代次数
            for X, y in train_iter:#遍历训练集中的每个小批量
                trainer.zero_grad()
                l = loss(net(X), y)#由于none，l是一个与batchsize相同的向量，包含每个样本的损失
                l.mean().backward()#mean优化器通常是基于平均梯度进行更新 backward计算这个平均损失相对于模型参数的梯度，这个时候L2正则化的梯度会包含在内
                trainer.step()#计算出梯度更新的模型参数  
            if (epoch + 1) % 5 == 0 :
                animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                         evaluate_loss(net, test_iter, loss)))
        print('w的L2范数是: ', torch.norm(w).item())
    
    # train_concise(0)
    # plt.show()
    # train_concise(3)
    # plt.show()


    # Dropout
    def dropout_layer(X, dropout):
        assert 0 <= dropout <= 1 #确保dropout在0-1之间，不然程序会报错
        #在此情况下所有元素都被丢弃
        if dropout == 1:
            return torch.zeros_like(X)
        #在此情况中，所有元素都被保留
        if dropout == 0:
            return X
        #生成一个与X形状相同的随机张量，其元素在0-1之间均匀分布
        #然后将这个随机张量与dropout比率进行比较
        #True为1 False为0
        #形成一个淹码张量
        mask = (torch.rand(X.shape) > dropout).float()
        #将输入X与Mask逐元素相乘 然后/（1-dropout）进行缩放
        #这种缩放是为了在训练阶段保持激活值的期望不变，因为一部分神经元已经被置0！
        return mask * X / (1.0 -dropout)   
    X = torch.arange(16, dtype = torch.float32).reshape((2, 8))#创建一个0-15的张量，然后塑形为（2，8）的二维张量。这个是为了测试dropout_layer函数
    # print(X)
    # print(dropout_layer(X, 0))
    # print(dropout_layer(X, 0.5))
    # print(dropout_layer(X, 1.))
    # 1定义模型参数
    # num_hiddens1第一个隐藏层神经元的数量 num_hiddens2第二个隐藏层神经元的数量
    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
    # 2定义模型
    dropout1, dropout2 = 0.2, 0.5

    class Net(nn.Module):#继承自pytorch的nn Module 是所有神经网络的基类
        def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, is_training = True):
            super(Net, self).__init__()#调用父类的初始化函数进行初始化
            self.num_inputs = num_inputs
            self.training = is_training
            self.lin1 = nn.Linear(num_inputs, num_hiddens1)#定义第一个全连接层 输入维度num_inputs 输出维度num_hiddens1
            self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
            self.lin3 = nn.Linear(num_hiddens2, num_outputs)
            self.relu = nn.ReLU()
        #定义网络的前向传播过程
        def forward(self, X):
            #将X reshape -1自动计算大小 将X传给lin1,然后应用激活函数 结果存储在H1
            H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
            #只有在训练模型时才使用暂退法
            if self.training == True:
                #在第一个全连接层后添加一个dropout层
                H1 = dropout_layer(H1, dropout1)
            #将H1传递给第二个全连接层，然后用relu， 结果存储在H2
            H2 = self.relu(self.lin2(H1))

            if self.training == True:
                #在第2个全连接层后添加一个dropout层
                H2 = dropout_layer(H2, dropout2)
            #将H2传递给第三个全连接层
            out = self.lin3(H2)
            return out

    net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)

    #train and test
    num_epochs, lr, batch_size = 10, 0.5, 256
    loss = nn.CrossEntropyLoss(reduction='none')
    train_iter, test_iter = load_data_fashion_minist(batch_size)
    updater = torch.optim.SGD(net.parameters(), lr=lr)
    #train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
    #plt.show()

    # API implement

    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(784, 256),
                        nn.ReLU(),
                        #在第一个全连接层后添加一个dropout层
                        nn.Dropout(dropout1),
                        nn.Linear(256, 256),
                        #在第二个全连接层后添加一个dropout层
                        nn.Dropout(dropout2),
                        nn.Linear(256, 10))
    
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)
    
    net.apply(init_weights)

    #train and test
    updater = torch.optim.SGD(net.parameters(), lr=lr)
    #train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
    #plt.show()


    # 实战Kaggle比赛：预测房价
    import hashlib #用于计算文件的hash值(例如sha-1)以验证文件的完整性
    import os
    import tarfile #用于处理tar/tar.gz压缩文件
    import zipfile #用于处理zip压缩文件
    import requests #用于发送http请求从网上下载文件

    #@save
    DATA_HUB = dict()#初始化一个空字典，可能用于存储数据集的元信息，比如文件名和对应的哈希值
    DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'#定义一个基础url 所有要下载的数据都将从这个url下载

    #定义一个函数用于下载指定文件
    #name：要下载的文件在DATA_HUB中的键名
    #cache_dir缓存目录，默认为当前目录的上一级目录下的data文件夹
    def download(name, cache_dir='/Users/ruben/Downloads/Kaggle_house'):#os.path.join('..', 'data')): #@save
        """下载一个DATA_HUB中的文件,返回本地文件名"""
        assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"#检查name是否存在于DATA_HUB字典中 如果不存在则抛出错误
        url, sha1_hash = DATA_HUB[name]#从DATA_HUB字典中根据name获取url和sha1_hash值
        #os.mkdirs(cache_dir, exist_ok=True)#创建缓存目录，如果目录已存在则抛出错误
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        #创建本地文件名，将缓存目录和url最后一部分通常是文件名连接起来 url.splite('/')[-1]提取url中的文件名
        fname = os.path.join(cache_dir, url.split('/')[-1])
        if os.path.exists(fname):#检查本地是否已存在同文件名
            #如果文件存在则初始化一个sha1哈希对象
            sha1 = hashlib.sha1()
            with open(fname, 'rb') as f:#以二进制读取模式打开本地文件
                while True:
                    data = f.read(1048576)#循环读取文件内容 每次读取1MB-1048576的数据
                    if not data:#如果读取的数据为空，说明文件已读完 跳出循环
                        break
                    sha1.update(data)#更新sha1哈希对象，将读取的数据添加到哈希计算中
            if sha1.hexdigest() == sha1_hash:#计算文件的最终哈希值并与预期的哈希值进行比较 hexdigest16进制
                return fname #hit the cache memory
        print(f'正在从{url}下载{fname}...')#如果文件不存在或者哈希值不匹配 则打印下载信息
        #使用requests库发起GET请求下载文件
        #url stream=true 允许以流的方式下载 而不是一次性下载所有内容到内存 这对大文件很重要
        #verift=true验证SSL证书 通常默认true 但明确写出更清晰
        r = requests.get(url, stream=True, verify=True)
        #以二进制写入模式打开本地文件 准备写入下载内容
        with open(fname, 'wb') as f:
            f.write(r.content)#将下载的内容写入本地文件 r.content是响应的原始二进制内容
        return fname
    # 函数用于下载并解压文件
    def download_extract(name, folder=None): #@save folder=none解压后子文件名称 可选参数
        """下载并解压zip/tar文件"""
        fname = download(name)
        base_dir = os.path.dirname(fname)#获取下载文件所在的目录路径
        data_dir, ext = os.path.splitext(fname)#分离文件名和扩展名
        if ext == '.zip':
            fp = zipfile.Zipfile(fname, 'r')#使用Zipfile打开zip文件对象 r表示读取模式
        elif ext in ('.tar', '.gz'):
            fp = tarfile.open(fname, 'r')
        else:
            assert False, '只有zip/tar文件可以被解压缩'
        fp.extractall(base_dir)#将压缩文件中的所有内容解压到base_dir目录下
        #返回解压后的数据目录路径：如果folder参数有值，则返回base_dir+folder的路径 否则返回data_dir(即不带扩展名的原始文件名路径)
        return os.path.join(base_dir, folder) if folder else data_dir

    def download_all(): #@save
        """下载DATA_HUB中所有的文件"""
        for name in DATA_HUB:
            download(name)

    import numpy as np
    import pandas as pd#用于数据分析或者数据操作
    import torch
    #form torch import nn
    #DATA_HUB字典用于保存数据的下载链接和对应文件的哈希值
    DATA_HUB['kaggle_house_train'] = ( #@save
        DATA_URL + 'kaggle_house_pred_train.csv',
        '585e9cc93e70b39160e7921475f9bcd7d31219ce')
    DATA_HUB['kaggle_house_test'] = ( #@save
        DATA_URL + 'kaggle_house_pred_test.csv',
        'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')
    
    train_data = pd.read_csv(download('kaggle_house_train'))
    test_data = pd.read_csv(download('kaggle_house_test'))

    # print(train_data.shape)
    # print(test_data.shape)
    #iloc函数是padas里按行和列获取数据的函数
    print(train_data.iloc[0:4], [0, 1, 2, 3, -3, -2, -1])
    #concat是拼接，目的是将训练数据和测试数据特征部分拼接起来 形成一个包含所有特征完整数据集 用于后面的统一处理如归一化、独立编码等
    #iloc[:,1:-1]是切片动作，逗号前：表示选择所有的行 1:表示从素引为1的列开始到最后一列
    all_features = pd.concat((train_data.iloc[:,1:-1], test_data.iloc[:, 1:]))#训练数据含有标签而测试数据不含标签

    #preprocess the dataset
    #筛选出all_features中数据类型不是object（通常表示数值型）的列的名称（索引） 并将这些列名存储在numeric_features变量中
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    #对所有数值特征进行标准化 每一列减去该列的均值并除以该列的标准差 使得所有数值特征的均值为0 标准差为1 有助于模型训练
    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda x: (x -x.mean()) / (x.std()))
    #用0填充数值型特征中的所有缺失值
    all_features[numeric_features] = all_features[numeric_features].fillna(0)

    #ummy_na=True将na（缺失值）视为有效的特征值，并为其创建指示符特征 即会将缺失值视为一个独立的类别 并为它创建一个新的指示器（哑变量）列
    all_features = pd.get_dummies(all_features, dummy_na=True)
    #打印all_features中数据类型仍然是object的列 在经过独立编码后 理论上不应该再有object类型的列 这一步可能用于检查
    print(f'hahah:', all_features.dtypes[all_features.dtypes == 'object'])
    print(all_features.shape)
    #获取原始训练数据的行数（样本数量）并存在n_train中，这个值用于后续将处理后的all_features重新分割为训练集和测试集
    n_train = train_data.shape[0]
    #将all_features前n_train行即训练集特征部分转换为pytorch的张量 .value将DateFrame转换为Numpy数组
    train_features = torch.tensor(all_features[:n_train].values.astype(np.float32), dtype=torch.float32)
    test_features = torch.tensor(all_features[n_train:].values.astype(np.float32), dtype=torch.float32)
    #将原始训练数据的SalePrice列通常是标签转换为张量 reshape(-1, 1)将其形状调整为列向量即每行一个标签
    train_labels = torch.tensor(
        train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)
    
    #train
    #首先训练一个线性模型作为基线模型
    loss = nn.MSELoss()
    in_features = train_features.shape[1]#获取训练数据的特征 列数 并将其存储在in_features变量中 这通常作为神经网络输入层的神经元数量

    def get_net():
        #Sequential是序贯容器
        net = nn.Sequential(nn.Linear(in_features,1))
        return net
    #定义log_rmse函数，用于计算对数均方根误差 这个指标常用于回归问题 特别是当标签数据呈现指数分布时 可以减少大小误差的影响
    def log_rmse(net, features, labels):
        #为了在取对数时进一步稳定该值，将小🐟1的值设为1
        #torch.clamp(net(features), 1, float('inf'))将预测值限制在一个范围内 任何小于1的值都会被裁剪为1 任何大于正无穷的值保持不变
        #clamp函数是pytorch库中的函数 它的含义是将输入张量中的所有元素裁剪（限制）到指定的数值范围之内
        clipped_preds = torch.clamp(net(features), 1, float('inf'))#clamp函数 inf=infinity无穷
        #计算对数均方根误差
        rmse = torch.sqrt(loss(torch.log(clipped_preds),
                               torch.log(labels)))
        return rmse.item()#item将pytorch张量中单个值提取为pytorch标量

    #用对初始学习率不敏感的Adam优化器 Adam是一种自适应学习率优化算法
    def train(net, train_features, train_labels, test_features, test_labels,
              num_epochss, learning_rate, weight_decay, batch_size):#weight_decay权重衰减 用于防止过拟合
        train_ls, test_ls = [], []#初始化两个空列表 用于存储每个epoch训练集和测试集上的损失
        #创建一个训练数据迭代器 将训练特征和标签打包成数据集 并按batch_size进行批量加载
        train_iter = load_array((train_features, train_labels), batch_size)
        #这里使用adam优化器
        optimizer = torch.optim.Adam(net.parameters(),
                                     lr = learning_rate,
                                     weight_decay = weight_decay)
        for epoch in range(num_epochs):#开始一个训练循环 迭代num_epochs次 每个循环代表一个训练周期
            for X, y in train_iter:#遍历训练迭代器
                optimizer.zero_grad()#每个训练批次开始时 清零优化器所有参数的梯度
                l = loss(net(X), y)#进行前向传播 计算预测值 计算损失
                l.backward()#进行反向传播 会计算损失函数相对于模型所有可学习参数的梯度
                optimizer.step()#根据计算出的梯度更新模型参数
            #每轮训练结束时 计算整个训练集上对数均方误差 并添加到train_ls列表中
            train_ls.append(log_rmse(net, train_features, train_labels))
            if test_labels is not None:#检查是否提供了测试集标签
                test_ls.append(log_rmse(net, test_features, test_labels))
        return train_ls, test_ls#返回损失列表
    
    #K折交叉验证 是一种常用的模型评估方法 它将数据集分成k个子集 每次用k-1个子集训练模型 用剩下的1个子集验证 重复k次
    def get_k_fold_data(k, i, X, y):#i 表示当前是第几折
        assert k > 1#至少需要2折
        fold_size = X.shape[0] //k #计算每折大小 X.shape[0]数据集样本数量 //k是整数除法 得到每折大致包含的样本数
        X_train, y_train = None, None #init
        for j in range(k):#循环k次
            #计算当前折的索引范围 slice对象用于创建切片 j * fold_size当前折的起始索引 (j + 1) * fold_size当前折的结束索引
            idx = slice(j * fold_size, (j + 1) * fold_size)
            X_part, y_part = X[idx, :], y[idx]#提取当前折的特征和标签?y[idx]是一维的 选择idx 不需要： ：表示选择所有的行或者列
            if j == i:#判断当前折j是否是我们要作为验证集的那一折i
                X_valid, y_valid = X_part, y_part#如果是 则将当前的折作为验证集
            elif X_train is None:#如果当前折j不是验证集 并且X_train还是空的即这是第一次遇到训练集数据
                X_train, y_train = X_part, y_part
            #当前折不是验证集且X_train已经有数据了即之前已经有其它折的数据被添加到训练集中
            else:
                X_train = torch.cat([X_train, X_part], 0)#使用cat函数将当前折的特征X_part与现有的训练集特征X_train沿着dim=0即行方向拼接起来
                y_train = torch.cat([y_train, y_part], 0)
        #返回k折交叉验证中指定折的训练集特征、训练集标签、验证集特征、验证集标签
        return X_train, y_train, X_valid, y_valid
    #执行完整的k折交叉验证训练过程
    def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
        train_l_sum, valid_l_sum = 0, 0#初始化训练损失总和和验证损失总和 这些将计算所有折的平均损失
        for i in range(k):#循环k次 每次执行k折交叉验证中的一轮
            #data是一个元祖 包含X_train_fold,y_train_fold,X_valid_fold,y_valid_fold
            data = get_k_fold_data(k, i, X_train, y_train)#获取当前折的训练数据和验证数据
            net = get_net()#在每一轮交叉验证开始时 创建一个新的 为经训练的神经网络模型net 这是确保每一折的训练都是从一个“干净”的模型开始 避免模型状态的泄露
            #调用train函数来训练当前折的模型 *data解包get_k_fold_data返回的训练和验证集 作为参数传给train函数 train函数会返回每个epoch的训练损失列表和验证损失列表
            train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
            train_l_sum += train_ls[-1]#将当前折训练的最后一个epoch的训练损失累加到train_l_sum
            valid_l_sum += valid_ls[-1]
            if i == 0:#如果是第一折
                #绘制训练和验证损失曲线图
                plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
            #打印当前折的训练轮次 以及该折训练的最后一个epoch的训练rmse和验证rmse
            print(f'折{i + 1}, 训练log rmse{float(train_ls[-1]):f}, '
                  f'验证log rmse{float(valid_ls[-1]):f}')
            return train_l_sum / k, valid_l_sum / k#返回所有K折的平均训练损失和平均验证损失
    
    k, num_epochs, lr, weight_decay, batch_size = 20, 800, 4, 0.0001, 64#num_epochs每个折训练100个epoch 5有些不合理
    # train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
    # print(f'{k}-折验证:平均训练log rmse: {float(train_l):f}, '
    #       f'平均验证log rmse: {float(valid_l):f}')
    #plt.show()

    #不使用K折 而是使用全部的数据进行训练
    def train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, 
                       batch_size):
        net = get_net()
        #None None传给test_features, test_labels表示我们用所有数据进行训练 训练中不进行测试集的评估
        #_表示不关心这个值
        train_ls, _ = train(net, train_features, train_labels, None, None,
                            num_epochs, lr, weight_decay, batch_size)
        plot(np.arange(1, num_epochs + 1), [train_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'训练log rmse: {float(train_ls[-1]):f}')#打印模型在训练结束时-最后一个epoch的训练日志RMSE
        #将网络应用于测试集
        #使用训练好的net模型对test_features进行预测
        #net(test_features)前向传播 得到预测值(pytorch张量)
        #detach是将预测值从计算图中分离出来 这样就不会再计算其梯度 numpy 张量-》数组
        preds = net(test_features).detach().numpy()
        #将其重新格式化以导出到Kaggle
        #创建一个pd.Series，包含模型的预测结果 并将其命名为SalePrice
        #preds.reshape(1, -1)[0]：将预测结果preds--通常是列向量或者是扁平数组--重塑为1行 然后取第一行 这样确保它是一个一维数组 方便创建series
        #将这个包含预测值的series作为新列添加到test_data DataFrame中 列名为SalePrice（这通常是kaggle提交文件的目标列名）
        #Series是padas里的重要数据结构 通常是一维度带idx的列表 可以看成一张表格中的一列数据 由量部分组成：1数据2index
        #类似于Numpy里的数组和python的字典
        test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
        #创建kaggle提交文件所需要的DataFrame
        #test_data['Id']：从原始test_data中获取‘id’列--通常是每个样本的唯一标识符
        #axis=1沿列的方向拼接
        submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
        submission.to_csv('submission.csv', index=False)#表示不将DataFrame的索引写入CSV文件
    #生成一个名为xxx.csv的文件
    # train_and_pred(train_features, test_features, train_labels, test_data,
    #                 num_epochs, lr, weight_decay, batch_size)
    # plt.show()

    #chapter 5 深度学习计算

    import torch
    #functional是nn的子模块，它包含了一些通常没有可学习参数的函数，如激活函数-ReLU/sigmoid，池化操作(max pool\average pool)
    from torch.nn import functional as F

    net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

    X = torch.rand(2, 20)#2代表批量大小batch_size 即一次处理2个样本 20代表20个特征
    print(net(X))

    class MLP(nn.Module):
        # 用模型参数声明层这里，我们声明两个全连接层
        def __init__(self):
            #这里调用MLP的父类的构造函数来执行必要的init
            #这样在类的实例化时也可以指定其他函数参数 如模型参数params
            super().__init__()
            self.hidden = nn.Linear(20, 256) # hidden layer
            self.out = nn.Linear(256, 10) # out layer
        
        # 定义模型的前向传播 即如何根据输入X返回所需的模型输出
        def forward(self, X):
            # 注意这里我们使用ReLU的函数版本，其在nn的functional模块中定义
            return self.out(F.relu(self.hidden(X)))
    
    net = MLP()
    #print(net(X))

    #################################################################################
    # 这段代码演示了如何从头开始构建一个类似于 PyTorch 内置 nn.Sequential 的自定义模型。      
    # 它展示了 nn.Module 的基本结构，包括 __init__ 方法（用于初始化和注册子模块）和 forward   
    # 方法（定义数据流经模块的计算过程）。通过这个例子，你可以更好地理解 PyTorch 模块化的设计理念。
    #################################################################################

    #nn.Module是pytorch中所有神经网络模块的基类 自定义的神经网络层、模型都应该继承自nn Module模块
    #继承Module会自动为你提供很多有用的功能 比如参数管理、子模块注册等
    class MySequential(nn.Module):
        #*args是一个可变参列表 表示构造函数可以接受任意数量的非关键字参数
        #在这里 它用于接收传递给MySequential的各个神经网络层
        def __init__(self, *args):#self代表实例本身 C++里的this
            #__init__函数将每个块逐个添加到有序字典_modules中
            super().__init__()
            for idx, module in enumerate(args):#用于遍历*args中传递进来的所有模块 enumerate(args)这个函数会返回一个iterator 每次迭代产生一个（索引/值）对
                #idx是当前模块在args中的索引 module当前迭代到的模块实例（如nn.Linear对象 nn.ReLU对象）
                #这里的module是Module子类的一个实例 我们把它保存在Module类的成员变量_modules中
                #_module的类型是OrdereDict
                #_modules是nn.Module的内部字典 用于这册其子模块 通过将子模块存储在这个字典中 pytorch能自动识别这些子模块
                #并正确地处理它们的参数（如调用model.parameters()时，这些子模块的参数也会被包含进去
                # register
                self._modules[str(idx)] = module#将当前的model注册为MySequential实例的一个子模块
        #这是一个特殊的方法 是pytorch神经网络模块的核心 所有nn.Module子类都必须实现这个方法
        #在模型实例上调用model(input_data)时就是调用这个方法
        def forward(self, X):
            #OrdereDict保证了按照成员的添加顺序来遍历他们
            #_modules是OrdereDict类型 因此在遍历时可以保持模块的添加顺序 这对于sequential类型的模型至关重要 因为数据需要按特定顺序流过不同的层
            for block in self._modules.values():#遍历_modules中所有已经注册的子模块
                #这个是前向传播的核心逻辑
                #每次循环中 当前的输入X会被传递给当前的block子模块处理
                #block(X)会调用该子模块的forward方法 并返回处理结果
                #block(X)的输出会重新赋给X。这样，下一个循环中的block将会接收到上一个block的输出作为输入 实现了数据在模块间顺序流动
                X = block(X)
            return X
    
    net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
    net(X)
    #print(net(X))
    
    #######################################################################
    # 组合不同模块来构建复杂模型
    #######################################################################

    #在前向传播函数中执行代码
    #特点是隐藏层权重固定不变
    class FixedHiddenMLP(nn.Module):
        def __init__(self):
            super().__init__()
            #不计算梯度的随机权重参数，因此其在训练期间保持不变 rand_weight是一个不参与梯度计算的随机权重参数 因此在训练过程中不会被更新
            self.rand_weight = torch.rand((20,20), requires_grad=False)
            self.Linear = nn.Linear(20, 20)
        #定义模块的前向传播逻辑
        #当model(input)被调用时就会执行这个函数
        def forward(self, X):
            X = self.Linear(X)#X首先通过这个线性层进行一次变换
            # 使用创建的常量参数以及relu和mm函数
            # 用一个固定权重的隐藏层
            X = F.relu(torch.mm(X, self.rand_weight) + 1)#模拟一个没有bias的线性变换
            # 复用全连接层 这相当于两个全连接层共享参数(同一个参数)
            X = self.Linear(X)
            # 控制流
            while X.abs().sum() > 1:
                X /= 2
            return X.sum()
    
    net = FixedHiddenMLP()
    #print(net(X))

    # 混合搭配各种组合块的方法
    class NestMLP(nn.Module):
        def __init__(self):
            super().__init__()
            #定义了一个net作为NestMLP的子模块 这是NestMLP第一个组成部分
            self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                     nn.Linear(64, 32), nn.ReLU())
            #定义了另一个名为linear的线性层作为NestMLP子模块 独立于self.net
            self.linear = nn.Linear(32, 16)

        def forward(self, X):
            return self.linear(self.net(X))#模块的嵌套和组合
    #创建一个名为chimera的nn.Sequential模型 这是一个复杂的模型 它将之前定义的模块（甚至是你自己定义的模块）组合在一起
    chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
    #print(chimera(X))

    #参数管理
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    X = torch.rand(size=(2, 4))
    print(net)
    print(net(X))

    #检查第二个全连接层的参数
    #模型就像一个列表一样 每层的参数都在其属性中
    print(net[2].state_dict())

    #每个参数都表示为参数类的一个实例
    print(type(net[2].bias))
    print(net[2].bias)
    print(net[2].bias.data)

    print(net[2].weight.grad == None)

    #一次性访问所有参数
    print(*[(name, param.shape) for name, param in net[0].named_parameters()])
    print(*[(name, param.shape) for name, param in net.named_parameters()])
    
    print(net.state_dict()['2.bias'].data)

    #从嵌套块收集参数
    def block1():
        return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                             nn.Linear(8, 4), nn.ReLU())
    
    def block2():
        net = nn.Sequential()
        for i in range(4):
            # 在这里嵌套
            net.add_module(f'block {i}', block1())
        return net
    
    rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
    rgnet(X)

    print(rgnet)
    print(rgnet[0][1][0].bias.data)

    #参数的初始化
    #深度学习框架默认是随机化初始化 pytorch会根据一个范围均匀地初始化权重和偏置矩阵
    #这个范围是根据输入维度和输出维度计算的
    #pytorch的nn.init模块提供了多种内置初始化的方法
    def init_normal(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.zeros_(m.bias)
    
    net.apply(init_normal)
    print(net[0].weight.data[0], net[0].bias.data[0])

    #init as constant
    def init_constant(m):
        if type(m) == nn.Linear:
            nn.init.constant_(m.weight, 1)
            nn.init.zeros_(m.bias)
    
    net.apply(init_constant)
    print(net[0].weight.data[0], net[0].bias.data[0]) 

    #针对某些块应用不同的初始化方法
    def init_xavier(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
    
    def init_42(m):
        if type(m) == nn.Linear:
            nn.init.constant_(m.weight, 42)
    
    net[0].apply(init_xavier)
    net[2].apply(init_42)
    print(net[0].weight.data[0])
    print(net[2].weight.data)

    #自定义初始化
    def my_init(m):
        if type(m) == nn.Linear:
            print("init", *[(name, param.shape) for name, param in m.named_parameters()][0])#最后一个[0]代表打印第一列也就是weight 不打印bias
            nn.init.uniform_(m.weight, -10, 10)#使用uniform_函数对对模块m的权重进行均匀分布初始化 权重的值将从-10到10之间随机均匀采样
            #m.weight.data.abs() >= 5创建一个布尔张量 True表示>=5 否则为false
            #True为1 false为0
            #代码效果是如果权重的绝对值小于5 则将其设置为0 大于等于5保持原始值 
            #这相当于一个阈值操作 将小权重归0
            m.weight.data *= m.weight.data.abs() >= 5

    net.apply(my_init)
    print(net[0].weight[:2])

    #我们可以直接设置参数
    net[0].weight.data[:] += 1
    net[0].weight.data[0, 0] = 42
    print(net[0].weight.data[0])

    #参数绑定
    #有时候我们希望在多个层之间共享参数 我们定义一个稠密层 然后使用这个稠密层的参数来设置另一个层的参数
    # 我们给共享层提供一个名称 以便可以引用它的参数
    shared = nn.Linear(8, 8)
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                        shared, nn.ReLU(),
                        shared, nn.ReLU(),
                        nn.Linear(8, 1))
    net(X)
    #检查参数是否相同
    #同一个shared的实例 因此具有相同的内存地址
    print(net[2].weight.data[0] == net[4].weight.data[0])
    net[2].weight.data[0,0] = 100
    #确保它们是同一个对象 而不只是相同的值
    #说明第三个和第五个神经网络层的参数是绑定的 不仅相等 而且由相同的张量表示
    #当参数绑定时 梯度会发生什么情况？
    #模型参数包括梯度 因此反向传播时第三个神经网络层和第五个神经网络层的梯度会加在一起
    #这个是梯度的累积原理
    #########################################################################
    #     梯度会加在一起的原因在于参数共享意味着多个层（或操作）实际上引用的是内存中的同一个张量作为它们的参数。
    # 在PyTorch（以及大多数深度学习框架）中，反向传播（或称为链式法则）的目的是计算损失函数关于每个模型参数的梯度。当多个操作共享同一个参数时，发生的情况如下：
    # * 单个参数实例： 尽管在模型结构中，你可能多次使用了 shared 层，但在内存中，shared.weight 和 shared.bias 实际上只有一份拷贝。
    # * 每个使用都会产生局部梯度： 当模型进行前向传播时，每个使用 shared 层的计算路径都会产生一个对 shared 层参数的“局部”梯度。例如，net[2] 会基于它的输入和后续计算对 shared.weight 产生一个梯度贡献，net[4] 也会基于它的输入和后续计算对同一个 shared.weight 产生一个梯度贡献。
    # * 梯度累积： PyTorch 的自动求导系统（Autograd）在反向传播时，会将所有对同一个张量的梯度贡献进行累加。这是因为，为了更新这个唯一的参数实例，你需要知道所有依赖于它的计算路径的总影响。如果梯度不加在一起，那么参数的更新就会不完整，因为它只考虑了部分计算路径的影响。
    # 具体来说，如果一个参数 W 被两个不同的计算路径 P1 和 P2 使用，并且 L 是最终的损失函数：
    # * 通过路径 P1，你会计算 \frac{\partial L}{\partial W_1}
    # * 通过路径 P2，你会计算 \frac{\partial L}{\partial W_2}
    # 由于 W1 和 W2 实际上是同一个参数 W，总的梯度是这两个局部梯度的和：
    # \frac{\partial L}{\partial W} = \frac{\partial L}{\partial W_1} + \frac{\partial L}{\partial W_2}
    # 这就是梯度累积的原理。每次 .backward() 调用，计算出的梯度都会默认累加到参数的 .grad 属性上。当多个层共享同一个参数时，所有这些层对该参数产生的梯度都会被加到该参数的唯一的 .grad 属性中。
    # 总结来说，梯度会加在一起是因为：
    # * 内存中只有一份参数副本： 共享参数的层并没有各自独立的参数副本，它们都指向同一个底层张量。
    # * 链式法则的累积效应： 损失函数对这个唯一参数的梯度，是所有经过该参数的计算路径的梯度之和。每次反向传播，框架都会负责将这些局部梯度正确地加到共享参数的梯度属性中。
    # 这种行为对于实现某些神经网络架构（如循环神经网络中的权重共享）非常重要，因为它使得模型能够学习到在不同时间步或不同部分之间共享的特征表示。
    #########################################################################
    print(net[2].weight.data[0] == net[4].weight.data[0])

    #延后初始化
    #defer initialization
    #框架的延时初始化 即直到数据第一次通过模型传递时 框架才会动态地推断出每个层的大小
    #大大的简化了定义和修改模型的任务
    #实例化网络：
    #首先 我们实例化一个多层感知机 输入维数是未知的
    #接下来 我们让数据通过网络 最终使框架初始化参数
    #一旦知道输入维数是20， 框架就可以通过代入值20来识别第一层权重矩阵的形状
    #识别出第一层形状后 框架处理第二层 以此类推 直到所有形状已知为至
    #这样的情况下 只有第一层需要延后初始化，但框架仍然是按顺序初始化的


    #自定义层
    #深度学习成功的背后的一个因素就是神经网络的灵活性 我们可以用创造性的方式组合不同的层 从而设计出适用于各种任务的架构
    #如：研究人员设计出适用于处理图像 文本 序列数据 执行动态规划的专有层
    #有时候我们会遇到或者自己要发明一个目前在深度学习框架中尚不存在的层
    #这样情况下 必须构建自定义层

    #首选构造一个没有任务参数的自定义层
    class CenteredLayer(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, X):
            #这是个逐元素操作 不会改变原始张量的X的维度或者每个维度的大小 输出张量的形状与输入张量X的形状完全相同
            return X - X.mean()#接收输入张量X 然后减去X的平均值 这实现了数据的“中心化”操作 即让数据的均值为0 这个操作没有可以学习的参数
    #向层提供一些数据 验证它是否按预期工作
    layer = CenteredLayer()
    print(layer(torch.FloatTensor([[1,2,3,4,5],[2,3,4,5,6]])))

    # 我们可以将层作为组件合并到更复杂的模型中
    net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())

    #test
    Y = net(torch.rand(4, 8))
    print(Y)
    print(Y.mean())#输出是个标量

    #带参数的层
    #这些参数可以通过训练调整
    #我们可以使用内置函数来创建参数 这些函数提供了一些基本的管理功能 如管理访问、初始化、共享、保存和加载模型参数
    #这样做的好处之一是我们不需要为每个自定义层编写自定义的序列化程序
    #由于框架已经提供参数的管理机制 我们不需要为每个自定义层手动编写复杂的序列化（保存/加载）代码

    ############################################################################################
    # * 加入 in_units 和 units （或 PyTorch 官方 nn.Linear 中的 in_features 和 out_features）这两个参数对于定义任何线性层（全连接层）来说是必不可少的，原因如下：
    # * 确定权重矩阵的形状 (Shape of the Weight Matrix):
    # * 在一个全连接层中，输入特征 X 会与一个权重矩阵 W 进行矩阵乘法。
    # * 如果输入 X 的形状是 (batch_size, in_units)（即每个样本有 in_units 个特征），那么为了进行矩阵乘法 X @ W，权重矩阵 W 的列数必须与 X 的行数相匹配，即 W 的形状必须是 (in_units, units)。
    # * 这里的 in_units 就指定了权重矩阵的行数，units 指定了权重矩阵的列数。
    # * 最终输出的形状将是 (batch_size, units)。
    # 没有这两个参数，你就无法正确地定义和初始化权重矩阵 self.weight = nn.Parameter(torch.randn(in_units, units))。
    # * 确定偏置向量的形状 (Shape of the Bias Vector):
    # * 偏置（bias）是一个向量，它会被加到线性变换的结果上。
    # * 偏置向量的维度必须与输出特征的维度相匹配。也就是说，如果输出是 units 个特征，那么偏置向量的形状就应该是 (units,)。
    # * self.bias = nn.Parameter(torch.randn(units,)) 中的 units 参数就是用来定义偏置向量的大小的。
    # * 定义层的接口 (Defining the Layer's Interface):
    # * 这两个参数明确地告诉用户这个线性层期望什么样的输入数据（多少个特征），以及它会产生多少个输出特征。
    # * 这使得用户在使用你的自定义层时，能够正确地连接它与其他层，确保数据流的维度匹配。
    # * 通用性和可配置性 (Generality and Configurability):
    # * 通过将 in_units 和 units 作为参数，你的 MyLinear 层变得通用化，可以应用于任何输入和输出特征数量的场景，而不需要为每种特定的输入/输出维度组合编写一个新的类。
    # * 例如，你可以创建 MyLinear(10, 5)，也可以创建 MyLinear(128, 1)，它们都使用同一个 MyLinear 类定义。
    # 总结来说，in_units 和 units 这两个参数是定义任何线性层（全连接层）的“蓝图”。它们：
    # * 决定了层内部可训练参数（权重和偏置）的维度。
    # * 定义了层接受的输入和产生的输出的特征数量。
    # * 使层具有高度的通用性和可重用性。
    # 如果没有这两个参数，自定义的 MyLinear 层将不知道如何构建其权重和偏置，也就无法执行正确的线性变换。
    ############################################################################################

    class MyLinear(nn.Module):
        def __init__(self, in_units, units):#接受两个参数：输入特征的数量 输出特征的数量
            super().__init__()
            #用内置函数Parameter来创建层的层的参数
            #nn.Parameter是pytorch中用于表示模型参数的特殊张量类 它会自动被nn.Module识别为可训练参数并包含在model.parameters()中
            #torch.randn(in_units, units)初始化形状为(in_units, units)的随机张量，值从标准呢正太分布中采样
            self.weight = nn.Parameter(torch.randn(in_units, units))
            #torch.randn或者其他张量创建函数指定形状时 形状参数通常期望一个表示维度的元组
            #torch.randn（）函数的size参数期望的是一个可迭代对象（通常是元组或列表），其中每个元素代表一个维度的大小
            #self.bias是一个偏置向量 它的维度应该和输出特征的数量units相匹配 所以我们需要创建一个具有units个元素的一维张量
            self.bias = nn.Parameter(torch.randn(units,))#units加，在python中表示这是个元组，即使它只包含一个元素
        def forward(self, X):
            linear = torch.matmul(X, self.weight.data) + self.bias
            return F.relu(linear)
    
    linear = MyLinear(5, 3)
    print(linear.weight)

    #使用自定义层直接执行前向传播计算
    print(linear(torch.rand(2, 5)))

    #使用自定义层构建模型
    net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
    print(net(torch.rand(2, 64)))


    #读写文件
    #单个张量 load和save函数读写它们
    #arange与python里的range Numpy中的np.arange()类似 用于生成一个连续的张量
    #device = torch.device("mps")
    print(torch.mps.device_count())
    x = torch.arange(4, device='mps:7')
    torch.save(x, 'x-file')

    x2 = torch.load('x-file')
    print(x2)

    #张量列表
    y = torch.zeros(4)
    torch.save([x , y], 'x-file')

    x2, y2 = torch.load('x-file')
    print(x2, y2)

    #从字符串映射到张量的字典
    mydict = {'x': x, 'y': y}
    torch.save(mydict, 'mydict')
    mydict2 = torch.load('mydict')
    print(mydict2)

    #加载和保存模型参数
    #深度学习框架提供了内置函数来保存和加载整个网络 注意的细节：这将保存模型的参数而不是整个模型
    #例如：如果有一个3层的多层感知机，我们需要单独制定架构。因为模型可以包含任意代码，所以模型本身难以序列化
    #因此为了恢复模型 我们需要用代码生成架构 然后从磁盘加载参数

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(20, 256)
            self.output = nn.Linear(256, 10)

        def forward(self, x):
            return self.output(F.relu(self.hidden(x)))

    net = MLP()
    X = torch.randn(size=(2, 20))
    Y = net(X)

    #将模型参数存储在'mlp.params'file
    torch.save(net.state_dict(), 'mlp.params')

    #为了恢复模型 我们实例化了原始多层感知机模型的一个备份 这里我们不需要随机化模型参数 而是直接读取文件中的参数
    clone = MLP()
    clone.load_state_dict(torch.load('mlp.params'))
    clone.eval()#切换到评估模式
    print(clone)

    Y_clone = clone(X)
    print(Y_clone == Y)

    #验证macbook是否支持pytorch的GPU
    print(torch.backends.mps.is_available())
    print(torch.backends.mps.is_built())

    #我们定义两个方便的函数，这两个函数允许在不存在所需GPU的情况下执行代码
    def try_gpu(i=0): #@save
        """如果存在,则返回gpu(i),否则返回cpu()"""
        if torch.cuda.device_count() > i + 1:
            return torch.device(f'cuda:{i}')
        return torch.device('cpu')

    def try_all_gpus(): #@save
        """返回所有可用的GPU,如果没有GPU 则返回[CPU(),]"""
        devices = [torch.device(f'cuda:{i}')
                 for i in range(torch.cuda.device_count())]
        
        return devices if devices else [torch.device('cpu')]

    print(try_gpu(), try_gpu(10), try_all_gpus())


    #张量与GPU
    x = torch.tensor([1, 2, 3])
    m = x.device
    print(m)
    print(x.device)

    #存储在GPU上
    #X = torch.ones(2, 3, device=try_gpu())
    X = torch.ones(2, 3, device='mps:0')
    print(X)

    Y = torch.rand(2, 3, device='mps:1')
    print(Y)

    Z = X
    Z.to("mps:0")
    print(Z)

    print(Z + Y)


    # NN and GPU
    # 只要所有的数据和参数在同一个设备上 我们就可以有效地学习模型
    # 神经网络模型可以指定设备
    net = nn.Sequential(nn.Linear(3 , 1))
    net = net.to(device="mps:0")
    print(net(X))

    print(net[0].weight.data.device)


    #Chapter 6 CNN
    #实现图像卷积

    import torch
    from torch import nn
    from PIL import Image
    import matplotlib.gridspec as gridspec

    if torch.backends.mps.is_available():
        device = torch.device("mps:0")
        print("Using  Apple  silicon MPS GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    def corr2d(X, K): #@save
        """二维互相关运算"""
        h, w = K.shape# 获取卷积核 K 的高度 h 和宽度 w
        Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))# 初始化输出 Y。
                                                            # Y 的高度 = 输入 X 的高度 - 核 K 的高度 + 1
                                                            # Y 的宽度 = 输入 X 的宽度 - 核 K 的宽度 + 1
                                                            # 这是一个二维张量，初始值全为 0
        for i in range(Y.shape[0]):# 外层循环，遍历输出 Y 的行索引 i
            for j in range(Y.shape[1]):# 内层循环，遍历输出 Y 的列索引 j
                Y[i, j] = (X[i:i + h, j:j + w] * K).sum()   # 核心操作：计算当前输出像素 Y[i, j] 的值
                                                            # X[i:i + h, j:j + w] 从输入张量 X 中取出当前卷积窗口覆盖的区域
                                                            # K 是卷积核
                                                            # * 是元素级乘法（element-wise multiplication）
                                                            # .sum() 将元素级乘法的结果进行求和

        return Y# 返回计算得到的输出张量 Y

    X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    print("Chapter 6: CoV =====================================================================================")
    print(corr2d(X, K))
    ####################################################################################################
    #这段代码定义了一个名为 corr2d 的函数，用于执行二维互相关运算。它接收一个二维输入张量 X 和一个二维卷积核 K。函数通过滑动 K 在 X 
    # 上，并在每个位置进行元素级乘法和求和，来计算输出张量 Y。最后，代码通过一个示例 X 和 K 来演示 corr2d 函数的使用，并打印出计算结果。
    #在深度学习的背景下，这个 corr2d 函数实际上就是卷积神经网络中卷积层的核心计算逻辑（尽管在数学上严格的卷积会包含一个核的翻转操作，
    # 但深度学习框架中的“卷积”通常实现的是互相关，因为它不影响学习能力）
    ####################################################################################################

    #Conv Layer
    class Conv2D(nn.Module):
        def __init__(self, kernel_size):
            super().__init__()
            self.weight = nn.Parameter(torch.rand(kernel_size))
            self.bias = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return corr2d(x, self.weight) + self.bias

    # 图像边缘检测
    X = torch.ones((6, 8))
    X[:, 2:6] = 0
    print(X)

    K = torch.tensor([[0.999, -1.010]])

    Y = corr2d(X, K)
    #print(Y)

    #print(corr2d(X.t(), K))

    #  学习卷积核
    # 构造一个二维卷积层 它具有1个输出通道和形状为(1,2)的卷积核
    # nn.Conv2d是pytorch用于创建二维卷积层的类
    # 第一个1是输入通道数in_channels表示数据有一个输入通道 如灰度图像
    # 第二个1是输出通道数out_chanels表示卷积层会产生1个输出通道
    # kernel_size=（1，2）卷积核的大小，这里是1X2的卷积核即高度为1 宽度为2
    # bias=False表示卷积层不使用偏置项
    conv2d = nn.Conv2d(1, 1, kernel_size=(1,2), bias=False)

    # 这个二维卷积层使用4维输入和输出格式（批量大小，通道，高度，宽度）
    # 其中批量大小和通道数都为1
    X = X.reshape((1,1,6,8))#批量大小，通道数，高度，宽度
    Y = Y.reshape((1,1,6,7))#8-7 将目标数据Y重塑为1167 的形状 这是卷积层输出应该匹配的目标形状 
                            #可以看出宽度从8变为7 这是1X2卷积核在特定填充和步长下可能导致的变化？
    lr =3e-2#控制每次权重更新的步长

    for i in range(10):#20个epoch或者训练迭代
        Y_hat = conv2d(X)#前向传播
        l = (Y_hat - Y) ** 2#loss为MSE
        conv2d.zero_grad()#反向传播前清零梯度 避免梯度积累
        l.sum().backward()#执行反向传播 计算损失l相对于模型参数-这里主要是卷积核的权重的梯度 sum是为了将所有元素的平方差加起来
                          #得到一个标量损失 然后才能进行反向传播
        # 迭代卷积核
        conv2d.weight.data[:] -= lr * conv2d.weight.grad#手动更新卷积核的权重 这是一种简单的梯度下降更新方法
        #conv2d.weight.data获取卷积层权重的张量数据（不包括梯度信息）
        #weight.grad获取卷积层权重的梯度
        #lr * conv2d.weight.grad得到权重应该调整的量
        #conv2d.weight.data[:] -= lr * conv2d.weight.grad将权重减去调整量 实现梯度下降--寻找损失函数的最小值
        print(f'epoch {i+1}, loss {l.sum(): .3f}')
    
    print(conv2d.weight.data.reshape((1,2)))

    # 填充 padding
    import torch
    # 函数接受两个参数：一个卷积层对象和输入数据X 这个函数作用是封装卷积操作 并处理输入和输出的维度
    def com_conv2d(conv2d, X):
        # 这里的(1,1)表批量大小和通道数都是1
        # 这里是在卷积操作之前将X调整为Pytorch卷积层所期望的4维格式（批量大小 通道数 高度 宽度）
        X = X.reshape((1,1) + X.shape)#将（1，1）与X的原始形状拼接起来
        Y = conv2d(X)
        # 省略前两个维度 批量大小和通道
        # 将输出Y从4维恢复到2维 以便观察和后续处理
        return Y.reshape(Y.shape[2:])#Y.shape[2:]表示从Y的第三个维度取到最后一个维度 (1,1,6,6)->(6,6)
    
    # 创建一个nn.Conv2d对象 即1个二维卷积层 1 1 输入 输出通道数 填充设为1
    # 实际是添加了2行2列
    conv2d = nn.Conv2d(1,1,kernel_size=3,padding=1)
    X = torch.rand(size=(8,8))
    print(com_conv2d(conv2d,X).shape)

    conv2d = nn.Conv2d(1,1,kernel_size=(5,3) ,padding=(2,1))
    print(com_conv2d(conv2d,X).shape)

    # 步幅 stride
    conv2d = nn.Conv2d(1,1,kernel_size=3,padding=1, stride=2)
    print(com_conv2d(conv2d,X).shape)

    conv2d = nn.Conv2d(1,1,kernel_size=(3,5),padding=(0,1), stride=(3,4))
    print(com_conv2d(conv2d,X).shape)

    #多输入多输出通道
    def corr2d_multi_in(X, K):
        # 先遍历X和K的第0个维度(通道维度) 再把它们加在一起
        # 生成器表达式 zip函数是将XK在它们的第n个维度上进行配对
        return sum(corr2d(x, k) for x, k in zip(X, K))

    X = torch.tensor([[[0.0,1.0,2.0], [3.0,4.0,5.0],[6.0,7.0,8.0]],
                    [[1.0,2.0,3.0], [4.0,5.0,6.0], [7.0,8.0,9.0]]])

    K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
    print(corr2d_multi_in(X,K))

    # 多输出通道
    # 随着神经网络的层时的增加 我们通常会增加输出通道的维数 通过减少空间分辨率以获得更大的通道深度
    # 直观的说 我扪可以将每个通道看成对不同特征的响应

    def corr2d_multi_in_out(X, K):
        # 遍历卷积核的第一个维度 K的形状应该是(out_channels,in_channels, K_height,K_width)
        # corr2d_multi_in(X,k)X是原始的多输入通道数据 而k是当前正在计算的一个输出通道对应的所有输入通道的卷积核集合 它的输出
        # 结果就是一个输出通道的特征图
        # [...]这个是列表推导式 它会为K中的每一个输出通道生成一个特征图
        # torch.stack(..., 0) 将列表中所有的特征图沿着新的第0个维度（输出通道维度）堆叠起来 这样形成了具有多个输出通道的最终结果
        return torch.stack([corr2d_multi_in(X,k) for k in K], 0)
    #演示如何构造一个具有更多输出通道的卷积核
    # K+1创建一个与原k形状相同 但所有元素值都加1的新张量 以此类推 （K，K+1，K+2）形成元组 stack(...,0）将这三个形状相同
    # 的张量沿着第0维度堆叠起来
    K = torch.stack((K, K+1, K+2), 0)
    print(K.shape)

    print(corr2d_multi_in_out(X , K))
    ##################################################################################################
    # * 解释多输出通道的意义：允许神经网络学习并输出多种不同的特征表示，每个通道响应一种特定特征。
    # * 实现 corr2d_multi_in_out 函数：该函数通过遍历代表不同输出通道的卷积核，对每个输出通道独立地执行多输入通道的卷积（或相关）操作，然后将所有结果堆叠起来，形成最终的多输出通道特征图。
    # * 演示卷积核 K 的构造：通过 torch.stack 操作，将一个单输出通道的卷积核“扩展”为多输出通道的卷积核，这在实际中是由 nn.Conv2d 模块自动处理的。
    # * 展示多输入多输出通道卷积的完整流程：通过调用 corr2d_multi_in_out 函数，展示了如何从多输入通道数据和多输出通道卷积核得到多输出通道的特征图。
    # 这进一步深入解释了卷积神经网络中卷积层如何处理复杂的多维数据，以及如何通过不同的卷积核来提取和表示多样的特征。
    ##################################################################################################

    #1X1卷积层
    def corr2d_multi_in_out_1x1(X, K):
        c_i, h, w = X.shape
        c_o = K.shape[0]
        X = X.reshape((c_i, h * w))#将特征图的高度和宽度展平 以便后续进行矩阵乘法
        K = K.reshape((c_o, c_i))#由(c_o, c_i, 1,1)->(c_o, c_i) 4D->2D 成为一个标准矩阵 在1X1卷积中 每个
                                 #输出通道是所有输入通道的线性组合 K中的每个K[j,i]值代表了第i个输入通道对第j个输出通道的贡献权重！！！
        #全连接层中的矩阵乘法
        Y = torch.matmul(K, X)#1X1卷积在数学上等价于对每个像素位置执行一个全连接操作 其中连接的神经元数量等于通道数
                              #通过重塑操作可以将卷积操作转换为矩阵乘法
        return Y.reshape((c_o, h, w)) #将结果Y重塑回标准的特征图形
    
    # 用样本数据验证
    X = torch.normal(0, 1, (3, 3, 3))#mean std size
    K = torch.normal(0, 1, (2, 3, 1, 1))
    Y1 = corr2d_multi_in_out_1x1(X, K)
    Y2 = corr2d_multi_in_out(X, K)
    #assert float(torch.abs(Y1 - Y2).sum()) < 1e-6#<1e-6 阈值用于判断两个浮点数结果是否近似相等

    # 汇聚层 池化层
    def pool2d(X, pool_size, mode='max'):
        p_h, p_w = pool_size
        Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                if mode == 'max':
                    Y[i,j] = X[i:i + p_h, j:j + p_w].max()
                elif mode == 'avg':
                    Y[i,j] = X[i:i + p_h, j:j + p_w].mean()
        return Y
    
    X = torch.tensor([[0.0,1.0,2.0], [3.0,4.0,5.0], [6.0,7.0,8.0]])
    print(pool2d(X, (2,2)))
    print(pool2d(X, (2,2), 'avg'))

    # 汇聚层的填充和步幅padding stide
    X = torch.arange(16, dtype=torch.float32).reshape((1,1,4,4))#样本数和通道都是1
    print(X)

    pool2d = nn.MaxPool2d(3) #使用形状为3，3的汇聚窗口 默认情况步幅形状为3，3
    print(pool2d(X))

    #手动设定padding and stripe
    pool2d = nn.MaxPool2d(3, padding=1, stride=2)
    print(pool2d(X))

    #设定任意大小的矩形窗口并设定padding和stipe的高度和宽带
    pool2d = nn.MaxPool2d((2,3), padding=(0,1), stride=(2,3))
    print(pool2d(X))

    X = torch.cat((X, X + 1), 1)
    print(X)
    pool2d = nn.MaxPool2d(3, padding=1, stride=2)
    print(pool2d(X))

    # #卷积神经网络LeNet
    # net = nn.Sequential(
    #     # 定义了一个二维卷积层 1是输入通道数 灰度图像通常是1个通道 6是输出通道数即卷积核的数量 
    #     # padding=2在输入图像周围添加2圈填充 有助于保持输出特征图的大小
    #     nn.Conv2d(1,6,kernel_size=2,padding=2), nn.Sigmoid(),
    #     #定义了一个二维平均池化层 窗口大小2X2 每次移动步长是2 这会使特征图的宽度和高度减半
    #     nn.MaxPool2d(kernel_size=2,stride=2),
    #     #定义一个二维卷积层 6是输入通道数 与上一个池化层的输出通道相对应
    #     nn.Conv2d(6,16,kernel_size=2), nn.Sigmoid(),
    #     #定义了第二个二维平均池化层 
    #     nn.MaxPool2d(kernel_size=2,stride=2),
    #     #微调：增加卷积层的数量
    #     nn.Conv2d(16,32,kernel_size=2), nn.Sigmoid(),
    #     nn.MaxPool2d(kernel_size=2,stride=2),
    #     #定义了展平层 它将所有输入维度展平为1维 为后面的全连接层做准备 如：前面的输出是(batch_size,channel,height,width)
    #     #展平后会变成(batch_size,channels*height*width)
    #     nn.Flatten(),
    #     #定义一个全连接层 16*5*5是输入特征数量 根据LeNet的结构，在经过两次卷积和池化后 特征图的大小通常是5x5 且有16个通道
    #     #所以展平后是16x5x5=400个特征 120是输出特征
    #     nn.Linear(32*3*3, 240), nn.ReLU(),
    #     #第二个全连接层 120与上一个全连接层输出对应 84是输出特征
    #     nn.Linear(240, 84), nn.ReLU(),
    #     #84是输入特征数量 10是输出特征数量 这通常对应于分类任务的类别数量
    #     nn.Linear(84,20))

    # 自定义net
    # --- (2) LeNet 模型定义：请根据你的实际模型结构进行确认 ---
# class LeNet(nn.Module):
#     def __init__(self):
#         super(LeNet, self).__init__()
#         # 根据你提供的输出形状推断的结构
#         self.conv1 = nn.Conv2d(1, 6, kernel_size=5) # 28x28 -> 24x24 (valid padding)
#         self.sigmoid1 = nn.Sigmoid()
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 24x24 -> 12x12
#         self.conv2 = nn.Conv2d(6, 16, kernel_size=5) # 12x12 -> 8x8
#         self.sigmoid2 = nn.Sigmoid()
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 8x8 -> 4x4
#         self.flatten = nn.Flatten() # 16*4*4 = 256
#         self.fc1 = nn.Linear(16 * 4 * 4, 120)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(120, 84)
#         self.relu2 = nn.ReLU()
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool1(self.sigmoid1(self.conv1(x)))
#         x = self.pool2(self.sigmoid2(self.conv2(x)))
#         x = self.flatten(x)
#         x = self.relu1(self.fc1(x))
#         x = self.relu2(self.fc2(x))
#         x = self.fc3(x)
#         return x
    
#     net.to(device)
    
#     X = torch.rand(size=(1,1,28,28), dtype=torch.float32)#创建一个随机的张量X作为输入数据 批次大小1 输入通道1 高度 宽度
#     X = X.to(device)
#     for layer in net:#开始循环 遍历net 即nn.Sequential中的每一个子层
#         X = layer(X)#将当前的输入X通过当前循环的layer进行向前传播 并将输出返回给X 这样x会在每次迭代中更新为经过当前层处理后的输出
#         print(layer.__class__.__name__, 'output shape: \t', X.shape)
#         #打印当前层的类名
#         #然后打印一个制表符和字符串 最后打印经过层处理的向量X的形状
#         #这样可以观察数据在网络中流动的形状变化
#         #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#         # * 卷积层和池化层的输出特性：
#         # * 卷积层（nn.Conv2d）和池化层（nn.AvgPool2d）通常处理的是具有空间结构的数据，比如图像。它们的输出仍然保持着这种多维结构，
#         # 例如一个图像经过几次卷积和池化后，可能得到一个形状像 (batch_size, 16, 5, 5) 的张量，表示批次大小、通道数、特征图的高度和
#         # 宽度。
#         # * 全连接层（线性层）的输入要求：
#         # * 全连接层（nn.Linear）的特点是它的每个神经元都与前一层的所有神经元相连接。它期望的输入通常是一个二维张量，其中第一个维度是
#         # 批次大小，第二个维度是特征向量的长度。例如，如果你的全连接层需要100个输入特征，那么输入张量的形状应该是 (batch_size, 100)。
#         # * 连接卷积/池化层和全连接层：
#         # * 由于卷积/池化层的输出是多维的（例如 (batch_size, 16, 5, 5)），而全连接层期望的是一维特征向量，因此需要一个转换步骤。
#         # nn.Flatten() 的作用就是完成这个转换。它会将除了批次维度之外的所有维度展平，将 (batch_size, 16, 5, 5) 转换为 
#         # (batch_size, 16 * 5 * 5)，也就是 (batch_size, 400)。
#         # * 在 LeNet 中的具体应用：
#         # * 在 LeNet 结构中，经过两次卷积和两次池化后，输出的特征图尺寸是 5x5，通道数是 16。因此，展平后会得到 
#         # 16 \times 5 \times 5 = 400 个特征。
#         # * nn.Flatten() 层之后紧接着的 nn.Linear(16*5*5, 120) 层，其 in_features 参数正是 16*5*5，
#         # 这与 nn.Flatten() 的输出维度完美匹配。
#         # 总而言之，nn.Flatten() 的作用是作为卷积层和全连接层之间的“桥梁”，将具有空间信息的特征图转换为一维的特征向量，
#         # 从而使数据能够被全连接层处理。
#         #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#     #(4)
#     # --- 注册钩子 ---
#     # 存储激活值的字典
#     activations = {}

#     def get_activation(name):
#         def hook(model, input, output):
#             activations[name] = output.detach()
#         return hook


#     # 辅助函数：可视化激活值 #(4)
#     def plot_activations(input_tensor, activations_dict, title_prefix=""):
#         """
#         可视化指定层的激活值
#         input_tensor: 原始输入图像的Tensor (1, C, H, W)
#         activations_dict: 包含层名和对应激活值的字典 {layer_name: activation_tensor}
#         title_prefix: 图表标题前缀
#         """
#         # 调整输入图像的显示，因为你的LeNet输入是灰度图 (1通道)
#         input_img = input_tensor.squeeze().cpu().numpy()
#         # 计算总共需要多少个“层组”来显示激活
#         num_layers_to_plot = len(activations_dict)

#         # 创建一个主图，并使用 GridSpec 定义整体布局
#         # 一行两列的主网格：左边是输入图像，右边是所有激活层的大区域
#         # 调整图大小以适应多层：每个激活层组分配约4单位高度，输入图区域高度匹配
#         fig = plt.figure(figsize=(15, 4 + num_layers_to_plot * 2.5)) # 调整图大小，确保右侧空间足够
#         gs_main = gridspec.GridSpec(1, 2, width_ratios=[1, 4], figure=fig) # figure=fig 是新版本语法

#         # 显示原始图像 (占据主网格的左侧区域)
#         ax_input = fig.add_subplot(gs_main[0, 0])
#         ax_input.imshow(input_img, cmap='gray')
#         ax_input.set_title(f"{title_prefix} Input Image", fontsize=12)
#         ax_input.axis('off') # 关闭坐标轴

#         # 为所有激活层创建一个大的父区域，它将垂直划分给每个激活层
#         # 这个是主网格的右侧区域，它将承载所有层的激活图组
#         gs_all_activations_parent = gs_main[0, 1].subgridspec(num_layers_to_plot, 1, hspace=0.3)
#         # ^^^^ 关键：在主网格的右侧区域内，创建一个垂直堆叠的子网格，每个子网格用于一个层

#         fig.suptitle(f"{title_prefix} Model Activations", fontsize=16, y=0.98) # 总标题

#         # 遍历每个激活层，并在其分配的区域内绘制过滤器
#         for i, (layer_name, activation_tensor) in enumerate(activations_dict.items()):
#             # 获取当前层激活值的父 SubplotSpec
#             # 这是一个 SubplotSpec，它对应于 gs_all_activations_parent 中的一个行
#             current_layer_parent_spec = gs_all_activations_parent[i, 0]

#             # 提取激活值，并确保在CPU上并转换为numpy
#             layer_output = activation_tensor[0].cpu().numpy() # [0] 取第一个 batch，移到CPU并转为numpy

#             # 判断激活值形状：如果是 FC 层的扁平化输出，特殊处理
#             if layer_output.ndim == 1: # 假设是 (num_features,) 扁平化特征
#                 num_features = layer_output.shape[0]
#                 # 为了可视化，将其重塑为近似正方形的二维网格
#                 rows_fc = int(np.ceil(np.sqrt(num_features)))
#                 cols_fc = (num_features + rows_fc - 1) // rows_fc

#                 # 为全连接层创建一个子网格（一个大的图，显示所有特征）
#                 sub_gs_layer = current_layer_parent_spec.subgridspec(1, 1) # 只有一个子图
#                 ax_layer = fig.add_subplot(sub_gs_layer[0, 0])
#                 #下面三行对齐缩进有问题么？
#                 ax_layer.imshow(layer_output.reshape(rows_fc, cols_fc), cmap='viridis', aspect='auto') # 重塑并显示
#                 ax_layer.set_title(f"{layer_name} (Features: {num_features})", fontsize=9)
#                 ax_layer.axis('off') # 关闭坐标轴
#             else: # 卷积层等具有多通道输出的层
#                 num_filters = layer_output.shape[0] # 获取过滤器数量

#                 # 计算一个合适的网格来显示所有滤波器
#                 rows_filters = int(np.ceil(np.sqrt(num_filters)))
#                 cols_filters = (num_filters + rows_filters - 1) // rows_filters

#                 # 在当前层父区域内创建子网格来显示所有滤波器
#                 sub_gs_layer = current_layer_parent_spec.subgridspec(rows_filters, cols_filters, wspace=0.05, hspace=0.05)

#                 # 为整个层组添加标题
#                 # 这里我们使用一个辅助轴来显示层名称，因为它位于 sub_gs_layer 的外部，可以作为整个组的标题
#                 ax_temp_title = fig.add_subplot(current_layer_parent_spec)
#                 ax_temp_title.set_title(f"{layer_name} ({num_filters} filters)", fontsize=10, pad=10)
#                 ax_temp_title.axis('off') # 隐藏这个辅助标题轴

#                 for j in range(num_filters):
#                     row_filter = j // cols_filters
#                     col_filter = j % cols_filters

#                     # 从 sub_gs_layer 获取一个子图规格，并添加到图中
#                     ax_filter = fig.add_subplot(sub_gs_layer[row_filter, col_filter])

#                     channel_activation = layer_output[j]

#                     # 对每个滤波器输出进行归一化以便显示
#                     min_val = channel_activation.min()
#                     max_val = channel_activation.max()
#                     if max_val - min_val > 1e-6: # 避免除以零
#                         display_activation = (channel_activation - min_val) / (max_val - min_val)
#                     else:
#                         display_activation = channel_activation # 如果全是相同值，直接显示
#                     #下面三行缩进问题？
#                     ax_filter.imshow(display_activation, cmap='viridis') # viridis 通常用于显示数值范围
#                     ax_filter.set_title(f"F{j+1}", fontsize=7) # 滤波器编号 (F1, F2...)
#                     ax_filter.axis('off') # 关闭坐标轴

#         plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局，为总标题留出空间
#         plt.show() # 显示所有图表

#         #for i, (layer_name, activation_tensor) in enumerate(activations_dict.items()):
#         #     # 激活值通常是 (batch_size, num_channels, height, width)
#         #     # 我们取第一个样本的激活值 (batch_size=1)
#         #     layer_output = activation_tensor[0].cpu().numpy()
#         #     num_filters = layer_output.shape[0]
        
#         # # 计算一个合适的网格来显示所有滤波器
#         # cols = int(np.ceil(np.sqrt(num_filters)))
#         # rows = int(np.ceil(num_filters / cols))

#         # # 为当前激活层创建一个新的子图区域（基于 current_ax_parent_spec）
#         # # current_ax_parent_spec 是一个 SubplotSpec 对象，现在可以在其上调用 subgridspec
#         # sub_gs = current_ax_parent_spec.subgridspec(rows, cols, wspace=0.05, hspace=0.05)


#         # # 添加一个主标题，描述当前激活层
#         # # 这里不能直接用 ax.set_title，因为 sub_gs 内部没有一个总的 Axes
#         # # 可以在循环外，或者在循环内部的每个层组上添加
#         # # 这里的 current_ax 是你在原代码中定义的，这里我们直接用 sub_gs 来创建子图
#         # # fig.suptitle(f"{title_prefix} Activations", fontsize=16) # 这个是总标题，放在外面
#         # ax_title = fig.add_subplot(current_ax_parent_spec) # 创建一个临时的轴来显示层名称
#         # ax_title.set_title(f"{layer_name} ({num_filters} filters)", fontsize=12, pad=10)
#         # ax_title.axis('off') # 隐藏这个临时的轴



#         # for j in range(num_filters):
#         #     row = j // cols
#         #     col = j % cols
#         #     # 从 sub_gs 获取一个子图规格，并添加到图中
#         #     sub_ax = fig.add_subplot(sub_gs[row, col])

#         #     channel_activation = layer_output[j]

#         # # 对每个滤波器输出进行归一化以便显示
#         # # 将数值范围映射到0-1，否则imshow可能显示不佳
#         # min_val = channel_activation.min()
#         # max_val = channel_activation.max()
#         # if max_val - min_val > 1e-6: # 避免除以零
#         #     display_activation = (channel_activation - min_val) / (max_val - min_val)
#         # else:
#         #     display_activation = channel_activation # 如果全是相同值，直接显示

#         # sub_ax.imshow(display_activation, cmap='viridis') # viridis通常用于显示数值范围
#         # sub_ax.set_title(f'{j+1}', fontsize=6) # 滤波器编号
#         # sub_ax.axis('off')
#         # plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局以避免标题重叠
#         # plt.show() # 显示所有图表

#     #注册钩子
#     activations.clear()
#     # 确保这里的层名称与你的 LeNet 模型中定义的层名称完全匹配
#     net.conv1.register_forward_hook(get_activation('conv1'))
#     net.sigmoid1.register_forward_hook(get_activation('sigmoid1'))
#     net.pool1.register_forward_hook(get_activation('pool1'))
#     net.conv2.register_forward_hook(get_activation('conv2'))
#     net.sigmoid2.register_forward_hook(get_activation('sigmoid2'))
#     net.pool2.register_forward_hook(get_activation('pool2'))
#     net.flatten.register_forward_hook(get_activation('flatten'))
#     net.fc1.register_forward_hook(get_activation('fc1'))
#     net.relu1.register_forward_hook(get_activation('relu1'))
#     net.fc2.register_forward_hook(get_activation('fc2'))
#     net.relu2.register_forward_hook(get_activation('relu2'))
#     net.fc3.register_forward_hook(get_activation('fc3'))
    
#     #train model
#     batch_size = 256
#     train_iter, test_iter = load_data_fashion_minist(batch_size=batch_size)

#     # 根据你的 net = nn.Sequential(...) 结构，第一个 nn.Conv2d 是 net[0]
#     # 第二个 nn.Conv2d 是 net[3] (因为中间有Sigmoid和MaxPool2d)
#     # 确保索引与你的实际模型层对应！
#     # 这两行代码是针对nn.Sequencial的
#     # hook_conv1 = net[0].register_forward_hook(get_activation('conv1'))
#     # hook_conv2 = net[3].register_forward_hook(get_activation('conv2')) # LeNet中的第二个卷积层

#     #(4) #TODO check it!
#     # --- 准备输入图像 ---
#     # 定义图像预处理，与你训练MNIST或FashionMNIST时使用的保持一致
#     transform = transforms.Compose([
#         transforms.Grayscale(num_output_channels=1),
#         transforms.Resize((28, 28)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,)) # MNIST数据集的均值和标准差
#     # 如果你训练的是FashionMNIST，请使用FashionMNIST的均值和标准差
#     # transforms.Normalize((0.2860,), (0.3530,)) # FashionMNIST的均值和标准差
#     ])
#     try:
#         mnist_test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
#         mnist_image_tensor, _ = mnist_test_dataset[0]
#         mnist_image_tensor = mnist_image_tensor.unsqueeze(0) # 添加 batch 维度
#     except ImportError:
#         print("PyTorch torchvision not found, creating dummy MNIST image tensor.")
#         mnist_image_tensor = torch.randn(1, 1, 28, 28)
#     mnist_image_tensor = mnist_image_tensor.to(device) # 确保输入图像在正确的设备上
#     print("\n--- Visualizing Activations for MNIST Digit ---")
#     output = net(mnist_image_tensor) # 进行前向传播
#     # 复制激活字典，因为第二次前向传播会覆盖它
#     mnist_activations = {name: act for name, act in activations.items()}
#     plot_activations(mnist_image_tensor, mnist_activations, title_prefix="MNIST Digit")

#     # # 3.1. 获取一个MNIST数字图像
#     # print("\n--- Preparing MNIST Image ---")
#     # mnist_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
#     # # 从测试集中获取第一个图像
#     # mnist_image_tensor, mnist_label = mnist_dataset[0]
#     # mnist_image_tensor = mnist_image_tensor.unsqueeze(0).to(device) # 添加batch维度并移到设备

#     # 自定义图像加载 (请替换为你的实际路径)
#     print("\n--- Visualizing Activations for Custom Image (Sweater/Jacket) ---")
#     try:
#         custom_image_path = '/Users/ruben/Downloads/sweater_jacket.png' # <-- **请在这里替换为你的实际图像路径**
#         custom_image = Image.open(custom_image_path).convert('L') # 转换为灰度图
#         custom_image_tensor = transform(custom_image).unsqueeze(0) # 应用同样的变换并添加 batch 维度
#     except FileNotFoundError:
#         print(f"Custom image not found at {custom_image_path}. Creating dummy custom image tensor.")
#         custom_image_tensor = torch.randn(1, 1, 28, 28) # 创建一个虚拟图像
#     custom_image_tensor = custom_image_tensor.to(device) # 确保在正确设备

#     # # 3.2. 准备一个自定义的“毛衣外套”或其他图片
#     # # 请确保 'sweater_jacket.png' 文件存在于与脚本相同的目录下
#     # custom_image_path = '/Users/ruben/Downloads/sweater_jacket.png'#'sweater_jacket.png'
#     # custom_image_tensor = None
#     # if os.path.exists(custom_image_path):
#     #     try:
#     #         custom_image = Image.open(custom_image_path).convert('L') # 确保灰度
#     #         custom_image_tensor = transform(custom_image)
#     #         custom_image_tensor = custom_image_tensor.unsqueeze(0).to(device) # 添加batch维度并移到设备
#     #         print(f"Loaded custom image from: {custom_image_path}")
#     #     except Exception as e:
#     #         print(f"Error loading custom image: {e}")
#     #         print("Using a random dummy image instead for custom_image_tensor.")
#     #         custom_image_tensor = torch.randn(1, 1, 28, 28).to(device) # 随机虚拟图片
#     # else:
#     #     print(f"Custom image '{custom_image_path}' not found. Using a random dummy image instead.")
#     #     custom_image_tensor = torch.randn(1, 1, 28, 28).to(device) # 随机虚拟图片

#     activations.clear() # 清空激活值以便进行新的捕获
#     output = net(custom_image_tensor) # 再次前向传播
#     custom_image_activations = {name: act for name, act in activations.items()}
#     plot_activations(custom_image_tensor, custom_image_activations, title_prefix="Custom Image (Sweater/Jacket)")

#     # #(4)
#     # # --- 进行前向传播并可视化 ---

#     # # 可视化 MNIST 图像的激活值
#     # print("\n--- Visualizing Activations for MNIST Digit ---")
#     # _ = net(mnist_image_tensor) # 进行前向传播
#     # # 复制 activations 字典，因为第二次前向传播会覆盖它
#     # mnist_activations = {name: act for name, act in activations.items()}
#     # plot_activations(mnist_image_tensor.cpu(), mnist_activations, title_prefix="MNIST Digit")

#     # # 可视化自定义图像的激活值
#     # print("\n--- Visualizing Activations for Custom Image ('Sweater/Jacket') ---")
#     # _ = net(custom_image_tensor) # 再次进行前向传播，更新activations字典
#     # custom_image_activations = {name: act for name, act in activations.items()}
#     # plot_activations(custom_image_tensor.cpu(), custom_image_activations, title_prefix="Custom Image (Sweater/Jacket)")

#     # --- 移除钩子（可选，但推荐在不再需要时清理） ---
#     # hook_conv1.remove()
#     # hook_conv2.remove()
#     # print("\nHooks removed.")
    

    


    # 对评估精度函数进行修改 因为在GPU上训练 需要将数据从内存copy到显存
    # 定义了一个用于评估模型在gpu上准确率的函数
    def evaluate_accuracy_gpu(net, data_iter, device=None): 
        """使用GPU计算模型在数据集上的精度"""
        if isinstance(net, nn.Module):#net是否是nn.Module的一个实例 即一个pytorch模型
            net.eval()#设置为评估模式 在评估模式下 Dropout/BatchNorm等层会表现出不同的行为 如dropout会被关闭
            if not device:#如果参数没有被指定
                #自动获取模型的第一个参数所在的设备 并将其设置为device 这样可以确保数据和模型在同一个设备上
                device = next(iter(net.parameters())).device
        #正确预测数量  总预测数量
        metric = Accumulator(2)#创建了一个Accumulator类的实例 用于积累两个值
        #进入一个torch.no_grad()上下文管理器 在这个上下文管理器中 pytorch不会计算梯度 这对于评估模型非常重要
        #因为我们不需要在评估时进行反向传播 可以省下内存和计算资源
        with torch.no_grad():
            for X, y in data_iter:
                #检查X是否是一个列表 这可能是为了处理一些特殊情况 如BERT模型可能需要多个输入（输入ID/注意力掩码等）
                #这些输入通常以列表的形式传入
                if isinstance(X, list):
                    #BERT微调所需要的 如果X是个列表 则遍历X中的每个元素x 并将其移动到指定的device（GPU/CPU）
                    X = [x.to(device) for x in X]
                else:
                    X = X.to(device)#如果不是列表 将特征张量y移动到指定的device
                y = y.to(device)
                #调用metric.add方法来更新累加 net(X)前向传播得到预测结果 调用accuracy函数 计算预测结果与真实标签y之间
                #的准确预测数量 y.numel()可能是为了获取预测数量的标量值
                metric.add(accuracy(net(X), y), y.numel())
        return  metric[0] / metric[1]
    
    def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
        """使用GPU训练模型"""
        def init_weights(m):#m是模型的子模块
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.xavier_uniform_(m.weight)#均匀分布初始化其权重 xavier是一种常用的初始化方法 有助于保持训练过程中梯度的方差稳定
        net.apply(init_weights)#将init_weights函数应用于神经网络net中所有的子模块 这意味着init_weights会遍历net中所有的层 并根据条件进行权重初始化
        print('training on', device)
        net.to(device)#将整个神经网络模型net移动到指定的device 如从CPU移动到GPU显存
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)#初始化一个随机梯度下降的优化器
        loss = nn.CrossEntropyLoss()
        #创建Animator对象 用于可视化训练过程中的损失和精度
        animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])#legend设置图例
        timer, num_batches = Timer(), len(train_iter)#num_batches存储训练迭代器中的批次数量
        for epoch in range(num_epochs):
            #训练损失之和 训练准确率之和 样本数
            metric = Accumulator(3)#创建Accumulator对象 累积三个值
            net.train()#将net置于训练模式 在训练模式下 Dropout/BatchNorm等层会开启训练时的行为
            for i, (X, y) in enumerate(train_iter):#开始遍历训练数据集中的每个批次 i是批次的索引 X是特征 y是标签
                timer.start()
                optimizer.zero_grad()
                X,y = X.to(device), y.to(device)
                y_hat = net(X)
                l = loss(y_hat,y)
                l.backward()
                optimizer.step()#根据计算出的梯度更新模型参数
                with torch.no_grad():#进入一个torch.no_grad上下文管理器 在这个管理器中 pytorch不会计算梯度
                    #更新metric累积器
                    #l * X.shape[0]当前批次的总损失 批次损失*批次大小
                    #accuracy(y_hat, y)计算当前批次的准确预测数量
                    #X.shape[0]当前批次的样本数
                    metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
                timer.stop()#记录当前批次的训练时间
                train_l = metric[0] / metric[2]#计算当前epoch的平均训练损失
                train_acc = metric[1] / metric[2]#计算当前epoch的训练准确率
                #条件语句 控制何时更新可视化器animator 表示每经过num_batches // 5即20%的批次或者当前epoch为最后一个批次时进行更新
                if (i + 1) % (num_batches // 5) == 0 or i == num_batches -1:
                    #将数据点添加到animator x轴的值是当前已经处理的epoch的比例
                    #添加的数据包括train_l train_acc None表示准确率在此时不更新 因为它是在整个epoch结束后才计算的
                    animator.add(epoch + (i + 1) / num_batches,
                                (train_l, train_acc, None))
            #在当前epoch训练结束后 调用evaluate_accuracy_gpu函数在测试集上评估模型的准确率
            test_acc = evaluate_accuracy_gpu(net, test_iter)
            animator.add(epoch + 1, (None, None, test_acc))#添加准确率
        #打印当前epoch训练损失和训练准确率 保留三位小数 打印当前epoch的测试准确率
        print(f'loss {train_l: .3f}, train acc {train_acc:.3f},'
              f'test acc {test_acc: .3f}')
        #打印训练速度 即每秒处理的样本数 metric[2] * num_epochs总样本数 timer.sum()总训练时间
        print(f'{metric[2] * num_epochs / timer.sum(): .1f} examples/sec '
              f'on {str(device)}')
#     lr, num_epochs = 0.9, 20
#     print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
#     train_ch6(net, train_iter, test_iter, num_epochs,lr, torch.device('mps:0'))#GPU
#     #CPU
#     #train_ch6(net, train_iter, test_iter, num_epochs,lr, try_gpu())
#     plt.show()

#     ############################################################################################################
#     # 1学习lr过低会导致早起梯度没有什么变化 浪费计算资源 学习lr过高要防止导致震荡或者不收敛。
#     # 2现在一般不考虑Sigmoid和Tanh激活函数，现代CNN普遍使用ReLU或者其变种Leaky ReLU、PReLU
#     # 3前期因为激活函数或者过低的学习率而学习慢，可以增加num_epoches
#     # 4卷积核由5X5改为2X2，卷积核大会增加感受野，从而在大的特征上表现好。卷积核小，配合网络深，可以学到更细致的特征，精确度提升
#     # 5池化Max和avg的区别 Max：特征选择能力更强 常用于提取纹理、边缘等显著特征。Avg：保留更多背景信息 常用于平滑特征、减少噪声
#     # 6调整全连接层的数量：更大的容量允许模型学习更复杂的非线形映射 这可能有助于在最终分类阶段获得更高的准确率 从120到240可能会提供更多
#     # 的“表达能力” 让网络能够更好的处理从卷积层提取的特征
#     # 7其他参数 如学习率 初始化 训练轮数？







    # * 增加 LeNet 的卷积层（从两层到三层）通常会有作用，但也可能没有作用，甚至可能产生负面影响，具体取决于多种因素：
    # * 数据集的复杂性：
    # * 如果您的数据集非常简单（例如 MNIST 手写数字，LeNet 原始设计的目标），额外的层可能带来冗余或过拟合。LeNet 的两层卷积已经足以学习这些简单图像的足够特征。
    # * 如果您的数据集更复杂，包含更多细节、更高分辨率或更丰富的纹理（例如 CIFAR-10，更复杂的自然图像），那么增加卷积层可以帮助模型学习更抽象、更高级的特征表示，从而可能提高性能。
    # * 梯度问题和激活函数：
    # * LeNet 原始使用 Sigmoid 激活函数。如果您像之前的代码一样，在所有层都使用 Sigmoid，那么增加层数会显著加剧梯度消失问题。模型可能会非常难以训练，出现学习停滞或收敛缓慢的情况，就像您之前看到的曲线在前期没有学习一样。
    # * 如果您将激活函数改为 ReLU（或其变种），那么增加层数会更有效，因为 ReLU 有助于缓解梯度消失，允许训练更深的网络。
    # * 参数数量和过拟合：
    # * 增加卷积层会增加模型的参数数量。
    # * 好处： 更大的模型容量可能使模型能够学习更复杂的映射。
    # * 坏处： 如果数据集不够大，或者增加的层数过多，模型可能会过拟合（在训练集上表现很好，但在测试集上表现差），因为模型有能力记忆训练数据中的噪声和特有模式，而不是学习通用规律。
    # * 计算资源和训练时间：
    # * 额外的层会增加模型的计算开销，导致训练时间更长，需要更多的 GPU 内存。
    # * 设计细节（通道数、卷积核、池化）：
    # * 如果增加一层，其 in_channels 应该与前一个池化层的 out_channels 匹配。
    # * out_channels 可以继续增加，比如从 16 到 32，再到 64。
    # * 卷积核大小的选择（2x2, 3x3, 5x5）也会影响特征的提取粒度。
    # * 池化层（MaxPool 或 AvgPool）的选择及其参数也会影响特征图的下采样方式。
    ############################################################################################################

    #practice question:
    #(4)显示不同的输入（例如毛衣和外套）时LeNet的第一层和第二层的激活值

    # Chapter7 现代卷积NN
    
    # AlexNet
    import torch
    from torch import nn

    net = nn.Sequential(
        # 这里定义一个11X11的更大窗口来捕获对象
        # 同时，步幅为4，以减少输出的高度和宽度
        # 另外 输出通道远大于LeNet
        nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        #减少卷积窗口 使用填充为2来使得输入与输出的高度和宽度一致 且增大输出的通道数
        nn.Conv2d(96, 256, kernel_size=5, padding=2),nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        #使用3个连续的卷积层和较小的卷积窗口
        #除了最后的卷积层 输出通道数进一步增加
        #在前两个卷积层之后 汇聚层不用于减少输入的高度和宽度
        nn.Conv2d(256, 384, kernel_size=3, padding=1),nn.ReLU(),
        nn.Conv2d(384, 384, kernel_size=3, padding=1),nn.ReLU(),
        nn.Conv2d(384, 256, kernel_size=3, padding=1),nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Flatten(),
        #这里 全连接层的数量是LeNet的好几倍
        nn.Linear(6400, 4096), nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 4096), nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 10))
    X = torch.randn(1,1,224,224)
    for layer in net:
        X=layer(X)
        print(layer.__class__.__name__,'output shape:\t',X.shape)

    batch_size = 128
    train_iter, test_iter = load_data_fashion_minist(batch_size, resize=224)

    lr, num_epochs = 0.01, 10
    #train_ch6(net, train_iter, test_iter, num_epochs, lr, try_gpu)
    #train_ch6(net, train_iter, test_iter, num_epochs, lr, torch.device('mps:0'))
    #plt.show()


    #VGG
    #define a VGG function
    def vgg_block(num_convs, in_channels, out_channels):
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels,
                                    kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)   

    conv_arch = ((1,64),(1,128),(2,256),(2,512),(2,512))
    #define VGG-11
    def vgg(conv_arch):
        conv_blks = []
        in_channels = 1
        #卷积层部分
        for (num_convs, out_channels) in conv_arch:
            conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels

        return nn.Sequential(
            *conv_blks, nn.Flatten(),
            #全连接层部分
            nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 10))

    net = vgg(conv_arch)

    #查看每块的形状
    X = torch.randn(1,1,224,224)
    for blk in net:
        X=blk(X)
        print(blk.__class__.__name__,'output shape:\t',X.shape)

    
    #training
    ratio = 4
    small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
    net = vgg(small_conv_arch)

    batch_size = 128
    train_iter, test_iter = load_data_fashion_minist(batch_size, resize=224)

    lr, num_epochs = 0.05, 10
    #train_ch6(net, train_iter, test_iter, num_epochs, lr, torch.device('mps'))

    #NiN
    def nin_block(in_channels, out_channels, kernel_size, strides, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels, kernel_size=1), nn.ReLU(),
            nn.Conv2d(out_channels,out_channels, kernel_size=1), nn.ReLU())
    
    net = nn.Sequential(
        nin_block(1, 96, kernel_size=11, strides=4, padding=0),
        nn.MaxPool2d(3, stride=2),
        nin_block(96, 256, kernel_size=5, strides=1, padding=2),
        nn.MaxPool2d(3, stride=2),
        nin_block(256, 384, kernel_size=3, strides=1, padding=1),
        nn.MaxPool2d(3, stride=2),
        nn.Dropout(0.5),
        # 标签类别数是10
        nin_block(384, 10, kernel_size=3, strides=1, padding=1),
        nn.AdaptiveAvgPool2d((1,1)),
        #将4维输出转换成二维输出
        nn.Flatten())

    X = torch.randn(1,1,224,224)
    for layer in net:
        X=layer(X)
        print(layer.__class__.__name__,'output shape:\t',X.shape)

    lr, num_epochs, batch_size = 0.05, 10, 128
    train_iter, test_iter = load_data_fashion_minist(batch_size, resize=224)
    #train_ch6(net, train_iter, test_iter, num_epochs, lr, torch.device('mps'))

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # NiN使用一个卷积层和多个1X1的卷积层组成的块。该块可以在卷积NN中使用，以允许更多的每像素的非线形。
    # NiN去除了容易造成过拟合的全连接层，将它们替换成全局平均汇聚层（即在所有位置上进行求和）该汇聚层的通道数量为所需的输出数量
    # 去除全连接层可以减少过拟合，同时显著减少NiN的参数数量
    # NiN的设计影响了后续许多卷积NN的设计。
    # 1x1卷积之所以能减少通道数，是因为它在操作上具有一种“跨通道（cross-channel）”或“特征融合（feature fusion）”的能力，
    # 同时不改变特征图的空间维度（高度和宽度）。
    # 如何设计款通道的1X1的卷积层？
    # 1x1 卷积核的运作方式：
    # 一个1x1卷积核的尺寸是 1X1XC_{in}。这意味着这个卷积核在空间上只覆盖一个像素点，但在深度（通道）方向上，它会覆盖所有 C_{in} 
    # 个输入通道。
    # 当这个1x1卷积核在输入特征图上滑动时，它会：
    # 在每个空间位置上： 对于输入特征图的每一个 HXW 位置，1x1卷积核会“看到”该位置上所有 C_{in} 个通道的像素值。
    # 进行加权求和： 这个1x1卷积核的权重会与这 C_{in} 个输入通道的像素值进行逐元素相乘，然后求和，再加上一个偏置项。
    # 这个操作本质上是对当前像素点在所有输入通道上的一个线性组合。
    # 生成一个输出值： 这个线性组合的结果就成为输出特征图在该空间位置上的一个像素值（我的理解：特征图的一个维度）。
    # 如何减少通道数：
    # 如果你使用多个1x1卷积核，比如 C_{out} 个1x1卷积核，那么每个卷积核都会独立地执行上述的加权求和操作。
    # 第一个1x1卷积核会产生输出特征图的第一个通道。
    # 第二个1x1卷积核会产生输出特征图的第二个通道。
    # ...
    # 第 C_{out} 个1x1卷积核会产生输出特征图的第 C_{out} 个通道。
    # 最终，通过 C_{out} 个1x1卷积核的并行操作，你将得到一个输出特征图，其维度是 HXWXC_{out}。
    # 核心思想：
    # 1x1卷积可以看作是在通道维度上进行的全连接操作（或者说，它在每个空间位置上执行一个小的全连接层）。
    # 每个1x1卷积核都学习如何将所有输入通道的信息聚合成一个新的输出通道。通过控制1x1卷积核的数量，你就可以精确地控制输出的通道数C_OUT。
    # 总结1x1卷积减少通道数的优点：
    # 维度缩减 (Dimensionality Reduction)： 最直接的作用就是减少通道数，从而降低模型的参数量和计算量，尤其是在通道数非常多的情况下。
    # 增加非线性 (Add Non-linearity)： 1x1卷积之后通常会跟着一个激活函数（如ReLU），这为网络引入了额外的非线性，增加了模型的表达能力。
    # 创建“瓶颈层” (Bottleneck Layer)： 在像Inception网络这样的架构中，1x1卷积常被用作“瓶颈层”，
    # 在进行计算量更大的3x3或5x5卷积之前先减少通道数，以降低计算成本。
    # 跨通道信息融合 (Cross-channel information fusion)： 它可以学习到不同通道之间的关系，并将它们有效地组合起来，
    # 从而提取更高级别的特征。
    # 所以，与简单地丢弃通道不同，1x1卷积是通过学习的方式将输入通道的信息映射到更少的输出通道中，尽可能保留有用的信息。
    # 全局平均汇聚层：
    # 在NiN（Network in Network）网络结构中，nn.AdaptiveAvgPool2d((1, 1)) 这一层扮演着非常关键的角色，
    # 它实现了一个 全局平均池化（Global Average Pooling, GAP） 的操作。
    # 什么是全局平均池化 (Global Average Pooling, GAP)？
    # 传统池化 vs. 全局池化：
    # 传统池化（如Max Pooling或Average Pooling） 是在特征图的局部区域内进行操作，并产生一个缩小尺寸的特征图。
    # 例如，一个 2x2 的最大池化会把 2x2 区域的最大值作为输出。
    # 全局平均池化则不同，它不是在局部区域操作，而是对每个通道的整个特征图计算平均值，并将这个平均值作为该通道的输出。
    # nn.AdaptiveAvgPool2d((1, 1)) 具体做什么？
    # AdaptiveAvgPool2d 是一个自适应的平均池化层。这意味着你不需要指定池化核的大小和步长，而是直接指定输出特征图的尺寸。
    # 当你设置 (1, 1) 时，它会确保每个通道的输出特征图都变成 1x1 的尺寸。
    # 这实际上就意味着：对于输入特征图的每个通道，它会计算该通道上所有像素的平均值，然后将这个平均值作为该通道的唯一输出值。
    # 举例说明：
    # 假设 nn.AdaptiveAvgPool2d((1, 1)) 之前的特征图是 B x C x H x W (批量大小 x 通道数 x 高度 x 宽度)。
    # 如果输入是 384x5x5（假设批量大小为1），那么对于这384个通道中的每一个，AdaptiveAvgPool2d((1, 1)) 会将 5x5 = 25 
    # 个像素值求平均，得到一个值。
    # 最终，输出会变成 BxCx1x1。
    # 在你的NiN网络中，经过 nn.AdaptiveAvgPool2d((1, 1)) 之后，如果之前的 nn.Conv2d(384, 10, ...) 的输出是
    #  10xH'x W' (其中 H' 和 W' 是某个空间维度)，那么 AdaptiveAvgPool2d((1, 1)) 会把这个 10xH'xW' 的特征图转换
    # 为 10x1x1 的形式。
    # 为什么NiN网络使用全局平均池化？
    # 图片文字中已经提到NiN网络的最后一个是全局平均池化层，并且解释了它的一个优点：“它显著减少了模型所需参数的数量。”
    # 除了减少参数数量，全局平均池化还有以下几个优点，这些是NiN网络提出的核心思想：
    # 减少参数量：
    # 替代全连接层： 传统的CNN在卷积层之后通常会接一个或多个全连接层（nn.Linear）。全连接层连接到所有前一层的神经元，
    # 因此会引入大量的参数。
    # NiN的创新： NiN用全局平均池化层取代了传统的全连接层。在NiN的架构中，最后一个卷积层
    # （或一系列卷积层，如你代码中的nn.Conv2d(384, 10, ...)）的输出通道数直接设置为等于分类任务的类别数（例如，这里的10）。
    # 消除参数： AdaptiveAvgPool2d((1, 1)) 层本身没有可训练的参数（它只进行平均操作）。
    # 当它将特征图缩减到 1X1 后，这个 1X1Xtext{类别数} 的输出可以直接输入到损失函数进行计算，而不需要额外的全连接层。
    # 这极大地减少了模型参数，有助于防止过拟合。
    # 提高模型的鲁棒性：
    # 空间信息保持： 全局平均池化可以看作是对整个特征图的聚合。与局部池化相比，它更好地保留了每个通道的整体空间信息。
    # 减少过拟合： 由于参数量显著减少，模型更不容易过拟合训练数据。
    # 对空间变换更鲁棒：
    # 因为是取整个特征图的平均值，所以特征图中某个小区域的平移或变形对最终的平均值影响相对较小，这使得模型对输入图像的空间变换更具鲁棒性。
    # 直接作为分类层的输入：
    # 如果最终的分类任务有N个类别，那么在进行GAP操作之前的卷积层，其输出通道数应该正好是N。经过GAP后，
    # 我们会得到一个 1x1xN 的特征向量，这个向量的每个值可以直接看作是对应类别的“置信度”或“分数”，可以直接输入到Softmax层
    # （如果需要概率输出）或损失函数。
    # 总结：
    # nn.AdaptiveAvgPool2d((1, 1)) 在NiN网络中是实现全局平均池化的关键，其主要目的是用无参数的全局平均操作替代传统全连接层，
    # 从而：
    # 大幅减少模型参数，降低过拟合风险。
    # 提高模型的鲁棒性。
    # 将特征图直接转换为对应类别数的输出，简化分类层设计。
    # 这是NiN网络的一个核心创新点，对后续的深度学习网络设计产生了深远影响。

    # GoogLeNet
    # Inception Block
    # 解决多大的卷积核是合适的，结论：有时候使用大小同的卷积核组合是有利的。为什么呢？首先我们考虑下滤波器的组合，我们
    # 可以利用各种滤波器的尺寸探索图像。这意味着不同尺寸的滤波器可以有效地识别不同范围的图像细节。同时，我们可以为不同滤波器分配不同
    # 数量的参数
    # 1x1 3x3 5x5 并行 从公不同空间spatial大小中提取信息 中间两条路径在输入前加入1X1卷积以减少通道数 从而降低模型复杂度
    # 这4条利用合适的padding以使输入和输出的高度/宽度一致
    # 最后我们将每条路径的输出在通道维度上合并 并构成inception block的输出
    # 在Inception块中 通常调整的超参数是每层输出通道数
    from torch.nn import functional as F

    class Inception(nn.Module):
        #c1--c4是每条路径的输出通道数
        # def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        #     super(Inception, self).__init__(**kwargs)
        #     #path 1 single 1x1 conv layer
        #     self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        #     #path 2 1x1 conv layer and than 3x3 conv layer
        #     self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        #     self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        #     #path 3 1x1 conv layer and than 5x5 conv layer
        #     self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        #     self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        #     #path 4 3x3 pooling layer and than 1x1 conv layer
        #     self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        #     self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

        # 加入BN代码
        def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
            super(Inception, self).__init__(**kwargs)
            #path 1 single 1x1 conv layer
            self.path_p1 = nn.Sequential(
                nn.Conv2d(in_channels, c1, kernel_size=1),
                nn.BatchNorm2d(c1),
                nn.ReLU(inplace=True)
            )
            #path 2 1x1 conv layer and than 3x3 conv layer
            self.path_p2 = nn.Sequential(
                nn.Conv2d(in_channels, c2[0], kernel_size=1),
                nn.BatchNorm2d(c2[0]),
                nn.ReLU(inplace=True),
                nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(c2[1]),
                nn.ReLU(inplace=True)
            )
            #path 3 1x1 conv layer and than 5x5 conv layer
            self.path_p3 = nn.Sequential(
                nn.Conv2d(in_channels, c3[0], kernel_size=1),
                nn.BatchNorm2d(c3[0]),
                nn.ReLU(inplace=True),
                nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2),
                nn.BatchNorm2d(c3[1]),
                nn.ReLU(inplace=True)
            )
            #path 4 3x3 pooling layer and than 1x1 conv layer
            self.path_p4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels, c4, kernel_size=1),
                nn.BatchNorm2d(c4),
                nn.ReLU(inplace=True)
            )


        def forward(self, x):
            # p1 = F.relu(self.p1_1(x))
            # p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
            # p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
            # p4 = F.relu(self.p4_2(self.p4_1(x)))
            p1 = self.path_p1(x)
            p2 = self.path_p2(x)
            p3 = self.path_p3(x)
            p4 = self.path_p4(x)
            # 在通道维度上连接输出 cat the connect out in channel dimentionary
            return torch.cat((p1,p2,p3,p4), dim=1)

        
    # GoogLeNet共用9个Inception块和全局平均汇聚层的叠叠来生成其估计值。inception块之间的最大汇聚层可以降低维度
    # Inception块的组合从VGG继承 全局平均汇聚层避免了在最后使用全连接层
    # No1 module use 64 channels 7x7 conv
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    # No2 module use 2 conv 
    b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.Conv2d(64, 192, kernel_size=3, padding=1),
                        nn.BatchNorm2d(192),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    # No3 module use 2 Inceptions
    b3 = nn.Sequential(Inception(192, 64, (96,128), (16,32), 32),
                        Inception(256, 128, (128,192), (32,96), 64),
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    # No4 module use 5 Inception and a 3x3 maxpooling layer
    b4 = nn.Sequential(Inception(480, 192, (96,208), (16,48), 64),
                        Inception(512, 160, (112,224), (24,64), 64),
                        Inception(512, 128, (128,256), (24,64), 64),
                        Inception(512, 112, (144,288), (32,64), 64),
                        Inception(528, 256, (160,320), (32,128), 128),
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    # N05 Modules
    b5 = nn.Sequential(Inception(832, 256, (160,320), (32,128), 128),
                        Inception(832, 384, (192,384), (48,128), 128),
                        nn.AdaptiveAvgPool2d((1,1)),
                        nn.Flatten())

    net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024,10))

    X = torch.rand(size=(1,1,96,96))
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__,'output shape:\t',X.shape)

    
    lr, num_epochs, batch_size = 0.9, 10, 128
    train_iter, test_iter = load_data_fashion_minist(batch_size, resize=96)
    #train_ch6(net, train_iter, test_iter, num_epochs, lr, torch.device('mps'))
    #plt.show()

    # BN Batch Normalization
    # 训练神经网络是比较困难的 特别是在较短时间内让它们收敛更加棘手。 BN是一种流行的技术 可以持续
    # 加深层网络的收敛 结合残差块 BN使得我们可以训练100层以上的网络
    # 训练一万层以上的网络（论文）

    # 从零实现BN函数
    # def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    #     # 通过is_grad_enable方法来判断当前的模式是训练模式还是预测模式
    #     if not torch.is_grad_enabled():
    #         # 如果是预测模式 直接使用传入的移动平均所得的均值和方差
    #         X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    #     else:
    #         assert len(X.shape) in (2,4)
    #         if len(X.shape) == 2:
    #             # 使用全连接层的情况 计算特征维上的均值和方差
    #             mean = X.mean(dim=0)
    #             var = ((X - mean) ** 2).mean(dim=0)
    #         else:
    #             # 使用二维卷积层的情况， 计算通道维上（axis=1）的均值和方差
    #             mean = X.mean(dim=(0, 2, 3), keepdim=True)
    #             var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
    #         # 训练模式下用当前的均值和方差做标准化
    #         X_hat = (X - mean) / torch.sqrt(var + eps)
    #         # update moving_mean and moving_var
    #         moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
    #         moving_var = momentum * moving_var + (1.0 - momentum) * var

    #     Y = gamma * Y_hat + beta # 缩放和位移
    #     return Y, moving_mean, moving_var
    
    # # class BN
    # class BatchNorm(nn.Module):
    #     # num_features全连接层的输出数量或者卷积层的输出通道数
    #     # num_dims 2 全连接层 4 卷积层
    #     def __init__(self, num_features, num_dims):
    #         super().__init__()
    #         if num_dims == 2:
    #             shape = (1, num_features)
    #         else:
    #             shape = (1, num_features, 1, 1)
    #         #参与求梯度和迭代的拉伸参数和偏移参数 其分别初始化为1和0
    #         self.gamma = nn.Parameter(torch.ones(shape))
    #         self.beta = nn.Parameter(torch.zeros(shape))
    #         #非模型参数变量初始化为0和1
    #         self.moving_mean = torch.zeros(shape)
    #         self.moving_var = torch.ones(shape)

    #     def forward(self, X):
    #         # 如果X不在内存上 将moving_mean and moving_var copy to X所在在显存上
    #         if self.moving_mean.device != X.device:
    #             self.moving_mean = self.moving_mean.to(torch.device('mps'))
    #             self.moving_var = self.moving_var.to(torch.device('mps'))

    #         #保存更新过的moving mean和moving var
    #         Y, self.moving_mean, self.moving_var = batch_norm(
    #             X, self.gamma, self.beta, self.moving_mean,
    #             self.moving_var, eps=1e-5, momentum=0.9)
    #         return Y
        
    
    # #LeNet that using BN
    # net = nn.Sequential(
    #     nn.Conv2d(1,6,kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
    #     nn.AvgPool2d(kernel_size=2, stride=2),
    #     nn.Conv2d(6,16,kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
    #     nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    #     nn.Linear(16*4*4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
    #     nn.Linear(120,84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
    #     nn.Linear(84,10)
    # )

    # net.to(torch.device('mps'))
    # lr, num_epochs, batch_size = 1.0, 10, 256
    # train_iter, test_iter = load_data_fashion_minist(batch_size)
    # #train_ch6(net, train_iter, test_iter, num_epochs, lr, torch.device('mps'))
    # #plt.show()
    
    import torch
    from torch import nn
    net = nn.Sequential(
        nn.Conv2d(1,6,kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6,16,kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
        nn.Linear(16*4*4, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
        nn.Linear(120,84), nn.BatchNorm1d(84), nn.Sigmoid(),
        nn.Linear(84,10)

    )
    lr, num_epochs, batch_size = 1.0, 10, 256
    train_iter, test_iter = load_data_fashion_minist(batch_size)
    train_ch6(net, train_iter, test_iter, num_epochs, lr, torch.device('mps'))
    plt.show()
    
    net = nn.Sequential(
        nn.Conv2d(1,6,kernel_size=5), nn.BatchNorm2d(6),nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6,16,kernel_size=5), nn.BatchNorm2d(16),nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
        nn.Linear(16*4*4, 120), nn.BatchNorm1d(120),nn.Sigmoid(),
        nn.Linear(120,84), nn.BatchNorm1d(84),nn.Sigmoid(),
        nn.Linear(84,10))
    lr, num_epochs, batch_size = 0.9, 10, 256
    train_iter, test_iter = load_data_fashion_minist(batch_size)
    train_ch6(net, train_iter, test_iter, num_epochs, lr, torch.device('mps'))
    plt.show()
    

    # ResNet
    # 核心思想：残差连接
    # 传统神经网络每一层都在尝试直接学习输出（比如预测图片是猫还是狗）。
    # ResNet不直接学输出，而是学输入和输出的差值（残差）。
    # 举个例子：假如输入是5，目标输出是6，传统网络直接学6，而ResNet学的是“6-5=1”（残差）。
    # 具体实现：ResNet在网络层之间加了一条“捷径”（shortcut），直接把输入加到输出上，公式是：
    # 输出 = 网络层计算的结果 + 输入
    # 这样，即使网络层学得不好（输出接近0），输入也能直接传到下一层，保证信息不丢失。
    
    import torch
    from torch.nn import functional as F

    # 实现残差块
    class Residual(nn.Module):
        def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
            super().__init__()
            self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
            self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

            if use_1x1conv:
                self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
            
            else:
                self.conv3 = None
            self.bn1 = nn.BatchNorm2d(num_channels)
            self.bn2 = nn.BatchNorm2d(num_channels)

        def forward(self, X):
            Y = F.relu(self.bn1(self.conv1(X)))
            Y = self.bn2(self.conv2(Y))
            if self.conv3:#判断最好放在init函数里 因为GPU不适合做if判断处理
                X = self.conv3(Y)
            # 确保X Y的维度匹配
            if X.shape != Y.shape:
                raise ValueError(f"Dimension mismatch: X={X.shape}, Y={Y.shape}")
            Y += X
            return F.relu(Y)

    # check input shape and ouput shape
    blk = Residual(3,3)
    X = torch.rand(4, 3, 6, 6)
    Y = blk(X)
    print(Y.shape)

    blk = Residual(3, 3, use_1x1conv=True, strides=2)
    X = torch.rand(4, 3, 6, 6)
    Y = blk(X)
    print(Y.shape)


    
    def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(num_channels, num_channels))
        return nn.Sequential(*blk)#blk
    b1 = nn.Sequential(nn.Conv2d(1,64, kernel_size=7, stride=2, padding=3),
                    nn.BatchNorm2d(64), nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    b2 = nn.Sequential(*resnet_block(64,64,2, first_block=True))
    b3 = nn.Sequential(*resnet_block(64,128,2))
    b4 = nn.Sequential(*resnet_block(128,256,2))
    b5 = nn.Sequential(*resnet_block(256,512,2))

    net = nn.Sequential(b1,b2,b3,b4,b5,
                        nn.AdaptiveAvgPool2d((1,1)),
                        nn.Flatten(), nn.Linear(512,10))

    X = torch.rand(size=(1,1,224,224))
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__,'output shape:\t',X.shape)
    

    
    # DenseNet
    def conv_block(input_channels, num_channels):
        return nn.Sequential(
            nn.BatchNorm2d(input_channels), nn.ReLU(),
            nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))
    
    class DenseBlock(nn.Module):
        def __init__(self, num_convs, input_channels, num_channels):
            super().__init__()
            layer = []
            for i in range(num_convs):
                layer.append(conv_block(num_channels * i + input_channels, num_channels))
            self.net = nn.Sequential(*layer)

        def forward(self, X):
            for blk in self.net:
                Y = blk(X)
                #连接通道上每个卷积块的输入和输出
                X = torch.cat((X,Y), dim=1)
            return X

    blk = DenseBlock(2,3,10)
    X = torch.randn(4, 3, 8, 8)
    Y = blk(X)
    print(Y.shape)

    def transition_block(input_channels, num_channels):
        return nn.Sequential(
            nn.BatchNorm2d(input_channels), nn.ReLU(),
            nn.Conv2d(input_channels, num_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    blk = transition_block(23,10)
    print(blk(Y).shape)

    # 构建DenseNet模型
    b1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    )

    #num_channels为当前的通道数
    num_channels, growth_rate = 64, 32
    num_convs_in_dense_blocks = [4,4,4,4]
    blks = []
    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        blks.append(DenseBlock(num_convs, num_channels, growth_rate))
        # 上一个稠密块输出的通道数
        num_channels += num_convs * growth_rate
        # 在稠密块之间添加一个过渡层 使通道数减半
        if i != len(num_convs_in_dense_blocks) - 1:
            blks.append(transition_block(num_channels, num_channels // 2))
            num_channels = num_channels // 2
    
    net = nn.Sequential(
        b1, *blks,
        nn.BatchNorm2d(num_channels), nn.ReLU(),
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(num_channels, 10)
    )

    # train the model
    lr, num_epochs, batch_size = 0.1, 10, 256
    train_iter, test_iter = load_data_fashion_minist(batch_size, resize=96)
    #train_ch6(net, train_iter, test_iter, num_epochs, lr, torch.device('mps'))
    #plt.show()

    # chapter 8 RNN

    # 构建简单的循环神经网络进行简单的预测
    # sin function
    import torch
    from torch import nn
    import matplotlib.pyplot as plt

    # T = 1000 # 定义时间序列长度为1000
    # time = torch.arange(1, T + 1, dtype=torch.float32)
    # x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
    # plt.figure(figsize=(6, 3))
    # plt.plot(time, x)
    # plt.xlabel('time')
    # plt.ylabel('x')
    # plt.xlim(1, 1000)
    # plt.grid(True)
    #plt.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
    #plt.show()

    # 这部分代码准备了用于训练的数据集 将时间序列数据转换成适合RNN训练的输入-输出对
    # 定义tau步长 这通常是RNN的“记忆”长度或者时间窗口大小 对于每个时间步 模型会考虑tau个过去观测值来预测下一个值
    # tau=4 样本数就是996
    #tau = 4 
    # 初始化了一个features张量 它将存储模型的输入 每一行代表一个样本 包含tau个历史观测值
    #features = torch.zeros((T - tau, tau))#T - tau是行数 也即样本数量 tau为列数 即每个样本的特征数量
    #因为我们使用一个大小为tau的滑动窗口来从原始时间序列中提取特征和标签 所以会有tau个数据点无法形成完整的tau长度的输入序列

    # 创建了一个滑动窗口 对于每个i从0-tau-1 它将x的一个切片填充到features的i列
    # for i in range(tau):
    #     features[:, i] = x[i:T - tau + i]
    
    # label的第j行将是x[j + tau]
    # labels = x[tau:].reshape((-1,1))#x[tau:]起始索引到x的末尾 reshape(-1,1)成一列

    # batch_size, n_train = 16, 600
    # train_iter = load_array((features[:n_train], labels[:n_train]),
    #                         batch_size, is_train=True)

    # def init_weights(m):
    #     if type(m) == nn.Linear:
    #         nn.init.xavier_uniform_(m.weight)

    # def get_net():
    #     net = nn.Sequential(nn.Linear(4,10),#4即tau 我们的样本过去的观测数量
    #                         nn.ReLU(),
    #                         nn.Linear(10,1))
    #     net.apply(init_weights)
    #     return net
    
    # loss
    # loss = nn.MSELoss(reduction='none')#none表不对样本每个样本的损失进行平均或求和 而是返回每个样本的损失值 这在某种情况下
    #可能要手动处理损失

    # def train(net, train_iter, loss, epochs, lr):
    #     trainer = torch.optim.Adam(net.parameters(), lr)#取模型所有可训练参数
    #     for epoch in range(epochs):
    #         for X, y in train_iter:
    #             trainer.zero_grad()
    #             l = loss(net(X), y)
    #             l.sum().backward()#sum是通常在none情况下将批次中所有的损失求和然后反向传播
    #             trainer.step()#更新模型的参数 优化器根据计算出来的梯度来更新w and b
    #         print(f'epoch {epoch + 1},'
    #               f'loss: {evaluate_loss(net, train_iter, loss):f}')

    # net = get_net()
    # train(net, train_iter, loss, 5 , 0.01)
    # plt.show()

    # 单步预测
    #onestep_preds = net(features)
    # detach是分离张量 使其不再参与梯度计算 numpy转换类型
    # plot([time, time[tau:]],
    #          [x.detach().numpy(), onestep_preds.detach().numpy()], 'time',
    #          'x', legend=['data', '1-step preds'], xlim=[1,1000],
    #          figsize=(6,3))
    # plt.show()

    # 多步预测
    # 这是指模型在进行第一次预测后 将其自身的预测结果作为后续预测的输入 不断进行预测 这更接近真实的世界的场景 因为未来的真实值是未知的
    #multistep_preds = torch.zeros(T)#初始化一个o张量T来存储多步预测结果
    # 多步预测一般使用训练集之外的真实值作为初始输入 这里 它将x中n_train - n_train+tau-1的真实值复制到multistep_preds中作为多
    # 步预测的“种子”[: n_train + tau]：索引从n_train开始 到n_train+tau-1结束 不包括n_train+tau
    # multistep_preds[: n_train + tau] = x[: n_train + tau]
    # for i in range(n_train + tau, T):
    #     multistep_preds[i] = net(
    #         multistep_preds[i - tau:i].reshape((1,-1))#reshape((1,-1))重塑成1行 因为net期望输入是（batch_size,features_num)的形状
    #     )
    # plot([time, time[tau:], time[n_train + tau:]],
    #      [x.detach().numpy(), onestep_preds.detach().numpy(),
    #      multistep_preds[n_train + tau:].detach().numpy()], 'time',
    #      'x', legend=['data', '1-step preds', 'multistep preds'],
    #      xlim=[1,1000], figsize=(6,3))

    # plt.show()

    # k步预测
    # max_steps = 64
    # # 可用于k步预测的样本数  tau + max_steps表示每个样本包含tau个输入和max_steps个未来预测
    # features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
    # # 列i（i<tau）是来自x的观测 其时间步从（i+1）到（i+T-tau-max_steps+1)
    # # 填充features的前tau列 这些事来自原始数据x的真实观测值 作为每个预测序列的起始输入
    # for i in range(tau):
    #     features[:, i] = x[i: i + T - tau - max_steps + 1]

    # 对于每个i从tau到tau+max_steps-1（即从从第一个预测步到第max_steps个预测步）
    # features[:, i - tau:i]去除feature中当前位置之前的tau个数据点作为输入 注意这些数据点可能包含了之前模型的预测值
    # for i in range(tau, tau + max_steps):
    #     #features[:, i]将预测值填充到features的对应的列中 这样features不仅包含了输入 也包含了未来max steps步的预测值
    #     features[:, i] = net(features[:, i - tau:i]).reshape(-1)#展平 

    # steps = (1,4,16)
    # plot([time[tau + i -1 : T - max_steps + i] for i in steps],
    #      [features[:,(tau + i -1)].detach().numpy() for i in steps], 'time', 'x',
    #      figsize=(6,3))

    #plt.show()


    # 8.2 文本预处理
    # 导入Python标准库中的 collections 模块。这个模块提供了一些特殊的数据结构，例如 Counter、defaultdict 等，
    # 它们在处理数据（尤其是文本数据）时非常有用。在这段代码中，虽然没有直接使用 collections 里的具体功能，
    # 但它通常会和文本处理一起出现，比如用于统计词频。
    import collections
    # 导入Python标准库中的 re 模块。re 代表 Regular Expression (正则表达式)。这个模块用于执行正则表达式操作，
    # 比如搜索、替换、分割字符串等。在文本预处理中，正则表达式非常常用，用来清洗文本，例如移除标点符号、数字等。
    import re
    #from d2l import torch as d2l

    # DATA_HUB['kaggle_house_train'] = ( #@save
    #     DATA_URL + 'kaggle_house_pred_train.csv',
    #     '585e9cc93e70b39160e7921475f9bcd7d31219ce')
    # 它为键 'time_machine' 分配了一个元组（tuple）作为值。
    # 元组的第一个元素是数据的下载URL
    # 元组的第二个元素 '090b5e7e70c295757f55df93cb0a180b9691891a' 看起来是一个文件的哈希值（通常是SHA-1或MD5）。
    # 这个哈希值的目的是验证下载文件的完整性和正确性。下载完成后，程序会计算下载文件的哈希值，并与这个预设的哈希值进行比较，
    # 如果一致，说明文件没有被损坏或篡改。
    DATA_HUB['time_machine'] = (DATA_URL + 'timemachine.txt',
                                      '090b5e7e70c295757f55df93cb0a180b9691891a')
    
    def read_time_machine():
        """将时间机器数据集加载到文本行的列表中"""
        with open(download('time_machine'), 'r') as f:
            lines = f.readlines()
        #列表推导式 用于对lines列表中每一行进行文本预处理 并返回处理后的新列表
        #这是列表推导式（list comprehension），用于对 lines 列表中的每一行进行文本预处理，并返回处理后的新列表
        return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]#strip()移除字符串开头和结尾的空白

    lines = read_time_machine()
    print(f'# 文本总行数:{len(lines)}')
    print(lines[0])
    print(lines[10])

    # 词元化 tokenize
    # 该函数将文本行lines作为输入 列表中的每个元素是一个文本序列如一个文本行 每个文本行序列
    # 又被拆分成一个词元列表token 词元是文本的基本单位 最后 返回一个由词元列表组成的列表 其中每个词元都是一个字符串
    def tokenize(lines, token='word'):
        """将文本拆分成单词或者字符词元"""
        if token == 'word':
            return [line.split() for line in lines]
        elif token == 'char':
            return [list(line) for line in lines]
        else:
            print('error: unknown token type: ' + token)
    
    tokens = tokenize(lines)
    for i in range(11):
        print(tokens[i])

    
    # 词表
    # number(model) ====== string mapping

    class Vocab:
        """"文本词表"""
        def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
            if tokens is None:
                tokens = []
            # reserved_tokens just like:<pad>用于填充 <bos>用于句首 <eos>用于句尾
            if reserved_tokens is None:
                reserved_tokens = []
            #按出现频率排序
            #构建词元频率列表并排序
            counter = count_corpus(tokens)#调用函数统计tokens中每个词元的频率 并获取一个包含（词元、频率）元组的列表
            #key=lambda x: x[1]指定排序的键是元组的第二个元素 reverse=True表示按降序排列
            self._token_freqs = sorted(counter.items(),key=lambda x: x[1],
                                       reverse=True)
            #未知词元索引为0
            #初始化idx_to_token列表 这是一个索引到词元的映射
            #unk一个特殊的未知词元 在处理文本时 如遇到不在词汇表中的词元就会用到这个来表示 它通常被赋予索引0
            #reserved_tokens也会被添加到这个列表 在unk之后
            self.idx_to_token = ['<unk>'] + reserved_tokens
            #这是一个词元到索引的映射
            #enumerate(self.idx_to_token)生成索引词元对
            self.token_to_idx = {token: idx
                                 for idx, token in enumerate(self.idx_to_token)}
            
            for token, freq in self._token_freqs:
                if freq < min_freq:
                    break
                if token not in self.token_to_idx:
                    self.idx_to_token.append(token)
                    self.token_to_idx[token] = len(self.idx_to_token) - 1#最后一个元素的索引 因为有顺序
        #这个是python的特殊方法 使得vocab对象可以使用len函数
        def __len__(self):
            return len(self.idx_to_token)
        #这个是python的特殊方法 使得Vocab对象可以使用【】进行索引访问 如vocan['hello']
        def __getitem__(self, tokens):
            if not isinstance(tokens, (list, tuple)):#如果传入的tokens不是list或者tuple 即是单个词元字符串
                return self.token_to_idx.get(tokens, self.unk)#从token_to_idx中取token对应的索引 如果token不存在 则返回self.unk
            #如果传入的tokens是list/tuple，则使用列表推导式递归调用__getitem__ 获得每个token的索引 并返回一个索引列表
            return [self.__getitem__(token) for token in tokens]

        #将词元索引转换为转换为词元字符串
        def to_tokens(self, indices):
            if not isinstance(indices, (list, tuple)):
                return self.idx_to_token[indices]
            return [self.idx_to_token[index] for index in indices]
        #一个属性装饰器 使得unk可以像访问属性一样被调用(vocab.unk)而不是像方法一样（vacab.unk()) 它返回未知词元的索引
        @property
        def unk(self):
            return 0
        #一个属性装饰器 返回self._token_freqs 它返回构建词汇表时使用的词元及其频率的列表
        @property
        def token_freqs(self):
            return self._token_freqs

    def count_corpus(tokens):
        """统计词元出现的频率"""
        # 这里的tokens是一维列表或二维列表
        if len(tokens) == 0 or isinstance(tokens[0], list):
            # 将词元列表展平成一个列表
            # 这是一个嵌套的列表推导式，用于将所有词元展平（flatten）并转换为它们的整数索引。
            # * for line in tokens: 遍历 tokens 列表中的每个子列表（即每行文本的词元列表）。
            # * for token in line: 遍历当前行中的每个词元（字符）。
            # * vocab[token]: 使用 vocab 对象将当前的词元（字符）映射到它在词汇表中的唯一整数索引。
            # Vocab 类很可能实现了 __getitem__ 方法，使得可以通过 vocab[token] 的方式获取词元的索引。
            # * 最终结果： corpus 将是一个长长的整数列表，其中每个整数代表一个字符在词汇表中的索引。
            # 这是神经网络能够处理的数值形式的文本数据。
            # 确保 collections.Counter 能够处理一维的词元序列。
            tokens = [token for line in tokens for token in line]
        return collections.Counter(tokens)

    # print 前几个高频词
    vocab = Vocab(tokens)
    print(list(vocab.token_to_idx.items())[:10])

    # 将每一个文本行转换成一个数字索引列表
    print('++++++++++++++++++++++++++')
    for i in [0, 10]:
        print('文本:', tokens[i])
        print('索引:', vocab[tokens[i]])

    # 所有功能打包
    def load_corpus_time_machine(max_tokens=-1):
        """返回时光机器数据集的词元索引列表和词表"""
        lines = read_time_machine()#调用 read_time_machine() 获取原始文本行
        tokens = tokenize(lines, 'char')#调用 tokenize 函数进行词元化，这里按字符进行
        vocab = Vocab(tokens)# 使用词元列表构建词汇表
        #因为时光机器中每一个文本行不一定是一个句子
        #或一个段落 所以将所有的文本行展平到一个列表中
        # 将词元列表展平成一个列表
        # 这是一个嵌套的列表推导式，用于将所有词元展平（flatten）并转换为它们的整数索引。
        # * for line in tokens: 遍历 tokens 列表中的每个子列表（即每行文本的词元列表）。
        # * for token in line: 遍历当前行中的每个词元（字符）。
        # * vocab[token]: 使用 vocab 对象将当前的词元（字符）映射到它在词汇表中的唯一整数索引。
        # Vocab 类很可能实现了 __getitem__ 方法，使得可以通过 vocab[token] 的方式获取词元的索引。
        # * 最终结果： corpus 将是一个长长的整数列表，其中每个整数代表一个字符在词汇表中的索引。
        # 这是神经网络能够处理的数值形式的文本数据。
        corpus = [vocab[token] for line in tokens for token in line]# 将所有词元展平并映射到它们的索引
        if max_tokens > 0:# 如果指定了最大词元数，则截断语料库
            corpus = corpus[: max_tokens]
        return corpus, vocab# 返回词元索引列表和词汇表对象

    corpus, vocab = load_corpus_time_machine()
    print(len(corpus), len(vocab))# 打印语料库的长度和词汇表的大小


    #自然语言统计
    import random
    import torch

    print("++++++++++++++++++++++++++++++")
    tokens = tokenize(read_time_machine())
    # 因为每个文本行不一定是一个句子或一个段落 所以我们把所有文本行连接到一起
    corpus = [token for line in tokens for token in line]
    vocab = Vocab(corpus)
    print(vocab.token_freqs[:10])
    freqs = [freq for token, freq in vocab.token_freqs]
    # plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
    #      xscale='log', yscale='log')
    #plt.show()
    #zip func是这两个错位的列表一一配对 形成一个个的二元元组
    bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
    bigram_vocab = Vocab(bigram_tokens)
    print(bigram_vocab.token_freqs[:10])

    trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
    trigram_vocab = Vocab(trigram_tokens)
    print(trigram_vocab.token_freqs[:10])

    bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
    trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
    # plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x', ylabel='frequency: n(x)',
    #      xscale='log', yscale='log',
    #      legend=['unigram','bigram','trigram'])
    #plt.show()

    # random samping
    def seq_data_iter_random(corpus, batch_size, num_steps):
        """使用随机抽样生成一个小批量子序列"""
        # 从随机偏移量开始对序列进行分区 随机范围包括num_steps-1
        # 用来随机选取语料库的起始位置 确保每次迭代从不同的随机位置开始
        corpus = corpus[random.randint(0, num_steps -1):]#random.randint(0, num_steps -1)随机选择一个整数
        # 然后用这个随机索引去访问corpus中的一个元素 应该是随机选择一个起始点 长度为num_steps -1
        # 减去1 是因为我们需要考虑标签 计算可以生成的子序列的数量
        num_subseqs = (len(corpus) - 1) // num_steps
        # 长度为num_steps的子序列的起始索引
        # 创建一个初始索引列表表示每个子序列的起始位置
        initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
        # 在随机抽样的迭代过程中 来自两个相邻的 随机的 小批量中的子序列不一定在原始序列中相邻
        random.shuffle(initial_indices)

        # 定义一个内部函数 用于根据位置pos获取数据
        def data(pos):
            # 返回从pos位置开始的长度为num_steps的序列
            return corpus[pos: pos + num_steps]

        num_batches = num_subseqs // batch_size
        # 循环遍历批次
        for i in range(0, batch_size * num_batches, batch_size):
            # 在这里 initial_indices包含子序列的随机起始索引 获取当前批次的初始索引
            initial_indices_per_batch = initial_indices[i: i + batch_size]
            X = [data(j) for j in initial_indices_per_batch]
            Y = [data(j + 1) for j in initial_indices_per_batch]
            # 通过yield把XY转换为tensor
            yield torch.tensor(X), torch.tensor(Y)

    my_seq = list(range(35))
    for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
        print('X: ', X, '\nY:', Y)

    

    # seqential partitioning
    # 保证两个相邻的小批量中的子序列在原始序列中也是相邻的
    # 这对于某些需要保持序列上下文的任务很重要
    def seq_data_iter_sequential(corpus, batch_size, num_steps):
        # 这个offset用于从语料库的随机起始点开始提取数据 但之后的数据依然是连续的 这有助于在每次训练迭代时
        # 不同批次的起始点发生变化 增加数据的随机性 避免模型过度依赖起始模式
        offset = random.randint(0, num_steps)#生成一个随机偏移量
        # 计算实际将用于构建数据集的总词语数量
        # 这里的-1是为了确保有足够的空间为标签Y 即X序列的下一个词 len(corpus) - offset - 1
        # 这计算了在考虑offset后 从有效的起始位置到可以形成完整的XY序列的末尾 还有多少个词
        # //batch_size整除 可以得到可以形成多少个完整的批次
        # *batch_size 得到实际用于构建完整批次的 可以被batch_size整除的总词语数量
        num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
        Xs = torch.tensor(corpus[offset:offset + num_tokens])
        Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
        # 将Xs Ys重塑成二维 第一个是批次大小 第二维是-1 自动计算
        # 将1维词语序列重塑为多个并行序列 每个并行序列的长度是num_tokens/batch_size
        # 这样每个批次可以在不同的子序列上并行处理
        Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
        # Xs.shape[1]每个并行序列的长度
        # //是计算每个并行序列可以切分成多少个长度为num_steps的块
        num_batches = Xs.shape[1] // num_steps

        for i in range(0, num_steps * num_batches, num_steps):
            X = Xs[:, i: i + num_steps]#选择每个并行序列中 从i至i+num_steps的子序列
            Y = Ys[:, i: i + num_steps]
            yield X,Y

    for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
        print('X: ', X, '\nY:', Y)


    # 封装
    class SeqDataLoader:
        # self代表实例
        def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
            if use_random_iter:
                self.data_iter_fn = seq_data_iter_random
            else:
                self.data_iter_fn = seq_data_iter_sequential
            self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
            # 将传入构造函数的batch_size num_steps赋值给实例的属性
            self.batch_size, self.num_steps = batch_size, num_steps

        # 这个返回的结果是一个生成器(或迭代器)
        # 每次调用next()都会产生一个(X,Y)的批次 这使得SeqDataLoader实例可以直接作为数据流
        # 在训练循环中使用
        # Grok：定义了iter方法 使得SeqDataLoader对象可以作为迭代器使用！！！
        def __iter__(self):
            return self.data_iter_fn(self.corpus,self.batch_size,self.num_steps)

    # 加载时光机器语料并准备数据
    # 提供了一个方便的接口 让用户通过load_corpus_time_machine函数直接获取用于模型训练的数据迭代器和词汇表
    # Grok：创建一个SeqDataLoader实例 传入参数 返回实例（作为迭代器）和其词汇表！！！
    def load_data_time_machine(batch_size, num_steps,
                                 use_random_iter=False, max_tokens=10000):
        """返回时光机器的数据集的迭代器和词表"""
        # atch_size, num_steps用于初始化内部的SeqDataLoader 传给它的构造函数 构造一个class对象
        # 这个实例本身是可迭代的（因为它实现了__iter__方法，可以直接在for循环中使用来获取数据批次）
        data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter,max_tokens)
        return data_iter, data_iter.vocab

    # RNN
    # 演示X W_xh H W_hh
    import torch
    
    X, W_xh = torch.normal(0, 1, (3, 1)), torch.normal(0, 1, (1, 4))
    H, W_hh = torch.normal(0, 1, (3, 4)), torch.normal(0, 1, (4, 4))
    print(torch.matmul(X, W_xh) + torch.matmul(H, W_hh))

    print(torch.matmul(torch.cat((X, H), 1), torch.cat((W_xh, W_hh), 0)))

    # create a rnn net from 0
    import math
    import torch
    from torch import nn
    from torch.nn import functional as F

    batch_size, num_steps = 32, 35
    train_iter, vocab = load_data_time_machine(batch_size, num_steps)

    # one-hot
    print(F.one_hot(torch.tensor([0, 2]), len(vocab)))

    X = torch.arange(10).reshape((2, 5))
    print(F.one_hot(X.T, 28).shape)


    
    # init parameter of the model
    # vocab_size词汇表的大小 用于输入层的维度
    def get_params(vocab_size, num_hiddens, device):#num_hiddens is super parameter
        num_inputs = num_outputs = vocab_size#可能是一种RNN或者特殊的自编吗器autoencoder
        #函数内部嵌套函数normal
        def normal(shape):#shape表示要创建张量的形状
            #标准正态分布抽样 乘以0.01 这是一种常见的权重初始化策略 旨在避免过大或者过小的初始权重导致训练不稳定
            return torch.randn(size=shape, device=device) * 0.01

        # hidden layer parameter
        # 输入到隐藏层的权重Matrix 它的形状是(num_inputs, num_hiddens) 表示num_inputs个输入特征到num_hiddens个隐藏单元的连接
        W_xh = normal((num_inputs, num_hiddens))
        # 隐藏层自身内部的权重 表示前一时刻的隐藏状态到当前隐藏状态的连接 它的形状是(num_hiddens, num_hiddens)
        W_hh = normal((num_hiddens, num_hiddens))
        # 是隐藏层的偏置向量 它是一个全零的张量 形状为num_hiddens 每个隐藏单元对应一个偏置
        b_h = torch.zeros(num_hiddens, device=device)
        # outlayer parameter
        # 输出的权重matrix 形状是(num_hiddens, num_outputs) 表示num_hiddens个隐藏单元到num_outputs个输出单元的连接
        W_hq = normal((num_hiddens, num_outputs))
        # 是输出层的偏置向量 它是一个全零的张量 num_outputs 每个隐藏单元对应一个偏置
        b_q = torch.zeros(num_outputs, device=device)
        # 附加梯度
        params = [W_xh, W_hh, b_h, W_hq, b_q]
        for param in params:
            param.requires_grad_(True)
        return params

    # RNN net Model
    # define a function 在初始化时返回隐状态
    # 在RNN中 隐藏状态在每一个时间步都会更新 而第一次处理输入时 通常会用一个全零的向量作为初始隐藏状态
    def init_rnn_state(batch_size, num_hiddens, device):
        return (torch.zeros((batch_size, num_hiddens), device=device),)#加逗号 不加逗号是创建一个指定形状的pytorch的张量
        # 加逗号是 python会将其解释为一个包含单个元素的元组

    # rnn函数定义了如何在一个时间步内计算隐状态和输出 RNN模型通过inputs最外层的维度（通常是时间步维度num_steps）实现循环
    # 以便逐时间步更新小批量数据的隐状态H
    def rnn(inputs, state, params):#state上一个时间的隐藏状态
        # input的形状为(时间步数，批量大小，词表大小)
        W_xh, W_hh, b_h, W_hq, b_q = params#将params列表解包
        H, = state#将传入的state(上一个时间步的隐藏状态)赋值给变量H，作为当前的时间步计算的起始隐藏状态
        outputs = []#初始化一个空列表用于存储每个时间步的输出
        # X的形状为（批量大小，词表大小）inputs形状是(时间步数，批量大小，词表大小)
        # for循环会逐个处理每个时间步的输入X 在每个时间步中 X的形状是（批量大小，词表大小）
        for X in inputs:
            # mm(X, W_xh)表示输入对当前隐藏状态的贡献 mm(H, W_hh)表示上一个隐藏状态对当前隐藏状态的贡献
            H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
            Y = torch.mm(H, W_hq) + b_q
            outputs.append(Y)
        # 将outputs列表中所有输出张量沿着维度0（即时间步维度）进行拼接 这样outputs的最终形状是(时间步数 批量大小 输出维数)
        return torch.cat(outputs, dim=0),(H,)#最后一个时间步的隐藏状态 这个隐藏状态通常会被传递给下一个序列的初始状态
        #或者序列结束时用于进一步的任务如分类

    # 创建一个class包含这些函数 并存储从零开始实现的RNN
    class RNNModelScratch:
        #init方法调用get-params方法来获取参数 并保存init_rnn_state and rnn作为其内部方法?
        def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
            self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
            self.params = get_params(vocab_size, num_hiddens, device)
            self.init_state, self.forward_fn = init_state, forward_fn
        #call方法会将输入数据 隐藏状态和模型参数传给rnn函数
        def __call__(self, X, state):
            X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
            return self.forward_fn(X, state, self.params)

        def begin_state(self, batch_size, device):
            return self.init_state(batch_size, self.num_hiddens, device)

    # 检查输出是否具有正确的形状 如隐藏态的维度是否保持不变
    num_hiddens = 512
    net = RNNModelScratch(len(vocab), num_hiddens, 'mps', get_params, init_rnn_state, rnn)

    #print("state before:", state)
    
    state = net.begin_state(X.shape[0], 'mps')
    #print("state after:", state)
    Y, new_state = net(X.to('mps'), state)
    print(Y.shape, len(new_state), new_state[0].shape)

    # predict  
    # 预热期
    def predict_ch8(prefix, num_preds, net, vocab, device):
        state = net.begin_state(batch_size=1, device=device)#batch_size=1一次预测一个序列
        outputs = [vocab[prefix[0]]]#存储生成的字符串索引 首先加入prefix的第一个字符的索引
        #用于获取当前最后一个生成的字符--作为下一个时间步的输入 它将索引转换为形如(1,1)的张量
        #outputs[-1]在列表中表示最后一个元素
        get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1,1))
        #在前缀中除第一个字符外的其他字符进行迭代
        #预热阶段
        for y in prefix[1:]:#预热期
            #每次迭代 模型接受get_input--前一个字符作为输入 并更新隐藏状态state _表示我们不关心这次的输出
            # 只关心更新后的隐藏状态
            _, state = net(get_input(),state)
            outputs.append(vocab[y])
        
        #预测阶段
        #模型接受最后一个字符作为输入 并生成新的预测 y是模型对应的下一个字符的预测--通常是各个字符的概率分布
        for _ in range(num_preds):#预测num_preds步
            y, state = net(get_input(), state)
            #从模型的输出中找到概率最高的字符的索引y.argmax(dim=1) 然后将其转成整数 并添加到outputs中
            outputs.append(int(y.argmax(dim=1).reshape(1)))
        #将outputs中的字符索引转换为实际字符 然后拼接成一个字符串返回
        return ''.join([vocab.idx_to_token[i] for i in outputs])

    print(predict_ch8('time traveller ', 10, net, vocab, 'mps'))

    # 梯度截断
    def grad_clipping(net, theta):
        if isinstance(net, nn.Module):
            params = [p for p in net.parameters() if p.requires_grad]
        else:
            params = net.params
        norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
        if norm > theta:
            for param in params:
                param.grad[:] *= theta / norm

    # train RNN

    #一轮内训练函数
    def train_epoch_ch8(net, train_iter, loss, update, device, use_random_iter):
        """训练网络一轮"""
        state, timer = None , Timer()
        #timer = Timer()
        #timer.start()
        metric = Accumulator(2) # 训练损失之和 词元数量
        #loss = nn.CrossEntropyLoss()
        for X, Y in train_iter:
            if state is None or use_random_iter:
                #第一次使用迭代或者使用随机抽样时初始化state
                state = net.begin_state(batch_size=X.shape[0], device=device)
            else:
                if isinstance(net, nn.Module) and not isinstance(state, tuple):
                    # state对于nn.GRU是一个张量
                    state.detach_()
                else:
                    # state对于nn.LSTM或者对于我们从零开始实施的模型是一个由张量组成的元组
                    for s in state:
                        s.detach_()
            y = Y.T.reshape(-1)

            X, y = X.to(device), y.to(device)
            y_hat, state = net(X, state)
            l = loss(y_hat, y.long()).mean()
            if isinstance(updater, torch.optim.Optimizer):
                updater.zero_grad()
                l.backward()
                grad_clipping(net, 1)
                updater.step()
            else:
                l.backward()
                grad_clipping(net,1)
                # 因为已经调用了mean函数
                updater(batch_size=1)
            metric.add(l * y.numel(), y.numel())
        return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
    
    def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
        loss = nn.CrossEntropyLoss()
        animator = Animator(xlabel='epoch', ylabel='perplexity', legend=['train'], xlim=[10, num_epochs])
        #初始化
        if isinstance(net, nn.Module):
            updater = torch.optim.SGD(net.parameters(), lr)
        else:
            updater = lambda batch_size:sgd(net.parameters(), lr, batch_size)
        predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
        # train and predict
        for epoch in range(num_epochs):
            ppl, speed = train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter=False)
            if (epoch + 1) % 10 == 0:
                print(predict('time traveller'))
                animator.add(epoch + 1, [ppl])
    
        print(f'perplexity {ppl: .1f}, {speed: .1f} 词元/秒 {str(device)}')
        print(predict('time traveller'))
        print(predict('traveller'))

    num_epochs, lr = 500, 1
    train_ch8(net, train_iter, vocab, lr, num_epochs, 'mps')
    plt.show()
    
    def get_params(vocab_size, num_hiddens, device):
        num_inputs = num_outputs = vocab_size

        def normal(shape):
            return torch.randn(size=shape, device=device) * 0.01

        # 隐藏层参数
        W_xh = normal((num_inputs, num_hiddens))
        W_hh = normal((num_hiddens, num_hiddens))
        b_h = torch.zeros(num_hiddens, device=device)
        # 输出层参数
        W_hq = normal((num_hiddens, num_outputs))
        b_q = torch.zeros(num_outputs, device=device)
        # 附加梯度
        params = [W_xh, W_hh, b_h, W_hq, b_q]
        for param in params:
            param.requires_grad_(True)
        return params

    def init_rnn_state(batch_size, num_hiddens, device):
        return (torch.zeros((batch_size, num_hiddens), device=device), )

    def rnn(inputs, state, params):
        # inputs的形状：(时间步数量，批量大小，词表大小)
        W_xh, W_hh, b_h, W_hq, b_q = params
        H, = state
        outputs = []
        # X的形状：(批量大小，词表大小)
        for X in inputs:
            H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
            Y = torch.mm(H, W_hq) + b_q
            outputs.append(Y)
        return torch.cat(outputs, dim=0), (H,)

    class RNNModelScratch: #@save
        """从零开始实现的循环神经网络模型"""
        def __init__(self, vocab_size, num_hiddens, device,
                    get_params, init_state, forward_fn):
            self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
            self.params = get_params(vocab_size, num_hiddens, device)
            self.init_state, self.forward_fn = init_state, forward_fn

        def __call__(self, X, state):
            X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
            return self.forward_fn(X, state, self.params)

        def begin_state(self, batch_size, device):
            return self.init_state(batch_size, self.num_hiddens, device)

    num_hiddens = 512
    net = RNNModelScratch(len(vocab), num_hiddens, 'mps', get_params,
                        init_rnn_state, rnn)
    state = net.begin_state(X.shape[0], 'mps')
    Y, new_state = net(X.to('mps'), state)
    print(Y.shape, len(new_state), new_state[0].shape)

    #预测函数
    def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
        """在prefix后面生成新字符"""
        state = net.begin_state(batch_size=1, device=device)
        outputs = [vocab[prefix[0]]]
        get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
        for y in prefix[1:]:  # 预热期
            _, state = net(get_input(), state)
            outputs.append(vocab[y])
        for _ in range(num_preds):  # 预测num_preds步
            y, state = net(get_input(), state)
            #y.argmax(dim=1) 直接选择了y(模型的输出logits或概率分布)中概率最大的那个索引 然后将其转换为相应的字符
            #改成抽样 1 对y进行softmax运算 模型的原始输出y通常是logits--未归一化的对数概率，需要通过softmax函数
            #对其转换为概率分布 2 应用温度参数(可选但推荐)：为了控制抽样的随机性 可以引入温度参数 在softmax之前对
            #logits进行缩放 3从概率分布中抽样 而不是直接选择最大值
            outputs.append(int(y.argmax(dim=1).reshape(1)))
        return ''.join([vocab.idx_to_token[i] for i in outputs])
    
    #预测函数-使用抽样
    # def predict_ch8(prefix, num_preds, net, vocab, device, temperature=1.0):  #@save
    #     """在prefix后面生成新字符"""
    #     state = net.begin_state(batch_size=1, device=device)
    #     outputs = [prefix[0]]
    #     get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    #     for y in prefix[1:]:  # 预热期
            
    #         get_input = torch.tensor(vocab[y],device=device).reshape((1,1))
    #         y, state = net(get_input, state)
    #     for _ in range(num_preds):  # 预测num_preds步
    #         # 1应用温度参数进行softmax
    #         Y_logits, state = net(get_input, state)
    #         Y_probs = F.softmax(Y_logits / temperature, dim=1)
    #         # 2 从概率分布中抽样
    #         idx = torch.multinomial(Y_probs, num_samples=1).squeeze(0)
    #         char = vocab.idx_to_token[idx.item()]
    #         outputs.append(char)
    #         get_input = torch.tensor(idx,device=device).reshape((1,1))
    #         #y.argmax(dim=1) 直接选择了y(模型的输出logits或概率分布)中概率最大的那个索引 然后将其转换为相应的字符
    #         #改成抽样 1 对y进行softmax运算 模型的原始输出y通常是logits--未归一化的对数概率，需要通过softmax函数
    #         #对其转换为概率分布 2 应用温度参数(可选但推荐)：为了控制抽样的随机性 可以引入温度参数 在softmax之前对
    #         #logits进行缩放 3从概率分布中抽样 而不是直接选择最大值
    #         #outputs.append(int(y.argmax(dim=1).reshape(1)))
    #     return ''.join(outputs)

    print(predict_ch8('time traveller ', 10, net, vocab, 'mps'))

    def grad_clipping(net, theta):  #@save
        """裁剪梯度"""
        if isinstance(net, nn.Module):
            params = [p for p in net.parameters() if p.requires_grad]
        else:
            params = net.params
        norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
        if norm > theta:
            for param in params:
                param.grad[:] *= theta / norm

    #@save
    def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
        """训练网络一个迭代周期"""
        # state存储网络状态
        state, timer = None, Timer()
        metric = Accumulator(2)  # 创建一个累加器 用于统计2个：训练损失之和,词元数量
        for X, Y in train_iter:
            if state is None or use_random_iter:
                # 在第一次迭代或使用随机抽样时初始化state 生成初始隐藏状态 X.shape[0]：X的第一维度 批次大小
                state = net.begin_state(batch_size=X.shape[0], device=device)
            else:# 非首次迭代或非随机迭代
                if isinstance(net, nn.Module) and not isinstance(state, tuple):#state不是元组
                    # state对于nn.GRU是个张量
                    # 断开state的梯度计算 防止梯度累积
                    state.detach_()
                else:#state是元组
                    # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                    for s in state:#遍历state中的每个张量
                        s.detach_()
            # 将Y转置并重塑一个一维向量 -1自动推断长度
            y = Y.T.reshape(-1)
            X, y = X.to(device), y.to(device)
            # 前向传播 state是更新后的隐藏状态 y_hat模型预测输出
            y_hat, state = net(X, state)
            # y.long()将y转换为长整形 适配损失函数要求 mean取损失函数的平均值
            l = loss(y_hat, y.long()).mean()
            # 判断updater是否是pytorch的优化器
            if isinstance(updater, torch.optim.Optimizer):
                updater.zero_grad()
                l.backward()
                #对网络参数的梯度进行裁剪--限制在1以内 防止梯度爆炸
                grad_clipping(net, 1)
                updater.step()
            else:#不是pytorch的优化器
                l.backward()
                grad_clipping(net, 1)
                # 因为已经调用了mean函数
                # 调用自定义的更新器 批次设置为1
                # d2l.sgd设计是针对没有平均的目标函数使用的 由batch_size长度的全体loss.sum() 所以要在梯度更新时除以batch_size
                updater(batch_size=1)
            # 累加损失(乘以元素) 和总元素
            metric.add(l * y.numel(), y.numel())
        # 平均损失 / 每秒处理的词元数
        return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

    #@save
    # vocab词汇表用于文本处理
    def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
                use_random_iter=False):
        """训练模型"""
        loss = nn.CrossEntropyLoss()
        # 用于可视化训练过程中的困惑度perplexity
        animator = Animator(xlabel='epoch', ylabel='perplexity',
                                legend=['train'], xlim=[10, num_epochs])
        # 初始化
        # 如果net是nn Module类型
        if isinstance(net, nn.Module):
            updater = torch.optim.SGD(net.parameters(), lr)
        else:
            # 匿名函数 接受批次大小作为参数
            # d2l.sgd设计是针对没有平均的目标函数使用的 由batch_size长度的全体loss.sum() 所以要在梯度更新时除以batch_size
            updater = lambda batch_size: sgd(net.params, lr, batch_size)
        # 定义个预测函数 生成长度为50的序列
        # lambda将这些固定参数封装起来 简化了调用 只暴露predix作为用户输入
        predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
        # 训练和预测
        for epoch in range(num_epochs):
            ppl, speed = train_epoch_ch8(
                net, train_iter, loss, updater, device, use_random_iter)
            if (epoch + 1) % 10 == 0:#每10个epoch执行一次
                print(predict('time traveller'))
                animator.add(epoch + 1, [ppl])
        print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
        print(predict('time traveller'))
        print(predict('traveller'))

    num_epochs, lr = 500, 1
    #train_ch8(net, train_iter, vocab, lr, num_epochs, 'mps')
    #plt.show()

    net = RNNModelScratch(len(vocab), num_hiddens, 'mps', get_params, init_rnn_state, rnn)
    #train_ch8(net, train_iter, vocab, lr, num_epochs, 'mps', use_random_iter=True)
    #plt.show()

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # 函数detch的作为见gmail邮箱
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    import torch
    from torch import nn
    from torch.nn import functional as F

    batch_size, num_steps = 32, 35
    train_iter, vocab = load_data_time_machine(batch_size, num_steps)

    # define model
    num_hiddens = 256#隐藏单元
    #创建一个简单的RNN层 输入维度--词汇表长度 输出维度
    rnn_layer = nn.RNN(len(vocab), num_hiddens)
    # init state (隐藏层数，批量大小，隐藏单元数)
    state = torch.zeros((1, batch_size, num_hiddens))
    print(state.shape)

    X = torch.rand(size=(num_steps, batch_size, len(vocab)))
    Y, state_new = rnn_layer(X, state)
    print(Y.shape, state_new.shape)

    class RNNModel(nn.Module):
        def __init__(self, runn_layer, vocab_size, **kwargs):
        # def __init__(self, gru_layer, vocab_size, **kwargs):
            super(RNNModel, self).__init__(**kwargs)
            self.rnn = rnn_layer
            # self.gru = gru_layer#clp2025
            self.vocab_size = vocab_size
            self.num_hiddens = self.rnn.hidden_size
            #如果RNN是双向的
            if not self.rnn.bidirectional:
                self.num_directions = 1
                self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
            else:
                self.num_directions = 2
                self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

        def forward(self, inputs,state):
            X = F.one_hot(inputs.T.long(), self.vocab_size)
            X = X.to(torch.float32)
            Y, state = self.rnn(X, state)
            #全连接层首先将Y的形状改为(时间步数x批量大小，隐藏单元数)
            #它的输出形状为(时间步数x批量大小， 词表大小)
            outputs = self.linear(Y.reshape((-1, Y.shape[-1])))
            return outputs, state

        def begin_state(self, device, batch_size=1):
            if not isinstance(self.rnn, nn.LSTM):
                # nn.GRU以张量作为隐状态
                return torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens),
                                    device=device)

            else:
                #如果是LSTM,返回一个元组 包含两个形状相同的零张量 分别对应隐状态和细胞状态
                return (torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens),
                                    device=device),torch.zeros((elf.num_directions * self.rnn.num_layers, batch_size, 
                                    self.num_hiddens), device=device))

    # device = 'mps'
    # net = RNNModel(rnn_layer, vocab_size=len(vocab))
    # net = net.to(device)
    # print(predict_ch8('time traveller', 10, net, vocab, device))

    # num_epochs, lr = 500, 1
    #train_ch8(net, train_iter, vocab, lr, num_epochs, device)
    #plt.show()

    ########################################################################
    #     `**kwargs` 参数
    # **kwargs 是一个可变关键字参数，允许函数接受任意数量的命名参数（关键字参数）。这些参数以字典的形式传递给函数。

    # 用法：在 __init__ 中使用 **kwargs 可以让类接受灵活的参数组合，而无需显式定义所有可能的参数。

    # 工作原理：kwargs 是一个字典，键是参数名，值是参数值。可以通过 kwargs.get() 或直接访问 kwargs['key'] 使用这些值。

    # 示例（包含 `**kwargs`）
    # class Person:
    #     def __init__(self, name, age, **kwargs):
    #         self.name = name
    #         self.age = age
    #         # 处理额外的关键字参数
    #         self.additional_info = kwargs

    # person = Person("Alice", 25, city="New York", occupation="Engineer")
    # print(person.name)              # 输出: Alice
    # print(person.age)               # 输出: 25
    # print(person.additional_info)   # 输出: {'city': 'New York', 'occupation': 'Engineer'}
    # 这里，**kwargs 捕获了 city 和 occupation 两个额外的参数，并存储在 self.additional_info 中。

    # 你还可以有选择地使用这些参数：

    # class Person:
    #     def __init__(self, name, age, **kwargs):
    #         self.name = name
    #         self.age = age
    #         self.city = kwargs.get('city', 'Unknown')  # 默认值 'Unknown'
    #         self.occupation = kwargs.get('occupation')

    # person = Person("Alice", 25, city="New York")
    # print(person.city)          # 输出: New York
    # print(person.occupation)    # 输出: None
    # 注意事项
    # 必须有 self：__init__ 的第一个参数必须是 self，表示当前实例。

    # **kwargs 的灵活性：它允许子类或用户扩展类的初始化，而无需修改父类定义。

    # 与其他参数的结合：__init__ 可以同时使用位置参数和 **kwargs，例如 def __init__(self, required, *args, **kwargs)。

    # 继承中的使用：在继承中，super().__init__(**kwargs) 可以将 kwargs 传递给父类的 __init__，实现参数的灵活传递。

    # 总结
    # __init__ 是 Python 类实例化的关键方法，用于设置初始状态。**kwargs 增强了其灵活性，允许接受任意关键字参数，从而使类更具扩展性。
    ##############################################################################################################


    #Chapter 9
    #现代RNN

    #GRU
    import torch
    from torch import nn

    batch_size, num_steps = 32, 35
    train_iter, vocab = load_data_fashion_minist(batch_size, num_steps)
    
    # 初始化模型参数
    def get_params(vocab_size, num_hiddens, device):
        num_inputs = num_outputs = vocab_size

        def normal(shape):
            return torch.randn(size=shape, device=device) * 0.01

        def three():#隐藏层数
            return (normal((num_inputs, num_hiddens)),
                    normal((num_hiddens, num_hiddens)),
                    torch.zeros(num_hiddens, device=device))

        W_xz, W_hz, b_z = three() #更新门参数
        W_xr, W_hr, b_r = three() #重置门参数
        W_xh, W_hh, b_h = three() #候选隐状态参数

        # 输出层参数
        W_hq = normal((num_hiddens, num_outputs))
        b_q = torch.zeros(num_outputs, device=device)

        # 附加梯度
        params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
        for param in params:
            param.requires_grad_(True)

        return params

    # define model
    def init_gru_state(batch_size, num_hiddens, device):
        return (torch.zeros((batch_size, num_hiddens), device=device),)

    def gru(inputs, state, params):
        W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
        H, = state
        outputs = []

        for X in inputs:
            Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
            R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
            H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
            H = Z * H + (1 - Z) * H_tilda
            Y = H @ W_hq + b_q
            outputs.append(Y)

        return torch.cat(outputs, dim=0), (H,)



    #train and predict
    device = device
    vocab_size, num_hiddens = len(vocab), 256
    num_epochs, lr = 500, 1
    #model = RNNModelScratch(len(vocab), num_hiddens, device, get_params, init_gru_state, gru)

    # train_ch8(model, train_iter, vocab, lr, num_epochs, device)
    # plt.show()
'''



    # high level API
    num_inputs = vocab_size
    gru_layer = nn.GRU(num_inputs, num_hiddens)
    model = RNNModel(gru_layer, len(vocab))
    model = model.to(device)
    train_ch8(model, train_iter, vocab, lr, num_epochs, device)

###end
