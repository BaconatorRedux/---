import os
import random
import re

import joblib
import numpy as np
import cv2
import pandas
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

img_path = "F:/dataset/files"
flg_path = "F:/dataset/labels.txt"
global mlp
global temp

layer_num = []
layer_activation = []
lr = []
learning_rates = []
accs= []
rs = []
def GetFiles(file_dir, file_type, IsCurrent=False):
    '''
        功能：获取指定文件路径&文件类型下的所有文件名
        传入：
            file_dir   文件路径,
            file_type  文件类型,
            IsCurrent  是否只获取当前文件路径下的文件，默认False
        返回：含文件名的列表
    '''
    file_list = []
    for parent, dirnames, filenames in os.walk(file_dir):
        for filename in filenames:
            if filename.endswith(('.%s' % file_type)):  # 判断文件类型
                temp = os.path.join(parent, filename)
                temp = re.sub("/",r"\\",temp)
                file_list.append(temp)

        if IsCurrent == True:
            break
    return file_list



def load_photo():
    print("调用load_photo")
    imgs = []

    dir = "F:/dataset/files"
    files = GetFiles(dir, "jpg")
    for i in files:
        img = cv2.imread(i,cv2.IMREAD_GRAYSCALE)
        res = img/255.0
        res = cv2.resize(res,(180,192))
        res = res.flatten()
        imgs.append(res)
    for i in imgs:
        print(i )
    imgs = np.array(imgs)

    return imgs



def load_label():
    print("调用load_label")
    label = pandas.read_csv(flg_path,header=None,sep=' ')
    lbls = label[0] #提取第一列数据，0为非高兴，1为高兴
    lbls = np.array(lbls)#创建数组
    return lbls

def classify():
    #划分测试集以及训练集（随机划分）test_size为样本占比，stratify=labels按照图片分层划分
    imgs = load_photo()
    labels = load_label()
    tr_data,tr_test,t_data,t_test = train_test_split(imgs,labels, test_size=0.1,shuffle=False)
    return tr_data,tr_test,t_data,t_test


def train():
    #采用sklearn提供的MLP人工神经网络
    #hidden_layer_sizes为隐藏层 hidden_layer_sizes(50,100)为第一层隐藏层含有50个神经元，第二层隐藏层含100个神经元
    #alpha 正则化项参数
    #learning_rate 学习率，用于权重更新 'constant':由初始化学习率给定的恒定学习率,'inscalling':随着时间t使用逆度指数不断降低学习率
    #learning_rate_init 学习率初始化
    #max_iter 最大迭代次数
    #solver：{“lbfgs”,"sgd","adam"} sgd:随机梯度下降,adam:随机梯度优化器,默认大数据集adam
    #activation 激活函数{"identity","logistic","tanh","relu"}
    #identity: f(x) = x
    #logistic: sigmod,f(x) = 1/(1+ exp(-x))
    #tanh: f(x) = tanh(x)
    #relu: f(x) = max(0,x)
    global tr_data,tr_test, t_data, t_test
    acc = []
    tr_data, tr_test, t_data, t_test = classify()

    #随机生成10个人工神经网络
    for i in range(0,30):
        r = random.randint(0,7)
        print("正在生成第" + str(i + 1) + "个模型,其中隐含层数为:"+str(r))
        layer_num.append(i)
        res = exert(r,i)

        print("第"+str(i+1)+"个模型训练准确率是："+str(res))
        acc.append(res)
    return acc


#生成神经网络，拟合数据集，并返回给定测试数据和标签上的平均准确度
def exert(a,i):
    global mlp
    num = range(50, 130, 10)
    activation = ['identity','logistic','tanh','relu']
    learning_rate = ['constant','adaptive','invscaling']
    solver = ['adam','sgd']
    learning_rate_init= [0.1,0.07,0.03,0.001,0.002,0.003,0.005,0.004,0.007]
    r = activation[random.randint(0,len(activation)-1)]
    rri = learning_rate_init[random.randint(0,len(learning_rate_init)-1)]
    rl = learning_rate[random.randint(0,2)]
    rs = solver[random.randint(0,1)]
    lr.append(rri)
    layer_activation.append(r)
    learning_rates.append(rl)


    print("激活函数采用:" + str(r))
    print("学习率初始化为:" + str(rri))
    print("学习率变化为:" + str(rl))
    print("优化算法为:" + str(rs))
    #a为隐含层数
    if (a == 0):
        mlp = MLPClassifier(hidden_layer_sizes=(10), alpha=0.01, max_iter=1000,
                            solver="sgd", learning_rate=rl, learning_rate_init=0.003, activation="relu")
    elif (a == 1):
        mlp = MLPClassifier(hidden_layer_sizes=(10, num[random.randint(0, 7)]), alpha=0.01, max_iter=1000,
                            solver='sgd', learning_rate=rl, learning_rate_init=rri, activation=r)
    elif (a == 2):
        mlp = MLPClassifier(hidden_layer_sizes=(10, num[random.randint(0, 7)], num[random.randint(0, 7)]), alpha=0.01,
                            max_iter=1000,
                            solver='sgd', learning_rate=rl, learning_rate_init=rri, activation=r)
    elif (a == 3):
        mlp = MLPClassifier(
            hidden_layer_sizes=(10, num[random.randint(0, 7)], num[random.randint(0, 7)], num[random.randint(0, 7)]),
            alpha=0.01, max_iter=1000,
            solver='sgd', learning_rate=rl, learning_rate_init=rri, activation=r)
    elif (a == 4):
        mlp = MLPClassifier(hidden_layer_sizes=(
        10, num[random.randint(0, 7)], num[random.randint(0, 7)], num[random.randint(0, 7)],
        num[random.randint(0, 7)]),
                            alpha=0.01, max_iter=1000, solver='sgd', learning_rate=rl, learning_rate_init=rri,
                            activation=r)
    elif (a == 5):
        mlp = MLPClassifier(hidden_layer_sizes=(
        10, num[random.randint(0, 7)], num[random.randint(0, 7)], num[random.randint(0, 7)], num[random.randint(0, 7)],
        num[random.randint(0, 7)]),
                            alpha=0.01, max_iter=1000, solver='sgd', learning_rate=rl, learning_rate_init=rri,
                            activation=r)
    elif (a == 6):
        mlp = MLPClassifier(hidden_layer_sizes=(
        10, num[random.randint(0, 7)], num[random.randint(0, 7)], num[random.randint(0, 7)], num[random.randint(0, 7)],
        num[random.randint(0, 7)], num[random.randint(0, 7)]),
                            alpha=0.01, max_iter=1000, solver='sgd', learning_rate=rl, learning_rate_init=rri,
                            activation=r)
    elif (a == 7):
        mlp = MLPClassifier(hidden_layer_sizes=(
        10, num[random.randint(0, 7)], num[random.randint(0, 7)], num[random.randint(0, 7)], num[random.randint(0, 7)],
        num[random.randint(0, 7)], num[random.randint(0, 7)], num[random.randint(0, 7)]),
                            alpha=0.01, max_iter=1000, solver='sgd', learning_rate=rl, learning_rate_init=rri,
                            activation=r)

    print("正在拟合....")
    mlp.fit(tr_data,t_data)
    print(mlp.n_iter_)
    print("损失函数为:",mlp.loss_)
    score = mlp.score(tr_data,t_data)
    joblib.dump(mlp, str(i+1)+"_"+str(score)+'.pkl')
    return score

if __name__ == "__main__":
    accs = train()
    print("layer_num:",layer_num)
    print("layer_activation:" ,layer_activation)
    print("lr:" , lr)
    print("solver:",rs)
    print("acc：",accs)
