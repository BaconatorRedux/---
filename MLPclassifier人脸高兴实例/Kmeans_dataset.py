import numpy as np
import random
from pyecharts import options as  opts
from pyecharts.charts import Scatter3D
from pyecharts.faker import Faker
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd  # 导入csv的 功能模块

size = 200  ##取值范围


##计算欧式距离
def distEuclid(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


# 导入csv 数据
def genDataset():
    data = []
    da = pd.read_csv("data.csv")
    data = da[["x", "y", "z"]]
    return data


## 初始化簇中心点 一开始随机从样本中选择k个 当做各类簇的中心
def initCentroid(data, k):
    num, dim = data.shape
    centpoint = np.zeros((k, dim))
    l = [x for x in range(num)]
    np.random.shuffle(l)
    for i in range(k):
        index = int(l[i])
        centpoint[i] = data[index]
    return centpoint


##进行KMeans分类
def KMeans(data, k):
    ##样本个数
    num = np.shape(data)[0]

    ##记录各样本 簇信息 0:属于哪个簇 1:距离该簇中心点距离
    cluster = np.zeros((num, 2))
    cluster[:, 0] = -1

    ##记录是否有样本改变簇分类
    change = True
    ##初始化各簇中心点
    cp = initCentroid(data, k)

    while change:
        change = False

        ##遍历每一个样本
        for i in range(num):
            minDist = 9999.9
            minIndex = -1

            ##计算该样本距离每一个簇中心点的距离 找到距离最近的中心点
            for j in range(k):
                dis = distEuclid(cp[j], data[i])
                if dis < minDist:
                    minDist = dis
                    minIndex = j

            ##如果找到的簇中心点非当前簇 则改变该样本的簇分类
            if cluster[i, 0] != minIndex:
                change = True
                cluster[i, :] = minIndex, minDist

        ## 根据样本重新分类  计算新的簇中心点
        for j in range(k):
            pointincluster = data[[x for x in range(num) if cluster[x, 0] == j]]
            cp[j] = np.mean(pointincluster, axis=0)

    print("finish!")
    return cp, cluster


##展示结果  各类簇使用不同的颜色  中心点使用X表示
def Show(data, k, cp, cluster):
    num = data.shape[0]
    color = ['r', 'g', 'b', 'c', 'y', 'm', 'k']

    ax = plt.subplot(111, projection='3d')
    for i in range(num):
        mark = int(cluster[i, 0])
        ax.scatter(data[i, 0], data[i, 1], data[i, 2], c=color[mark])

    for i in range(k):
        ax.scatter(cp[i, 0], cp[i, 1], cp[i, 2], c=color[i], marker='x')

    plt.ylim(plt.ylim()[::-1])  # 翻转Y轴
    ax.set_xlabel('X')  # 设置x轴的文本，用于描述x轴代表的是什么
    ax.set_ylabel('Y')  # 设置y轴的文本，用于描述y轴代表的是什么
    ax.set_zlabel('Z')  # 设置z轴的文本，用于描述z轴代表的是什么
    plt.savefig(fname="Kmeans_result.svg",format="svg")
    plt.show()

num = 200  ##点个数
k = 5  ##分类个数
data = np.array(genDataset())
cp, cluster = KMeans(data, k)
Show(data, k, cp, cluster)
print("over")