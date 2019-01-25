#K近邻算法

from numpy import *
import operator

#创建数据集
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    lables = ['A','A','B','B']
    return group,lables

#根据数据集推算输入数据的分类
def classify0(inX,dataSet,lables,k):
    dataSetSize = dataSet.shape[0] #集合内第一维度的数据个数
    '''
    tile(inX, (dataSetSize, 1)) 根据集合数据个数，扩展输入数据，和集合数据的维度保持一致
    计算输入数据和训练集合数据的差值
    '''
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    '''
    计算欧式距离
    '''
    sqDiffMat = square(diffMat) #平方
    print(sqDiffMat)
    sqDistances =  sqDiffMat.sum(axis=1) #求和
    print(sqDistances)
    distances = sqrt(sqDistances) #开方 得到欧式距离
    print(distances)
    #排序 - 从小到大，argsort排序后展示的为数据的索引
    sortedDistances = distances.argsort()
    print(sortedDistances)
    classCount = {}
    for i in range(k):
        voteIlabel = lables[sortedDistances[i]] #根据排序数据，获取对应的标签
        # print(voteIlabel)
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
        # print(classCount)
    #reverse=True逆序排序，operator.itemgetter(1)用于获取哪些维的数据
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    print(sortedClassCount)
    return sortedClassCount[0][0]

if __name__ == '__main__':
    group,lables = createDataSet();
    lable = classify0([1,2],group,lables,3)
    print(lable)
