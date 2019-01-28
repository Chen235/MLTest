#K近邻算法

from numpy import *
import operator
import matplotlib.pyplot as plt

class Test1():
    @classmethod
    #创建数据集
    def createDataSet(self):
        group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
        lables = ['A','A','B','B']
        return group,lables

    @classmethod
    #根据数据集推算输入数据的分类
    def classify0(self,inX,dataSet,lables,k):
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
        # print(sqDiffMat)
        sqDistances =  sqDiffMat.sum(axis=1) #求和
        # print(sqDistances)
        distances = sqrt(sqDistances) #开方 得到欧式距离
        # print(distances)
        #排序 - 从小到大，argsort排序后展示的为数据的索引
        sortedDistances = distances.argsort()
        # print(sortedDistances)
        classCount = {}
        for i in range(k):
            voteIlabel = lables[sortedDistances[i]] #根据排序数据，获取对应的标签
            # print(voteIlabel)
            classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
            # print(classCount)
        #reverse=True逆序排序，operator.itemgetter(1)用于获取哪些维的数据
        sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
        # print(sortedClassCount)
        return sortedClassCount[0][0]

class Test2():
    @classmethod
    #kNN练习，处理数据
    def file2matrix(self,filename):
        fr = open(filename)
        arrayLines = fr.readlines()
        numbers = len(arrayLines)
        returnMat = zeros((numbers,3)) #构建(number,3)的全是0的2维矩阵
        classLabelVector = []
        index = 0
        for line in arrayLines:
            line = line.strip() #去除换行符
            listFromLine = line.split(',')
            returnMat[index,:] = listFromLine[0:3]
            classLabelVector.append(int(listFromLine[2]))
            index += 1
        return returnMat,classLabelVector

    @classmethod
    #归一化特征值
    def autoNorm(self,dataSet):
        minValues = dataSet.min(0) #取得列的最小值
        maxValues = dataSet.max(0) #取得列的最大值
        ranges = maxValues - minValues #差值范围
        # normDataSet = zeros(dataSet.shape)
        m = dataSet.shape[0]
        normDataSet = dataSet - tile(minValues,(m,1))
        normDataSet = normDataSet/tile(ranges,(m,1))
        return normDataSet,ranges,minValues

    @classmethod
    #展示散点图
    def showFileSetData(self,mat,lables):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(mat[:,0],mat[:,1],15*array(lables),15*array(lables))
        plt.show()

    @classmethod
    def datingClassTest(self):
        hoRatio = 0.5
        returnMat, classLabels = Test2.file2matrix("datingTestSet.txt")
        normDataSet, ranges, minValues = Test2.autoNorm(returnMat)
        m = normDataSet.shape[0]
        numTestVecs = int(m*hoRatio)
        for i in range(numTestVecs):
            classifyierResult = Test1.classify0(normDataSet[i,:],normDataSet[numTestVecs:m,:],classLabels[numTestVecs:m],3)
            print(classifyierResult)

if __name__ == '__main__':
    # group,lables = Test1.createDataSet()
    # lable = Test1.classify0([1,2],group,lables,3)
    # print(lable)

    Test2.datingClassTest()