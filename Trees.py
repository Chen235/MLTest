#决策树
import operator
import pickle
from math import log

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels


class Util():

    #计算数据集的总熵
    @classmethod
    def calcEnt(self,dataSet):
        dataSetLen = len(dataSet) #数据集总长度
        labelCounts = {} #标签频次字典
        for data in dataSet: #遍历每一行数据
            label = data[-1] #最后一列：标签列
            #标签计数
            if label not in labelCounts.keys():
                labelCounts[label] = 0
            labelCounts[label] += 1
        ent = 0 #总熵
        for key in labelCounts:
            prob = float(labelCounts[key])/dataSetLen #该标签在数据集中出现的概率
            ent -= prob * log(prob,2) #香农熵公式
        return ent

    #获取指定列的Value数据集
    @classmethod
    def getAxisValueDataSet(self,dataSet,axis,value):
        selDataSet = []
        '''
        遍历数据集
        若指定列数据==指定数据Value，则截取该行数据(去除指定列),并附加到返回列表中
        '''
        for col in dataSet:
            if col[axis] == value:
                selCol = col[:axis]
                selCol.extend(col[axis + 1:])
                selDataSet.append(selCol)
        return selDataSet

    #获取最佳信息增益列
    @classmethod
    def getBestFeature(self,dataSet):
        dataLen = len(dataSet[0])-1 #去除标签列的总列数
        ent = Util.calcEnt(dataSet) #总熵
        bestInfoGain = 0 #最佳信息增益值/信息数据无序度减少量
        bestFeature = -1 #最佳特征列
        for i in range(dataLen):
            featList = [col[i] for col in dataSet] #特征列数据集合
            uniqueVals = set(featList) #特征列数据去重
            newEnts = 0
            '''
            遍历每一个特征
            计算每一个特征值在整个数据集中的熵，并累加为该列特征的总熵
            '''
            for val in uniqueVals:
                selDataSet = Util.getAxisValueDataSet(dataSet,i,val)
                prob = len(selDataSet)/float(len(dataSet))
                newEnts += prob*Util.calcEnt(selDataSet)
            infoGain = ent - newEnts #增益值/数据无需度减少量
            if infoGain > bestInfoGain:
                bestInfoGain = infoGain
                bestFeature = i
        return bestFeature

    #多数计数器
    @classmethod
    def majorityCnt(self, classList):
        classCount = {}
        #遍历数据集，标签计数
        for vote in classList:
            if vote not in classList.keys(): classCount[vote] = 0
            classCount[vote] += 1
        #根据计数从大到小排序
        sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0] #返回计数最多的标签数据

    #保存树形数据
    @classmethod
    def saveTree(self,inputTree,fileName):
        file = open(fileName,'wb')
        pickle.dump(inputTree,file)
        file.close()

    #抓取树形数据
    @classmethod
    def grapTree(self,fileName):
        file = open(fileName,'rb')
        return pickle.load(file)


class Tree():

    #根据数据集生成树形数据
    @classmethod
    def createTree(self,dataSet,labels):
        classList = [col[-1] for col in dataSet] #获取数据集最后一列标签列作为新的数据标签列表
        #在新的数据标签列表中，若第一列元素在列表中出现的次数=列表的长度，则列表数据属于同一类数据，直接返回数据标签
        if classList.count(classList[0]) == len(classList):
            return classList[0]
        #使用完所有特征后，仍然不能划分数据，则按照标签次数比例返回最多次的标签数据
        if len(dataSet[0]) == 1 :
            return Util.majorityCnt(classList)
        bestFeat = Util.getBestFeature(dataSet) #最佳数据增益列
        bestFeatLabel = labels[bestFeat] #最佳数据增益列标签
        myTree = {bestFeatLabel:{}} #树形结构数据
        del(labels[bestFeat]) #删除增益标签
        featVals = [col[bestFeat] for col in dataSet] #最佳数据增益列数据集
        uniqueVals = set(featVals) #最佳数据增益列数据集去重
        '''
        遍历去重的增益列数据
        获取增益列和增益数据对应的其他列数据集，并循环进行树形数据处理
        '''
        for val in uniqueVals:
            subLabels = labels[:]
            myTree[bestFeatLabel][val] = Tree.createTree(Util.getAxisValueDataSet(dataSet,bestFeat,val),subLabels)
        return myTree

    #树信息
    @classmethod
    def retrieveTree(self,i):
        listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                       {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head':{0: 'no', 1: 'yes'}}, 1: 'no'}}}}]
        return listOfTrees[i]

    #分类
    @classmethod
    def classify(self,inputTree,featLabels,testData):
        firstKey = list(inputTree)[0] #第一个分类标签
        secondDict = inputTree[firstKey] #第一个分类下一级的数据
        featIndex = featLabels.index(firstKey) #第一个分类标签所在索引
        '''
        遍历分类下一级的key(第一级分类的数据)
        key=测试数据在该标签索引下的数据值的情况：若分类数据不在有分支，则赋值分类标签，否则递归查询分类标签
        '''
        for firstKeyVal in secondDict.keys():
            if testData[featIndex] == firstKeyVal:
                if type(secondDict[firstKeyVal]).__name__ == 'dict':
                    classLabel = Tree.classify(secondDict[firstKeyVal],featLabels,testData)
                else:
                    classLabel = secondDict[firstKeyVal]
        return classLabel

if __name__ == '__main__':
    dataSet,labels = createDataSet()
    ent = Util.calcEnt(dataSet)
    # selDataSet = Util.getAxisValueDataSet(dataSet,0,0)
    # print(dataSet)
    # bestFeature = Util.getBestFeature(dataSet)
    # # print(bestFeature)
    treeLabels = labels[:]
    tree = Tree.createTree(dataSet,treeLabels)
    fileName = 'classifyStorage.txt'
    Util.saveTree(tree,fileName)
    treeStr = Util.grapTree(fileName)
    print(treeStr)
    # classTabel = Tree.classify(tree,labels,[0,0])
    # print(classTabel)
