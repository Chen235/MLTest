#决策树
import operator
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

    #求数据集的总熵
    @classmethod
    def calcShannonEnt(self,dataSet):
        numEntries = len(dataSet)
        labelCounts = {}
        for featVec in dataSet:
            currentLabel = featVec[-1]
            if currentLabel not in labelCounts.keys():
                labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1
        shannonEnt = 0
        for key in labelCounts:
            prob = float(labelCounts[key])/numEntries
            shannonEnt -= prob * log(prob,2)
        return shannonEnt

    @classmethod
    #获取指定列的Value数据集
    def getAxisValueDataSet(self,dataSet,axis,value):
        selDataSet = []
        for col in dataSet:
            if col[axis] == value:
                selCol = col[:axis]
                selCol.extend(col[axis + 1:])
                selDataSet.append(selCol)
        return selDataSet

    @classmethod
    #获取最佳信息增益列
    def getBestFeature(self,dataSet):
        dataLen = len(dataSet[0])-1
        ent = Util.calcShannonEnt(dataSet)
        bestInfoGain = 0
        bestFeature = -1
        for i in range(dataLen):
            featList = [col[i] for col in dataSet]
            uniqueVals = set(featList)
            newEnts = 0
            for val in uniqueVals:
                selDataSet = Util.getAxisValueDataSet(dataSet,i,val)
                prob = len(selDataSet)/float(len(dataSet))
                newEnts += prob*Util.calcShannonEnt(selDataSet)
            infoGain = ent - newEnts
            if infoGain > bestInfoGain:
                bestInfoGain = infoGain
                bestFeature = i
        return bestFeature

class Tree():

    @classmethod
    def madorityCnt(self,classList):
        classCount = {}
        for vote in classList:
            if vote not in classList.keys():classCount[vote] = 0
            classCount[vote] += 1
        sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
        return sortedClassCount[0][0]

    @classmethod
    def createTree(self,dataSet,labels):
        classList = [col[-1] for col in dataSet]
        if classList.count(classList[0]) == len(classList):
            return classList[0]
        if len(dataSet[0]) == 1 :
            return Tree.madorityCnt(classList)
        bestFeat = Util.getBestFeature(dataSet)
        bestFeatLabel = labels[bestFeat]
        myTree = {bestFeatLabel:{}}
        del(labels[bestFeat])
        featVals = [col[bestFeat] for col in dataSet]
        uniqueVals = set(featVals)
        for val in uniqueVals:
            subLabels = labels[:]
            myTree[bestFeatLabel][val] = Tree.createTree(Util.getAxisValueDataSet(dataSet,bestFeat,val),subLabels)
        return myTree

if __name__ == '__main__':
    dataSet,labels = createDataSet()
    shannonEnt = Util.calcShannonEnt(dataSet)
    selDataSet = Util.getAxisValueDataSet(dataSet,0,0)
    print(dataSet)
    bestFeature = Util.getBestFeature(dataSet)
    # print(bestFeature)
    tree = Tree.createTree(dataSet,labels)
    print(tree)
