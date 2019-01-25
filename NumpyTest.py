import numpy as np

#基本属性操作
def test1():
    data = [[1,2],[2,3]]
    x = np.array(data)
    print(x)
    print(x.dtype) #类型
    print(x.ndim) #维度
    print(x.shape) #各维度长度

#数组运算
def test2():
    data = [1,2,3,4,5,6]
    x = np.array(data)
    print(x)
    print(x*2)
    print(x+2)
    print(x>3)
    y = np.array([2,3,4,5,6,7])
    print(x*y)
    print(x+y)

#取值、赋值
def test3():
    data = [1, 2, 3, 4, 5, 6]
    x = np.array(data)
    print(x[0])
    print(x[0:2])
    print(x[:4])
    print(x[0:4:2])
    print('----------')
    data2 = [[1,2],[3,4],[5,6]]
    x = np.array(data2)
    print(x[:2, :1])  # [[1],[3]]
    x[:2, :1] = 0  # 用标量赋值
    print(x)  # [[0,2],[0,4],[5,6]]
    x[:2, :1] = [[8], [6]]  # 用数组赋值
    print(x)  # [[8,2],[6,4],[5,6]]

#布尔索引
def test4():
    x = np.array([0,2,1,4,0,3])
    y = np.array([True,False,True,False,False,True])
    print(x[y])
    print(x[y==False])
    print(x[(x>=2)])
    print((x==2)|(x==1))
    x[(x==2)|(x==1)] = 0
    print(x)

#整型索引
def test5():
    x = np.array([1,2,3,4,5,6])
    print(x[0:3])
    print(x[[0,1,2]])
    print(x[[-1,-2,-3]])
    print("-----------")
    x = np.array([[1,2],[3,4],[5,6]])
    print(x[[0,1]])
    print(x[[0,1],[0,1]])
    print(x[[0, 1]][:, [0, 1]]) # 打印01行的01列 [[1,2],[3,4]]
    # 使用numpy.ix_()函数增强可读性
    print(x[np.ix_([0, 1], [0, 1])])  # 同上 打印01行的01列 [[1,2],[3,4]]
    x[[0, 1], [0, 1]] = [0, 0]
    print(x)  # [[0,2],[3,0],[5,6]]

#转置和轴对称
def test6():
    print("-----维度转换------")
    k = np.arange(8) #1维
    print(k)
    m = k.reshape(2,2,2)#3维
    print(m)
    print("-----转置------")
    #转置 m[x][y][z] =m[z][y][x]
    print(m.T)
    print("-----内积------")
    #内积/点乘
    print(np.dot(m,m.T))
    print("-----轴变换------")
    # 高维数组的轴对象
    k = np.arange(8).reshape(2,2,2)
    print(k) # [[[0 1],[2 3]],[[4 5],[6 7]]]
    print(k[1][0][0])
    # 轴变换 transpose 参数:由轴编号组成的元组
    m = k.transpose((1,0,2)) # m[y][x][z] = k[x][y][z]
    print(m) # [[[0 1],[4 5]],[[2 3],[6 7]]]
    print(m[0][1][0])
    print("-----轴交换------")
    # 轴交换 swapaxes (axes：轴)，参数:一对轴编号
    m = k.swapaxes(0, 1)  # 将第一个轴和第二个轴交换 m[y][x][z] = k[x][y][z]
    print(m)  # [[[0 1],[4 5]],[[2 3],[6 7]]]
    print(m[0][1][0])
    # 使用轴交换进行数组矩阵转置
    m = np.arange(9).reshape((3, 3))
    print(m)  # [[0 1 2] [3 4 5] [6 7 8]]
    print(m.swapaxes(0, 1))  # [[0 3 6] [1 4 7] [2 5 8]]

#基本函数
def test7():
    x = np.arange(6)
    print(x)
    print(np.sqrt(x))
    print(np.square(x))
    x = np.array([0.5, 1.6, 1.7, 2.8])
    print(x)
    print(np.ceil(x))
    print(np.floor(x))
    print(np.modf(x))

#二元函数
def test8():
    x = np.array([[1, 4], [6, 7]])
    y = np.array([[2, 3], [5, 8]])
    print(np.add(x,y))
    print(np.multiply(x, y))
    print(np.maximum(x, y))  # [[2,4],[6,8]]
    print(np.minimum(x, y))  # [[1,3],[5,7]]

#where函数
def test9():
    cond = np.array([True,False,False,True])
    x = np.where(cond,1,-1)
    print(x)
    x = np.arange(6)
    print(x)
    print(np.where(x>2,1,-1))
    y1 = np.array([1,2,3,4,5,6])
    y2 = np.array([-1, -2, -3, -4, -5, -6])
    x = np.where(x>2,y1,y2)
    print(x)

#统计函数
def test10():
    x = np.array([[1,2],[2,3],[3,4]])
    print(np.mean(x)) #算数平均值
    print(x.mean(axis=1)) #对每一行求平均值
    print(x.mean(axis=0)) #对每一列求平均值
    print(np.sum(x))
    print(x.sum(axis=1))
    print(x.sum(axis=0))
    print(np.std(x)) #标准差
    print(np.var(x)) #方差
    print(np.min(x))
    print(np.max(x))
    print(np.cumsum(x)) #累积和
    print(np.cumprod(x)) #累计积

#布尔数据统计
def test11():
    x = np.array([[True, False], [False, False]])
    print(x.sum())  #统计某一维度中True元素个数
    print(x.sum(axis=1)) #行
    print(x.any(axis=0))  #统计某一个维度中是否存在一个或多个True元素
    print(x.all(axis=1))  #统计某一个维度中是否都是True

#排序
def test12():
    x = np.array([[1,5,2],[2,3,1],[8,2,4]])
    # print(np.sort(x))
    x.sort(axis=1)#行
    # x.sort(axis=0)#列
    print(x)

#去重、合并等集合运算
def test13():
    x = np.array([['a','c','b'],['e','d','a']])
    print(np.unique(x)) #去重并排序
    y = np.array([['a','e','f']])
    print(np.intersect1d(x,y)) #获取公共部分并排序
    print(np.union1d(x,y)) #获取并集并排序
    print(np.in1d(x,y)) #x的元素是否包含在y中的布尔型数组
    print(np.setdiff1d(x,y)) #x中有y中没有的元素并排序
    print(np.setxor1d(x,y)) #仅存在某一个集合中的元素并排序


#线性代数
import numpy.linalg as nla
def test14():
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[1,3],[2,4]])
    print(np.dot(x,y)) #内积
    x = np.array([[1, 1], [1, 2]])
    y = nla.inv(x) #矩阵求逆
    print(y)
    print(x.dot(y)) #单位矩阵
    print(nla.det(x)) #求行列式

#随机数
import numpy.random as npr
def test15():
    # nrd.seed(2)
    print(npr.permutation(5))
    print(npr.rand(10))
    x = npr.randint(0,2,10000)
    print((x>0).sum()) #抛硬币正反面
    print(npr.normal(size=(2,2)))

#数组拆分与合并
def test16():
    x = np.array([[1,2,3],[4,5,6]])
    y = np.array([[7,8,9],[10,11,12]])
    print(np.concatenate([x,y],axis=1)) #链接合并
    print(np.vstack((x,y))) #垂直/列合并
    print(np.hstack((x,y))) #水平/行合并
    # x = np.array([[1, 2, 3], [4, 5, 6],[7,8,9]])
    print(x)
    print(np.split(x, 2, axis=0)) #沿着列-横向拆分
    print(np.split(x,[0,1],axis=1)) #沿着行-纵向拆分
    print("------数组堆叠------")
    arr = np.arange(6)
    arr1 = arr.reshape((2, 3))
    arr2 = np.random.randn(2, 3)
    print(arr1)
    print(arr2)
    print(np.r_[arr1, arr2]) #按行堆叠(在行下堆叠)
    print(np.c_[arr1,arr2]) #按列堆叠(在列后堆叠)
    print(np.c_[1:6, -10:-5]) #切片直接转为数组

#数组重复操作
def test17():
    x = np.array([[1,2],[3,4]])
    print(x)
    print(x.repeat(2)) #元素重复2次
    print(x.repeat(2,axis=0)) #纵向重复两次
    print(x.repeat(2,axis=1)) #横向重复两次
    # x = np.array([1, 2])
    print(x)
    print(np.tile(x,2)) #元素顺序复制
    print(np.tile(x, (2, 3))) #指定从低维到高维复制次数



if __name__ == '__main__':
    test17()
