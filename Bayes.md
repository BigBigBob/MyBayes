
# 基于贝叶斯的分类方法 —— 朴素贝叶斯

### 一、使用朴素贝叶斯进行文档分类 

我们可以观察文档中出现的词，并把每个词的出现或者不出现作为一个特征，这样得到的特征数目就会跟词汇表中的词目一样多。朴素指的是特征之间相互独立。  

从文本中获得词条，一个词条是字符的任意组合。然后将一个文本片段表示为一个词条向量，其中值为1表示词条出现在文档中，0表示词条未出现。  

以在线社区留言板为例，建立两个类别：侮辱类和非侮辱类，分别用1和0表示。

#### 1.1、准备数据，从文本中构建词向量


```python
#加载事先准备好的数据
#返回结果：
#  词条切分后的文档集合
#  类别标签的集合
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 代表侮辱性文字，0代表正常言论
    return postingList,classVec
```


```python
#构造一个词汇表
#参数：词条切分后的文档集合
#返回结果：
#  文档集合中不重复的词汇表
def createVocabList(dataSet):
    vocabSet=set([])
    for document in dataSet:
        vocabSet=vocabSet | set(document)
    return list(vocabSet)
```


```python
#构建文档的词条向量
#参数：
#  词汇表
#  某个文档
#返回结果：
#  文档的词条向量
def setOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList) #以词汇表的长度为基准，构建0列表向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1 #出现则在词汇表对应的位置上标记为1
        else:
            print("the word:%s is not in my Vocabulary!" % word)
    return returnVec
```


```python
listOPosts,listClasses=loadDataSet()
myVocabList=createVocabList(listOPosts)
print(myVocabList)
```

    ['buying', 'stupid', 'I', 'has', 'problems', 'how', 'to', 'love', 'cute', 'garbage', 'please', 'posting', 'him', 'steak', 'not', 'take', 'ate', 'stop', 'licks', 'food', 'quit', 'is', 'flea', 'my', 'help', 'maybe', 'worthless', 'mr', 'park', 'so', 'dalmation', 'dog']



```python
print(setOfWords2Vec(myVocabList,listOPosts[0]))
```

    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1]


#### 1.2、朴素贝叶斯分类器的训练


```python
from numpy import *
```


```python
#训练分类器
#参数：
#  文档的词条向量矩阵
#  每篇文档的类别标签
#返回参数：
#   P(W|C=0)向量，P(W|C=1)向量，P(C=1)
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix) #获得矩阵行数，每一行代表一片文档的词条向量，即词条向量的数量
    numWords=len(trainMatrix[0]) #获得列数，即词条向量的数字的数量
    pAbusive=sum(trainCategory)/float(numTrainDocs) #侮辱文章出现的概率

    p0Num=zeros(numWords)
    p1Num=zeros(numWords)
    p0Denom=0.0;p1Denom=0.0

    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i]  #当类别标签为1时，各个特征出现的次数
            p1Denom+=sum(trainMatrix[i]) #总数
        else:
            p0Num+=trainMatrix[i]  #当类别标签为0时，各个特征出现的次数
            p0Denom+=sum(trainMatrix[i])

    p1Vect=p1Num/p1Denom  #P(W|C=1)向量
    p0Vect=p0Num/p0Denom  #P(W|C=0)向量
    return p0Vect,p1Vect,pAbusive
```


```python
listOPosts,listClasses=loadDataSet()
print(listOPosts)
print(listClasses)
```

    [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'], ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'], ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'], ['stop', 'posting', 'stupid', 'worthless', 'garbage'], ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'], ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    [0, 1, 0, 1, 0, 1]



```python
myVocabList=createVocabList(listOPosts) #构建了词表
print(myVocabList) 
```

    ['buying', 'stupid', 'I', 'has', 'problems', 'how', 'to', 'love', 'cute', 'garbage', 'please', 'posting', 'him', 'steak', 'not', 'take', 'ate', 'stop', 'licks', 'food', 'quit', 'is', 'flea', 'my', 'help', 'maybe', 'worthless', 'mr', 'park', 'so', 'dalmation', 'dog']



```python
trainMat=[]
for postinDoc in listOPosts:  #每篇训练文章的单词在词表中是否出现
    trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
print(trainMat)
```

    [[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]]



```python
p0V,p1V,pAb=trainNB0(trainMat,listClasses)
```


```python
print(pAb)
print(p0V)
print(p1V)
```

    0.5
    [0.         0.         0.04166667 0.04166667 0.04166667 0.04166667
     0.04166667 0.04166667 0.04166667 0.         0.04166667 0.
     0.08333333 0.04166667 0.         0.         0.04166667 0.04166667
     0.04166667 0.         0.         0.04166667 0.04166667 0.125
     0.04166667 0.         0.         0.04166667 0.         0.04166667
     0.04166667 0.04166667]
    [0.05263158 0.15789474 0.         0.         0.         0.
     0.05263158 0.         0.         0.05263158 0.         0.05263158
     0.05263158 0.         0.05263158 0.05263158 0.         0.05263158
     0.         0.05263158 0.05263158 0.         0.         0.
     0.         0.05263158 0.10526316 0.         0.05263158 0.
     0.         0.10526316]


#### 训练算法的改进
1、当计算P(w0|1)P(w1|1)P(w2|1)，如果其中一个概率值为0，那么最后的乘积也为0，为了减低这种影响。将所有词出现的次数初始化为1，并将分母初始化为2  
2、下溢出。对乘积取自然对数


```python
def trainNB01(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)  
    numWords=len(trainMatrix[0])  
    pAbusive=sum(trainCategory)/float(numTrainDocs) 

    p0Num=ones(numWords)
    p1Num=ones(numWords)
    p0Denom=2.0;p1Denom=2.0
    
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])

    p1Vect=log(p1Num/p1Denom)  #change to log()
    p0Vect=log(p0Num/p0Denom)  #change to log()
    return p0Vect,p1Vect,pAbusive
```


```python
p0V,p1V,pAb=trainNB01(trainMat,listClasses)
print(p0V)
print(p1V)
print(pAb)
```

    [-3.25809654 -3.25809654 -2.56494936 -2.56494936 -2.56494936 -2.56494936
     -2.56494936 -2.56494936 -2.56494936 -3.25809654 -2.56494936 -3.25809654
     -2.15948425 -2.56494936 -3.25809654 -3.25809654 -2.56494936 -2.56494936
     -2.56494936 -3.25809654 -3.25809654 -2.56494936 -2.56494936 -1.87180218
     -2.56494936 -3.25809654 -3.25809654 -2.56494936 -3.25809654 -2.56494936
     -2.56494936 -2.56494936]
    [-2.35137526 -1.65822808 -3.04452244 -3.04452244 -3.04452244 -3.04452244
     -2.35137526 -3.04452244 -3.04452244 -2.35137526 -3.04452244 -2.35137526
     -2.35137526 -3.04452244 -2.35137526 -2.35137526 -3.04452244 -2.35137526
     -3.04452244 -2.35137526 -2.35137526 -3.04452244 -3.04452244 -3.04452244
     -3.04452244 -2.35137526 -1.94591015 -3.04452244 -2.35137526 -3.04452244
     -3.04452244 -1.94591015]
    0.5


#### 1.3、测试算法


```python
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1=sum(vec2Classify*p1Vec)+log(pClass1)  #相加是因为之前已经取了对数，log(a*b)=loga+logb，所以采用求和的方式
    p0=sum(vec2Classify*p0Vec)+log(1.0-pClass1)
    if p1>p0:
        return 1
    else:
        return 0
```


```python
def testingNB():
    listOPosts,listClasses=loadDataSet() #初始化数据
    myVocabList=createVocabList(listOPosts) #获得词汇表
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))  #获得文档词条向量矩阵
    p0V,p1V,pAb=trainNB01(array(trainMat),array(listClasses))  #训练
    testEntry=['love','my','dalmation']  #测试文档数据
    thisDoc=array(setOfWords2Vec(myVocabList,testEntry)) #获得测试文档数据的词条向量
    print(testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb))
```


```python
testingNB()
```

    ['love', 'my', 'dalmation'] classified as: 0


#### 词汇袋


```python
def bagOfWords2VecMN(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
    return returnVec
```

### 二、使用朴素贝叶斯过滤垃圾邮件

#### 2.1 文件解析
获得切分后的字符串列表


```python
def textParse(bigString):
    import re
    listOfTokens=re.split(r'\w*',bigString)
    return [ tok.lower() for tok in listOfTokens if len(tok)>2]
```

#### 2.2 测试算法


```python
def spamTest():
    docList=[];classList=[];fullText=[]
    for i in range(1,26):  #从文件中获得字符串列表，并保存其对应的类别标签1,0
        wordList=textParse(open('email/spam/%d.txt' % i,encoding='ISO-8859-15').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList=textParse(open('email/ham/%d.txt' % i,encoding='ISO-8859-15').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList=createVocabList(docList)  #构建词汇表
    trainingSet=range(50);testSet=[]
    for i in range(10):   #构造随机的10个文件作为测试集
        randIndex=int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(list(trainingSet)[randIndex]) # del(trainingSet[randIndex]) - 'range' object doesn't support item deletion
    trainMat=[];trainClasses=[]
    for docIndex in trainingSet:  #剩下的文件作为训练集
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam=trainNB0(array(trainMat),array(trainClasses))  #以训练集来训练分类器
    errorCount=0
    for docIndex in testSet: #遍历测试集
        wordVector=setOfWords2Vec(vocabList,docList[docIndex]) #转换为词条向量
        if classifyNB(array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            errorCount+=1
    print('the error rate is:',float(errorCount)/len(testSet))
```


```python
spamTest()
```

    the error rate is: 0.1


    /root/anaconda3/lib/python3.6/re.py:212: FutureWarning: split() requires a non-empty pattern match.
      return _compile(pattern, flags).split(string, maxsplit)

