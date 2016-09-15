#coding=utf-8
'''
Created on 2013年11月11日

@author: Wangliaofan
'''
from numpy import *
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
#关闭交互模式，从而产生多个窗口
#plt.ion()

def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

#X,N*M
#RowY,1*M
#Kxy是N*1的列向量
def KernelTransfrom(X,RowY,KTpyeTuple):
    N=shape(X)[0]
    Kxy=mat(zeros((N,1)))#N*1
    if KTpyeTuple[0]=='linear' : 
        Kxy=X*RowY.T
    elif KTpyeTuple[0]=='rbf' or KTpyeTuple[0]=='gaussian' :
        for j in range(N):
            Row=X[j,:]-RowY
            Kxy[j]=Row*Row.T
        Kxy=exp(Kxy/(-1*KTpyeTuple[1]**2))
    else:
        raise  NameError('Unknown Kernel Type')      
    return Kxy

class optStruct(object):
    '''
    class doc
    '''
    def __init__(self,dataMatIn, classLabels, C, toler,KTpyeTuple):
        '''
        dataMatIn:    输入数据
        classLabels：    数据的标签
        C:            设置的上界
        toler:        容忍误差
        '''      
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]            #N*M,N为样本个数,M为样本的维数
        self.MofX = shape(dataMatIn)[1]
        self.alphas = mat(zeros((self.m,1)))    #N*1,初始化所有要训练的拉格朗日算子为0
        self.b = 0                              #初始化b为0
        self.eCache = mat(zeros((self.m,2)))    #N*2,第一列为是否有效的标志，第二列存误差
        self.K=mat(zeros((self.m,self.m)))  
        for i in range(self.m):
            self.K[:,i]=KernelTransfrom(self.X,self.X[i,:],KTpyeTuple)
        
def selectJrand(i,m):
    #在区间(0,m)选择和i不相等的随机数
    j=i
    while (j==i):
        j = int(random.uniform(0,m))
    return j

def clipAlpha(Alpha,H,L):
    #对应公式中修正Alpha的值
    if Alpha > H: 
        Alpha = H
    if L > Alpha:
        Alpha = L
    return Alpha
        
def calcEk(oS, k):
    #计算第k个样本的分类误差,即用判别函数计算出的值与标签值相减
    #w=sum(T*ALPHAS*X),y(k)=wT*x(k)+b
    #Error=y(k)-t(k),t表示标签
    #oS.X                             N*M
    #oS.X[k,:]                        1*M
    #multiply(oS.alphas,oS.labelMat)  N*1
    #fXk = float(multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T)) + oS.b
    fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k]) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJ(i, oS, Ei):
    #在选择了 Ei的基础上选择 Ej，即第二个启发式选择，思路很简单，即选择|Ei - Ek|有最大值的Ej 
    '''
    "Once a first Lagrange multiplier is chosen, 
    SMO chooses the second Lagrange multiplier to 
    maximize the size of the step taken during joint optimization"
    '''
    maxK = -1; maxDeltaE = 0; Ej = 0
    #set valid
    oS.eCache[i] = [1,Ei]
    #choose the alpha that gives the maximum delta E
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:   #loop through valid Error cache values and find the one that maximizes delta E
            if k == i: continue 
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:   
        return -1,-1


def updateEk(oS, k):
    #及时更新误差
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]
    
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        #启发式选择第二个要优化的拉格朗日算子
        j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand
        if j!=-1:
            if InsideLoop(i, oS, Ei, j, Ej)==1:
                return 1
        #启发式选择失败，则遍历非零和非C的拉格朗日算子
        #print 'heuristic failed' 
        jList=nonzero(oS.alphas[:].A)[0]
        if (len(jList)) > 1:
            for j in jList:
                #print 'current optimize',j
                if j==i or oS.alphas[j]>=oS.C: continue
                Ej = calcEk(oS, j)
                if InsideLoop(i, oS, Ei, j, Ej)==1:
                    return 1
        #遍历非零和非C的拉格朗日算子失败，那么遍历所有alpha，除了i对应的alpha
        #print 'non 0 and non C failed!!!!!!!!!!!!!!!!!!!!!!!!!!'
        
        while True:
            j = selectJrand(i, oS.m)
            if j!=i : break
        for idx_j in range(j,oS.m):
            if idx_j==i:continue
            Ej = calcEk(oS, idx_j)
            if InsideLoop(i, oS, Ei, idx_j, Ej)==1:
                return 1
        for idx_j in range(0,j-1):
            if idx_j==i:continue     
            Ej = calcEk(oS, idx_j)
            if InsideLoop(i, oS, Ei, idx_j, Ej)==1:
                return 1
        
        #print 'All failed! Try next............................'    
        return 0
    else: 
        return 0
    
def InsideLoop(i, oS, Ei, j, Ej):      
    alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        
    if (oS.labelMat[i] != oS.labelMat[j]):
        L = max(0, oS.alphas[j] - oS.alphas[i])
        H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
    else:
        L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
        H = min(oS.C, oS.alphas[j] + oS.alphas[i])
    if L==H: 
        #print "L==H"
        return 0
        
    #eta=k11+k22-2*k12
    #eta = oS.X[i,:]*oS.X[i,:].T + oS.X[j,:]*oS.X[j,:].T - 2.0 * oS.X[i,:]*oS.X[j,:].T
    eta = oS.K[i,i] + oS.K[j,j] - 2.0 * oS.K[i,j]
        
    if 0 == eta:
        #直线
        #print "eta=0"
        kgradient=oS.labelMat[j]*(Ei - Ej)
        if kgradient>=0:
            oS.alphas[j]=H
        else:
            oS.alphas[j]=L
    elif eta<0:
        #开口朝上的抛物线
        #print "eta<0"
        Xmiddle=0.5*(oS.alphas[j]+oS.labelMat[j]*(Ei - Ej)/eta)
        if Xmiddle>=0.5*(H+L):
            oS.alphas[j]=L
        else:
            oS.alphas[j]=H            
    else:
        #ai=ai_old+ti(Ej-Ei)/eta
        oS.alphas[j] += oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
            
    updateEk(oS, j) #added this for the Error cache
        
    #if (|a2-alph2| < eps*(a2+alph2+eps))
    eps=0.001
    stepOfj=abs(oS.alphas[j] - alphaJold)
    if (stepOfj < eps*(oS.alphas[j] + alphaJold + eps)): 
        #print "j not moving enough",stepOfj
        return 0
        
    oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
        
    updateEk(oS, i)
        
    #求b的值 
    #b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
    #b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
    b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
    b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
    
    if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): 
        oS.b = b1
    elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): 
        oS.b = b2
    else: 
        oS.b = (b1 + b2)/2.0
    return 1

#full John C. Platt SMO
def smoPlatt(dataMatIn, classLabels, C, toler, maxIter,KTpyeTuple):    
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler,KTpyeTuple)
    iternum = 0
    entireSet = True; alphaPairsChanged = 0
    
    while (iternum < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        #遍历所有样本
        if entireSet:   
            for i in range(oS.m):        
                alphaPairsChanged += innerL(i,oS)
                print "fullSet, iterate: %d i:%d, pairs changed %d" % (iternum,i,alphaPairsChanged)
            iternum += 1
        
        #遍历边界内样本
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print "non-bound, iterate: %d i:%d, pairs changed %d" % (iternum,i,alphaPairsChanged)
            iternum += 1
        if entireSet:
            #toggle entire set loop
            #再上一次完全遍历后，如果有alpha改变，这样只遍历边界内的样本 
            entireSet = False 
        elif (alphaPairsChanged == 0):
            #如果上一次完全遍历后，没有alpha改变，继续遍历整个样本，其中随机选取了新的起始样本点 
            entireSet = True  
        print "iteration number: %d" % iternum
        #计算线性情况下的W
        w=mat(zeros((oS.MofX,1))) 
        for i in range(oS.m):
            w+=multiply(oS.labelMat[i]*oS.alphas[i],oS.X[i,:].T)

    return oS.b,oS.alphas,w

def CalculateAccuracy(weights,dataArr,dataLabel):
    b = weights[2]; w0=weights[0]; w1=weights[1]
    DataNum=len(dataArr)
    ErrorCount=0.0
    
    for k in range(DataNum):
        Calculatelabel=(dataArr[k][0]*w0 + dataArr[k][1]*w1 +b)
        if Calculatelabel*dataLabel[k]<0:
            ErrorCount += 1
 
    return 1.0 - ErrorCount/float(DataNum),DataNum

def plotBestFit(weights,dataMat,labelMat): 
    dataArr = array(dataMat)
    n = shape(dataArr)[0] 
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,0]); ycord1.append(dataArr[i,1])
        else:
            xcord2.append(dataArr[i,0]); ycord2.append(dataArr[i,1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=20, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=20, c='green')
    x = arange(-2.0, 12.0, 0.1)
    y = (-weights[2]-weights[0]*x)/weights[1]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()
    
def PlotSupportVectors(dataArr,dataLabel,weights,alphas,C,KTpyeTuple):
    xcord0 = [];ycord0 = [];
    xcord1 = [];ycord1 = [];
    xmin=999999;xmax=-999999;ymin=999999;ymax=-999999;
    length=len(dataArr)
    for kk in range(0,length):
        
        if dataArr[kk][0]<xmin : xmin=dataArr[kk][0]
        if dataArr[kk][0]>xmax : xmax=dataArr[kk][0]
        if dataArr[kk][1]<ymin : ymin=dataArr[kk][1]
        if dataArr[kk][1]>ymax : ymax=dataArr[kk][1]
              
        if (dataLabel[kk] == -1):
            xcord0.append(dataArr[kk][0])
            ycord0.append(dataArr[kk][1])
        else:
            xcord1.append(dataArr[kk][0])
            ycord1.append(dataArr[kk][1]) 
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord0,ycord0, marker='s', s=90)
    ax.scatter(xcord1,ycord1, marker='o', s=50, c='red')
    ax.axis([xmin-2,xmax+2,ymin-2,ymax+2])
    plt.title('Support Vectors Circled')
    
    b = weights[2]; w0=weights[0]; w1=weights[1]
    
    validEcacheList = nonzero(alphas[:,0].A)[0]
    for k in validEcacheList:
        #TnYn=1-Eipsi
        #Eipsi=1-TnYn
        #Eipsi>1时说明该样本点错分，即outlier
        #Eipsi<=1时说明该样本点在margin内部
        if alphas[k,0]==C:
            Eipsi=1-dataLabel[k]*(dataArr[k][0]*w0 + dataArr[k][1]*w1 +b)
            if Eipsi<=1:
                circle = Circle((dataArr[k][0], dataArr[k][1]), 0.4, facecolor='none', edgecolor='yellow', linewidth=2, alpha=0.4)
                ax.add_patch(circle)
            else:
                circle = Circle((dataArr[k][0], dataArr[k][1]), 0.4, facecolor='none', edgecolor='red', linewidth=2, alpha=0.4)
                ax.add_patch(circle)
        #支持向量 
        else:
            circle = Circle((dataArr[k][0], dataArr[k][1]), 0.2, facecolor='none', edgecolor='green', linewidth=2, alpha=0.4)
            ax.add_patch(circle)
    
    #画分类超平面
    if KTpyeTuple[0]=='rbf' or KTpyeTuple[0]=='gaussian':
        plt.show()
        return
        dataMat=mat(dataArr);
        LabelMat=mat(dataLabel).transpose()
        svInd=nonzero(alphas.A>0)[0]
        svs=dataMat[svInd];
        labelsv=LabelMat[svInd]
        w_rbf=multiply(labelsv,alphas[svInd])
        #print svInd,svs,w_rbf
        x = arange(xmin-2.0, xmax+2.0, 0.1)
        y = arange(xmin-2.0, xmax+2.0, 0.1)
        xlength=shape(x)[0]
        matx=mat(zeros((xlength,2)))
        for i in range(xlength):
            matx[i,0]=x[i]
        for i in range(xlength):
            kernelEvaluate=KernelTransfrom(svs, matx[i,:], KTpyeTuple)
            y[i]=kernelEvaluate.T*w_rbf+b
        ax.plot(x,y)
        
    elif KTpyeTuple[0]=='linear':
        x = arange(xmin-2.0, xmax+2.0, 0.1)
        y = (-w0*x - b)/w1
        ax.plot(x,y)
        y1 = (-w0*x - b + 1)/w1
        ax.plot(x,y1)
        y2 = (-w0*x - b - 1)/w1
        ax.plot(x,y2)       
        plt.show()