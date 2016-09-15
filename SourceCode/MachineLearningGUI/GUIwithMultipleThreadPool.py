#coding=utf-8
'''
Created on 2014年3月22日

@author: Wangliaofan
'''

from PyQt4 import QtCore, QtGui
import sys
import numpy
import Regression
import OriginSVM
import MixGaussian
import copy
import MLLoadDataSet
import BasicNeuralNetwork
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle

QtCore.QTextCodec.setCodecForTr(QtCore.QTextCodec.codecForName("utf8"))  

#截获控制台输出
class EmittingStream(QtCore.QObject):
    textWritten = QtCore.pyqtSignal(str)
    def write(self, text):
        self.textWritten.emit(str(text))
        
class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None, width=10, height=8, dpi=100):
        #fig = Figure(figsize=(width, height), dpi=dpi)
        fig = Figure()
        FigureCanvas.__init__(self, fig)
        self.axes = fig.add_subplot(111)
        # We want the axes cleared every time plot() is called
        self.axes.hold(True)
        self.compute_initial_figure()
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,QtGui.QSizePolicy.Expanding,QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass
#3D绘制窗口，派生于MyMplCanvas
class MyMplCanvas3D(MyMplCanvas):
    def __init__(self, parent=None, width=10, height=8, dpi=100):
        #fig = Figure()
        fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, fig)
        #Axes3D.mouse_init()
        self.axes =  Axes3D(fig)  
        # We want the axes cleared every time plot() is called
        self.axes.hold(True)
        self.HasPlot=False
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,QtGui.QSizePolicy.Expanding,QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_3d_figure(self,data,label,x=0,y=1,z=2):
        xs, ys, zs = [], [], []
        xs1, ys1, zs1 = [], [], []
        for i in range(len(data)):
            if 1.0==label[i]:
                xs.append(data[i][x])
                ys.append(data[i][y])
                zs.append(data[i][z])
            else:
                xs1.append(data[i][x])
                ys1.append(data[i][y])
                zs1.append(data[i][z])
        self.axes.scatter(xs, ys, zs, zdir='z', s=40, c=(0,0,1))
        self.axes.scatter(xs1, ys1, zs1, zdir='z', s=40, c=(0,1,0)) 
        self.draw()
        self.HasPlot=True
        return
#2D绘制窗口，派生于MyMplCanvas   
class MyDynamicMplCanvas(MyMplCanvas):
    # """A canvas that updates itself every second with a new plot."""
    def __init__(self, *args, **kwargs):
        MyMplCanvas.__init__(self, *args, **kwargs)
        #timer = QtCore.QTimer(self)
        #QtCore.QObject.connect(timer, QtCore.SIGNAL("timeout()"), self.update_figure)
        #timer.start(1000)
        self.HasPlot=False
        
    def compute_initial_figure(self):
        #self.axes.plot([0, 1, 2, 3], [1, 2, 0, 4], 'r')
        return
    #绘制回归结果
    def update_regession_figure(self,dataMat,labelMat,weights):
        if True==self.HasPlot:
            self.axes.cla()
        weight=weights.getA()
        dataArr = numpy.array(dataMat)
        n = numpy.shape(dataArr)[0] 
        xcord1 = []; ycord1 = []
        xcord2 = []; ycord2 = []
        for i in range(n):
            if int(labelMat[i])== 1:
                xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
            else:
                xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
        self.axes.scatter(xcord1, ycord1, s=30, c='red', marker='s')
        self.axes.scatter(xcord2, ycord2, s=30, c='green')
        x = numpy.arange(-3.0, 3.0, 0.1)
        y = (-weight[0]-weight[1]*x)/weight[2]
        self.axes.plot(x, y)
        
        #self.xlabel('X1'); self.ylabel('X2');
        #plt.xlabel('X1'); plt.ylabel('X2')
        self.axes.set_xlabel("X1"); self.axes.set_ylabel('X2')
        self.draw()
        self.HasPlot=True
        return
    #绘制SVM分类结果
    def update_figure(self,PlotType,dataArr,dataLabel,weights,alphas,C,KTypeTuple):
        if 'SVM'==PlotType:
            if True==self.HasPlot:
                self.axes.cla()
            # Build a list of 4 random integers between 0 and 10 (both inclusive)
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
            self.axes.scatter(xcord0,ycord0, marker='s', s=90)
            self.axes.scatter(xcord1,ycord1, marker='o', s=50, c='red')
            self.axes.axis([xmin-2,xmax+2,ymin-2,ymax+2])
            #plt.title('Support Vectors Circled')
            b = weights[2]; w0=weights[0]; w1=weights[1] 
            validEcacheList = numpy.nonzero(alphas[:,0].A)[0]
            for k in validEcacheList:
                if alphas[k,0]==C:
                    Eipsi=1-dataLabel[k]*(dataArr[k][0]*w0 + dataArr[k][1]*w1 +b)
                    if Eipsi<=1:
                        circle = Circle((dataArr[k][0], dataArr[k][1]), 0.4, facecolor='none', edgecolor='yellow', linewidth=2, alpha=0.4)
                        self.axes.add_patch(circle)
                    else:
                        circle = Circle((dataArr[k][0], dataArr[k][1]), 0.4, facecolor='none', edgecolor='red', linewidth=2, alpha=0.4)
                        self.axes.add_patch(circle)
                #支持向量 
                else:
                    circle = Circle((dataArr[k][0], dataArr[k][1]), 0.2, facecolor='none', edgecolor='green', linewidth=2, alpha=0.4)
                    self.axes.add_patch(circle)
            #画分类超平面
            if KTypeTuple[0]=='linear':
                x = numpy.arange(xmin-2.0, xmax+2.0, 0.1)
                y = (-w0*x - b)/w1
                self.axes.plot(x,y)
                self.draw()
                y1 = (-w0*x - b + 1)/w1
                self.axes.plot(x,y1)
                y2 = (-w0*x - b - 1)/w1
                self.axes.plot(x,y2)
            ####################################################       
            self.axes.set_xlabel("X1"); self.axes.set_ylabel('X2')
            self.draw()
            self.HasPlot=True
        return
    #绘制混合高斯结果
    def update_gaussian_figure(self,data,Pik,Uk,ListCovk):
        if True==self.HasPlot:
            self.axes.cla()
        Xn=numpy.mat(data)
        N = numpy.shape(Xn)[1]
        xcord1 = []; ycord1 = []

        for i in range(N): 
            p1=Pik[0,0]*MixGaussian.NDimensionGaussian(Xn[:,i],Uk[:,0],ListCovk[0])
            p2=Pik[0,1]*MixGaussian.NDimensionGaussian(Xn[:,i],Uk[:,1],ListCovk[1])
            #print p1,p2
            c1=float(p1)/(float(p1)+float(p2))
            c2=float(p2)/(float(p1)+float(p2))
            #print c1,c2
            xcord1.append(Xn[0,i]); ycord1.append(Xn[1,i])
            self.axes.scatter(xcord1[i], ycord1[i], s=100, c=(c2,c1,0), marker='o')
        
        #plt.xlabel('X1'); plt.ylabel('X2')
        self.axes.set_xlabel("X1"); self.axes.set_ylabel('X2')
        self.draw()
        self.HasPlot=True
        return
    def update_ArtificialNeuralNetwork_figure(self,dataMatIn,Predict):
        if True==self.HasPlot:
            self.axes.cla()
        Xn= numpy.array(dataMatIn)
        N = numpy.shape(Xn)[0]
    
        xcord1 = []; ycord1 = []
        xcord2 = []; ycord2 = []
        for i in range(N):
            #print Predict[0,i]
            if Predict[0,i]>= 0.5:
                xcord1.append(Xn[i,0]); ycord1.append(Xn[i,1])
            else:
                xcord2.append(Xn[i,0]); ycord2.append(Xn[i,1])
        self.axes.scatter(xcord1, ycord1, s=30, c='red', marker='o')
        self.axes.scatter(xcord2, ycord2, s=30, c='green', marker='o')
        self.axes.set_xlabel("X1"); self.axes.set_ylabel('X2')
        self.draw()
        self.HasPlot=True
        return
    
class WorkerSignals(QtCore.QObject):
    result = QtCore.pyqtSignal(str)
 
#采用多线程实现SVM防止界面假死
class MultiThreadSVM(QtCore.QRunnable):
    #trigger = QtCore.pyqtSignal(str) # trigger传输的内容是字符串
    def __init__(self, dynamicplot,currentData,currentLabel,para_C, Tol, MaxIter,KernelType,parent=None):  
        super(MultiThreadSVM, self).__init__()       
        self.currentData = copy.deepcopy(currentData)
        self.currentLabel = copy.deepcopy(currentLabel)
        self.para_C = copy.deepcopy(para_C)
        self.Tol = copy.deepcopy(Tol)
        self.MaxIter = copy.deepcopy(MaxIter)
        self.KernelType = copy.deepcopy(KernelType)
        self.plotarea=dynamicplot
        self.signals = WorkerSignals()
    #线程完成的工作
    def run(self): #Overwrite run() method, put what you want the thread do here  
        b,alphas,w=OriginSVM.smoPlatt(self.currentData,self.currentLabel,\
                                      self.para_C, self.Tol, self.MaxIter,self.KernelType)   
        a=[]
        a.append(w[0,0]);a.append(w[1,0]);a.append(b[0,0])
        self.signals.result.emit("w1:"+QtCore.QString("%1").arg(float(a[0]))+"\r\n"+\
                                "w2:"+QtCore.QString("%1").arg(float(a[1]))+"\r\n"+\
                                "b:"+QtCore.QString("%1").arg(float(a[2]))+"\r\n")
        
        self.plotarea.update_figure('SVM',self.currentData,self.currentLabel,a,alphas,self.para_C,self.KernelType)
        #有问题还得改系数的维数不一样
        weights=[]
        weights.append(w[0,0]);weights.append(w[1,0]);weights.append(b[0,0])
        accuracy,DataNum=OriginSVM.CalculateAccuracy(a,self.currentData,self.currentLabel)
        self.signals.result.emit("Total Data Number:"+QtCore.QString("%1").arg(DataNum)+"\r\n")
        self.signals.result.emit("Accuracy:"+QtCore.QString("%1").arg(accuracy)+"\r\n")
        return
#采用多线程实现混合高斯EM防止界面假死
class MultiThreadGaussian(QtCore.QRunnable):
    #trigger = QtCore.pyqtSignal(str) # trigger传输的内容是字符串
    def __init__(self, dynamicplot,currentData,K,MaxIter,parent=None):  
        super(MultiThreadGaussian, self).__init__()       
        self.InputData = copy.deepcopy(currentData)
        self.K = copy.deepcopy(K)
        self.MaxIter = copy.deepcopy(MaxIter)
        self.plotarea=dynamicplot
        self.signals = WorkerSignals()
    #线程完成的工作
    def run(self): #Overwrite run() method, put what you want the thread do here  
        pi_new,UK_new,List_cov=MixGaussian.BatchEMforMixGaussian(self.InputData,self.K,self.MaxIter)
        #print pi_new
        pi_size=numpy.shape(pi_new)[1]
        for pi_index in range(pi_size):
            self.signals.result.emit("pi"+ QtCore.QString("%1").arg(pi_index) + ":"+QtCore.QString("%1").arg(float(pi_new[0,pi_index]))+"\r\n")
        D,K=numpy.shape(UK_new)
        #print UK_new
        for Kiter in range(K):
            self.signals.result.emit("Uk"+ QtCore.QString("%1").arg(Kiter) + ": ")
            for Diter in range(D):
                self.signals.result.emit(QtCore.QString("%1").arg(float(UK_new[Diter,Kiter])) + "     ")
            self.signals.result.emit("\r\n")
        #print List_cov
        self.signals.result.emit("\r\n")
        for Liter in range(len(List_cov)):
            D1,D2=numpy.shape(List_cov[Liter])
            if D1==D2:
                self.signals.result.emit("Cov_k"+ QtCore.QString("%1").arg(Liter) + ": ")
                for D1iter in range(D1):
                    for D2iter in range(D2):
                        self.signals.result.emit( QtCore.QString("%1").arg(float(List_cov[Liter][D1iter,D2iter])) + "     ")
                    self.signals.result.emit("\r\n"+ "        ")
                self.signals.result.emit("\r\n")
                
        #self.plotarea.update_figure(self.currentData,self.currentLabel,a,alphas,self.para_C,self.KernelType)
        self.plotarea.update_gaussian_figure(self.InputData,pi_new,UK_new,List_cov)       
        return
    
#采用多线程实现基本人工神经网络防止界面假死
class MultiThreadBasicNeuralNetwork(QtCore.QRunnable):
    def __init__(self, dynamicplot,currentData,currentLabel,i,j,k,maxCycles,parent=None):
        super(MultiThreadBasicNeuralNetwork, self).__init__()
        self.InputData = currentData
        self.Label = currentLabel
        self.I = i
        self.J = j
        self.K = k
        self.MaxIter = maxCycles
        self.plotarea= dynamicplot
        self.signals = WorkerSignals()
    def run(self):
        W1,W2=BasicNeuralNetwork.BPTrainNetwork(self.InputData,self.Label,self.I,self.J,self.K,self.MaxIter)
        Predict,count=BasicNeuralNetwork.NetworkPredict(self.InputData,self.Label,W1,W2)
        self.plotarea.update_ArtificialNeuralNetwork_figure(self.InputData,Predict)
        self.signals.result.emit("Accuracy:"+str(count))
        return
          
#获得SVM参数的对话框    
class GetSVMInfo(QtGui.QDialog):
    def __init__(self,parent=None):
        super(GetSVMInfo, self).__init__()
        self.positive=False
        mainLayout = QtGui.QVBoxLayout()
        self.para1edit=QtGui.QLineEdit()
        self.para1edit.setText("0.5")       #设置默认C
        self.para2edit=QtGui.QComboBox()
        self.para2edit.addItem("linear")
        self.para2edit.addItem("rbf")
        #self.para2edit.addItem("gaussian")
        self.para3edit=QtGui.QLineEdit()
        self.para3edit.setText("0.01")      #设置默认误差tolerance
        self.para4edit=QtGui.QLineEdit()
        self.para4edit.setText("500")       #设置默认最大迭代次数
        
        self.formGroupBox = QtGui.QGroupBox("SVM Parameters...")
        
        Formlayout = QtGui.QFormLayout()
        Formlayout.addRow(QtGui.QLabel("Parameter: C"), self.para1edit)
        Formlayout.addRow(QtGui.QLabel("Kernel Type"), self.para2edit)
        Formlayout.addRow(QtGui.QLabel("Tolerance"), self.para3edit)
        Formlayout.addRow(QtGui.QLabel("Max Iterate Number"), self.para4edit)
        
        self.formGroupBox.setLayout(Formlayout)
        mainLayout.addWidget(self.formGroupBox)
        
        buttonBox = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(self.setok)
        buttonBox.rejected.connect(self.reject)

        mainLayout.addWidget(buttonBox)
        self.setLayout(mainLayout)
        self.setWindowTitle("SVM Parameter InputDialog")
        
    def returnpara(self): 
        self.para_C=self.para1edit.text()
        self.para_KernelType=self.para2edit.currentText()
        self.para_Tolerance=self.para3edit.text()
        self.para_Num=self.para4edit.text()
        return self.para_C,self.para_KernelType,self.para_Tolerance,self.para_Num
    
    def setok(self):
        self.positive=True
        self.accept()
        return
        
class Dialog(QtGui.QMainWindow):
    def __init__(self):
        super(Dialog, self).__init__()
        sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)
        sys.stderr = EmittingStream(textWritten=self.errorOutputWritten)
        #初始化部分重要变量
        self.currentfileName=""
        #self.threads = [] # this will keep a reference to threads

        #for i in range(20):
            #self.threads.append("threads"+str(i))
        #self.threadcount=0
        
        #利用QThreadPool实现多线程
        self.pool = QtCore.QThreadPool()
        self.pool.setMaxThreadCount(20)
        
        self.appName="Machine Learning Demo by Shuai"
        self.setWindowTitle(self.appName)  
            
        self.DataView = QtGui.QTableWidget(100000,1000)
        self.createDataView()    
        
        self.menuBar = QtGui.QMenuBar()
        self.createMenu()
        #self.setMenuBar(self.menuBar)
        self.creatToolBar()
        
        self.setMenuWidget(self.menuBar)

        #停靠窗口2  
        dock2=QtGui.QDockWidget(self.tr("数据显示窗口"),self)  
        dock2.setFeatures(QtGui.QDockWidget.DockWidgetFloatable|QtGui.QDockWidget.DockWidgetClosable)    
        dock2.setWidget(self.DataView)  
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea,dock2) 
        
        self.bigEditor = QtGui.QTextEdit()
        pl_bE=self.bigEditor.palette()
        pl_bE.setColor(QtGui.QPalette.Text, QtGui.QColor(0,0,255))
        self.bigEditor.setPalette(pl_bE)
        self.bigEditor.setPlainText("Output Information!")
        #停靠窗口1  
        dock1=QtGui.QDockWidget(self.tr("计算输出信息"),self)  
        dock1.setFeatures(QtGui.QDockWidget.AllDockWidgetFeatures)  
        dock1.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea|QtCore.Qt.RightDockWidgetArea)    
        dock1.setWidget(self.bigEditor)  
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea,dock1)  
        
        self.main_widget_plot = QtGui.QWidget(self)
        self.dynamicplot = MyDynamicMplCanvas(self.main_widget_plot, width=5, height=4, dpi=100)      
        #停靠窗口3  
        dock3=QtGui.QDockWidget(self.tr("图像显示区域(MatPlotLib)"),self)  
        dock3.setFeatures(QtGui.QDockWidget.AllDockWidgetFeatures)  
        dock3.setWidget(self.dynamicplot)  
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea,dock3) 
        
        self.main_widget_plot_3d = QtGui.QWidget(self)
        self.dynamicplot_3d = MyMplCanvas3D(self.main_widget_plot_3d, width=5, height=4, dpi=100)      
        #停靠窗口4  
        dock4=QtGui.QDockWidget(self.tr("3D图像显示区域(MatPlotLib)"),self)  
        dock4.setFeatures(QtGui.QDockWidget.AllDockWidgetFeatures)  
        dock4.setWidget(self.dynamicplot_3d)  
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea,dock4) 
        
        
        jpeg=QtGui.QPixmap(self)
        jpeg.load("icon/background.jpg")
        palette1 = QtGui.QPalette(self)
        palette1.setBrush(self.backgroundRole(), QtGui.QBrush(jpeg))        
        
        self.text=QtGui.QTextEdit()
        pl2=self.text.palette()
        pl2.setBrush(QtGui.QPalette.Base,QtGui.QBrush(QtGui.QColor(255,0,0,0)))
        self.text.setPalette(pl2)
        self.text.setTextColor(QtGui.QColor(255,165,0))
        
        mainLayout = QtGui.QHBoxLayout()
        mainLayout.addWidget(self.text)
        centerWidget=QtGui.QWidget()
        
        centerWidget.setLayout(mainLayout)
        centerWidget.setPalette(palette1)
        centerWidget.setAutoFillBackground(True)
      
        self.setCentralWidget(centerWidget)  
    ############################################################# 
    #def __del__(self):
        # Restore sys.stdout
        #sys.stdout = sys.__stdout__
        #sys.stderr = sys.__stderr__     
    #############################################################    
    def createMenu(self):
        self.fileMenu = QtGui.QMenu("&File Operation", self)
        self.DataMenu = QtGui.QMenu("&Data Operation", self)    
        self.RegressionMenu = QtGui.QMenu("&Regression", self)
        self.SvmMenu = QtGui.QMenu("&SVM", self)
        self.MixGaussianMenu = QtGui.QMenu("&MixGaussian", self)
        self.WindowMenu = QtGui.QMenu("&Window", self)
        self.ArtificialNetworkMenu = QtGui.QMenu("&Neural Network", self)
        
        self.menuBar.addMenu(self.fileMenu)
        self.menuBar.addMenu(self.DataMenu)
        self.menuBar.addMenu(self.RegressionMenu)
        self.menuBar.addMenu(self.SvmMenu)
        self.menuBar.addMenu(self.MixGaussianMenu)
        self.menuBar.addMenu(self.ArtificialNetworkMenu)
        self.menuBar.addMenu(self.WindowMenu)


        self.exitAction = self.fileMenu.addAction(QtGui.QIcon("icon/11.ico"),"&Exit",self.close,QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.ShowInfoAction = self.fileMenu.addAction(QtGui.QIcon("icon/6.ico"),"&About")
        self.LoadDataAction = self.DataMenu.addAction(QtGui.QIcon("icon/21.ico"),"&LoadData")
        self.RegressionAction = self.RegressionMenu.addAction(QtGui.QIcon("icon/19.ico"),"&LogisticRegression")
        self.CSvmAction = self.SvmMenu.addAction(QtGui.QIcon("icon/14.ico"),"&C-SVM")
        self.MixGaussianAction = self.MixGaussianMenu.addAction(QtGui.QIcon("icon/5.ico"),"&EM Mix Gaussian")
        self.NeuralNetWorkAction = self.ArtificialNetworkMenu.addAction(QtGui.QIcon("icon/28.ico"),"&BP NetWork")

        
        self.connect(self.exitAction,QtCore.SIGNAL("triggered()"),self.slotExist)
        self.connect(self.LoadDataAction,QtCore.SIGNAL("triggered()"),self.slotOpenFile)
        self.connect(self.RegressionAction,QtCore.SIGNAL("triggered()"),self.LogisticRegression)
        self.connect(self.CSvmAction,QtCore.SIGNAL("triggered()"),self.CSVM)
        self.connect(self.MixGaussianAction,QtCore.SIGNAL("triggered()"),self.EmMixGaussian)
        self.connect(self.NeuralNetWorkAction,QtCore.SIGNAL("triggered()"),self.ArtificialNeuralNetwork)
        return
    
    def slotExist(self):
        self.close(self.exitAction)
        return
    
    def creatToolBar(self):
        fileToolBar=self.addToolBar("File")
        fileToolBar.addAction(self.exitAction)
        #fileToolBar.addAction(self.LoadDataAction)
        fileToolBar.addAction(self.ShowInfoAction)
        dataToolBar=self.addToolBar("Data")
        dataToolBar.addAction(self.LoadDataAction)
        OperationToolBar=self.addToolBar("Operation")
        OperationToolBar.addAction(self.RegressionAction)
        OperationToolBar.addAction(self.MixGaussianAction)
        OperationToolBar.addAction(self.NeuralNetWorkAction)
        return
    #打开数据文件，并在   self.DataView 中显示
    def slotOpenFile(self):
        self.currentfileName=QtGui.QFileDialog.getOpenFileName(self)
        if self.currentfileName=="":
            self.bigEditor.setPlainText("No File Selected!!!")
            return
        self.setWindowTitle(self.appName+" - "+self.currentfileName)
        self.bigEditor.setPlainText(self.currentfileName+"\r\n")
        #获得数据的数量和维数并在提示信息栏显示
        try:
            self.currentData,self.currentLabel=OriginSVM.loadDataSet(self.currentfileName)
        except:
            self.currentData,self.currentLabel,DataDimension=MLLoadDataSet.LoadLibData(self.currentfileName)
            if DataDimension>=3:
                self.dynamicplot_3d.compute_3d_figure(self.currentData,self.currentLabel,2,3,4)
                
        DataHeight,DataWidth=numpy.shape(self.currentData)
        self.bigEditor.insertPlainText("Data Number: "+QtCore.QString("%1").arg(DataHeight)+"\r\n")
        self.bigEditor.insertPlainText("Data Dimension: "+QtCore.QString("%1").arg(DataWidth)+"\r\n")
        
        for Dh in range(DataHeight):
            for Dw in range(DataWidth):
                tempString = QtCore.QString("%1").arg(self.currentData[Dh][Dw])
                currentItem = QtGui.QTableWidgetItem(tempString)
                self.DataView.setItem(Dh,Dw,currentItem)
        
            tempString = QtCore.QString("%1").arg(self.currentLabel[Dh])
            currentItem = QtGui.QTableWidgetItem(tempString)
            self.DataView.setItem(Dh,Dw+1,currentItem)
        return
    
    #创建    DataView
    def createDataView(self):
        
        return
    
    def LogisticRegression(self):
        if (""==self.currentfileName):
            self.bigEditor.insertPlainText("No data Loaded! \r\n")
            return
        else:
            self.currentData,self.currentLabel=Regression.loadDataSet(self.currentfileName)
        #self.currentData,self.currentLabel=Regression.loadDataSet(self.currentfileName)
        #newweights是numpy的矩阵
        newweights=Regression.IRLSmethod(self.currentData,self.currentLabel)
        m=numpy.shape(newweights)[0]
        self.bigEditor.insertPlainText("Weights:\r\n")
        for i in range(m):
            self.bigEditor.insertPlainText(QtCore.QString("%1").arg(float(newweights[i])) + "\r\n")
        #Regression.plotBestFit(newweights)
        self.dynamicplot.update_regession_figure(self.currentData,self.currentLabel, newweights)  
        return
    
    def CSVM(self):
        if (""==self.currentfileName):
            self.bigEditor.insertPlainText("No data Loaded! \r\n")
            return   
        SVMpara=GetSVMInfo()
        SVMpara.exec_()
        go_on_flag=SVMpara.positive
        
        if (False==go_on_flag):
            self.bigEditor.insertPlainText("SVM Cancel!"+"\r\n")
            return
        
        para_C,Kernel,Tol,MaxIter=SVMpara.returnpara()
        para_C=str(para_C);para_C=float(para_C)
        Kernel=str(Kernel);
        Tol=str(Tol);Tol=float(Tol)
        MaxIter=str(MaxIter);MaxIter=int(MaxIter)
        
        self.bigEditor.insertPlainText("Parameter C:"+QtCore.QString("%1").arg(para_C)+"\r\n"+\
                                       "Kernel Type:"+Kernel+"\r\n"+\
                                       "Tolerance:"+QtCore.QString("%1").arg(Tol)+"\r\n"+\
                                       "Max Iterate Number:"+QtCore.QString("%1").arg(MaxIter)+"\r\n")

        KernelType=[Kernel,2.0]
        self.currentData,self.currentLabel=OriginSVM.loadDataSet(self.currentfileName)
        worker=MultiThreadSVM(self.dynamicplot,self.currentData,self.currentLabel,para_C, Tol, MaxIter,KernelType)
        #b,alphas,w=OriginSVM.smoPlatt(self.currentData,self.currentLabel,para_C, Tol, MaxIter,KernelType)
        worker.signals.result.connect(self.update_text)  # connect to it's signal
        self.pool.start(worker)
        #self.pool.waitForDone()
        return
    
    #?????????????????????????????????
    def EmMixGaussian(self):
        if (""==self.currentfileName):
            self.bigEditor.insertPlainText("No data Loaded! \r\n")
            return
        self.currentData=MixGaussian.loadDataSet(self.currentfileName)
        worker=MultiThreadGaussian(self.dynamicplot,self.currentData, 2, 300)
        worker.signals.result.connect(self.update_text)
        self.pool.start(worker)
        #self.pool.waitForDone()
        return
    
    def ArtificialNeuralNetwork(self):
        if (""==self.currentfileName):
            self.bigEditor.insertPlainText("No data Loaded! \r\n")
            return
        self.currentData,self.currentLabel=BasicNeuralNetwork.loadDataSet(self.currentfileName)
        worker=MultiThreadBasicNeuralNetwork(self.dynamicplot,self.currentData,self.currentLabel,2,3,1,1000)
        worker.signals.result.connect(self.update_text)
        self.pool.start(worker)
        return
    
    def update_text(self, message):
        self.bigEditor.moveCursor(QtGui.QTextCursor.End)
        self.bigEditor.insertPlainText(message) # use insertPlainText to prevent the extra newline character
        
        return
    
    def normalOutputWritten(self, text):
        """Append text to the QTextEdit."""
        # Maybe QTextEdit.append() works as well, but this is how I do it:
        cursor = self.text.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.text.setTextCursor(cursor)
        self.text.ensureCursorVisible()
        return
    
    def errorOutputWritten(self, text):
        """Append text to the QTextEdit."""
        # Maybe QTextEdit.append() works as well, but this is how I do it:
        cursor = self.text.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.text.setTextCursor(cursor)
        self.text.ensureCursorVisible()
        return
    
if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    dialog = Dialog()
    dialog.show()
    sys.exit(app.exec_())
    pass