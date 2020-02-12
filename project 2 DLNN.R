library(neuralnet)
library(forecast)
library(lmtest)
library(tseries)
library(readxl)
library(DataCombine)

data_waduk_wonorejo <- read_excel("D:/STATISTIKA/S2/COLLEGE/STAT MACHINE LEARNING/week_12/data waduk wonorejo.xlsx")
dat=data_waduk_wonorejo[,4:5]
dat=data_waduk_wonorejo[,4:5]
colnames(dat)=c("Y","X")
training=as.ts(dat[1:228,1])
testing=as.ts(dat[229:252,1])
mean.Yt = mean(training)
sd.Yt   = sd(training)
Y <- as.ts(dat$Y)
Yt_std <- as.data.frame(scale(Y))

#############################################################
#########################################################################


#PREPROCESSING STANDARDIZED#
lag=c(1,3,5,12) #masukkan lag pacf signifikan 
lag1=lag+12 
data1=slide(Yt_std, Var = 'V1', slideBy = -lag)   
data2=slide(Yt_std, Var = 'V1', slideBy = -lag1)
data3=cbind(data1,data2[,-1])
data=data1[max(lag)+1:(228-max(lag)),]
#data=cbind(data2[max(lag1)+1:(228-max(lag1)),1],data)
data=data.frame(data)
colnames(data)=c("yt","V1.1","V1.3","V1.5","V1.12","V1.13","V1.15","V1.17","V1.24")
yt=data[,1]
datatest=data2[229:252,]

#PREPROCESSING STANDARDIZED#
lag=c(1,3,5,12) #masukkan lag pacf signifikan 
lag1=lag+12 
data1=slide(Yt_std, Var = 'V1', slideBy = -lag)   
data2=slide(Yt_std, Var = 'V1', slideBy = -lag1)
data3=cbind(data1,data2[,-1])
data=data3[max(lag1)+1:(228-max(lag1)),]
#data=cbind(data2[max(lag1)+1:(228-max(lag1)),1],data)
data=data.frame(data)
colnames(data)=c("yt","V1.1","V1.3","V1.5","V1.12","V1.13","V1.15","V1.17","V1.24")
yt=data[,1]
datatest=data3[229:252,]

#MEMBENTUK MODEL NEURAL NETWORK#
X <- t(as.matrix(expand.grid(1:4, 1:2)))
neuron=as.data.frame(cbind(X[1,],X[2,]))
n_fore=24
seed=c(1,3,3,1,5,2,2)                #berdasarkan percobaan beberapa set.seed yang berbeda

best.model_NN=list
fits.model_NN=matrix(0,nrow(data),nrow(neuron))
fore.model_NN=matrix(0,n_fore,nrow(neuron))

allvars=colnames(data)
predictorvarss=allvars[!allvars%in%"yt"]
predictorvarss=paste(predictorvarss,collapse = "+")
form=as.formula(paste("yt~",predictorvarss,collapse="+"))

alg=c("backprop","rprop+","rprop-","sag","slr")
for (k in 1:nrow(neuron))
{
  #set.seed(1)
  best.model_NN=neuralnet(formula=form,data=data,hidden=c(neuron[k,1],neuron[k,2]),
                          linear.output=TRUE,
                          #act.fct="tanh",
                          #learningrate=0.01,
                          algorithm=alg[2],err.fct="sse",likelihood=TRUE)
  best.model_NN[[k]]=best.model_NN
  fits.model_NN[,k]=(as.ts(unlist(best.model_NN[[k]]$net.result)))*sd.Yt+mean.Yt  #hasil ramalan data training
  
  #ARSITEKTUR NEURAL NETWORK#
  #win.graph()
  #plot(best.model_NN[[k]])
  
  #FORECAST k-STEP AHEAD#
  Ytest=c(data[,1],rep(0,n_fore))
  for(i in (nrow(data)+1):(nrow(data)+n_fore))
  {
    Xtest=datatest[,2:ncol(datatest)]
    Ytest[i]=compute(best.model_NN[[k]],Xtest)$net.result[i-nrow(data)]
  }
  fore.model_NN[,k]=Ytest[(nrow(data)+1):(nrow(data)+n_fore)]*sd.Yt+mean.Yt       #hasil ramalan data testing
}


#MEMBERI NAMA KOLOM UNTUK MATRIKS HASIL FORECAST#
colnames(fore.model_NN)=c("Neuron 1","Neuron 2","Neuron 3","Neuron 4",
                          "Neuron 5","Neuron 6","Neuron 7","Neuron 8","neuron9","neuron10") 

#MENGHITUNG TINGKAT KESALAHAN PERAMALAN#
akurasi=matrix(0,nrow(neuron),6)
colnames(akurasi)=c("RMSE_training","MAE_training","MAPE_training",
                    "RMSE_testing","MAE_testing","MAPE_testing")
rownames(akurasi)=colnames(fore.model_NN)
trainingnew=training[max(lag1)+1:(228-max(lag1)),]
#yhat=as.data.frame(yhat)
datatest=data.frame(datatest)
for (i in 1:nrow(neuron))
{
  akurasi[i,1]=accuracy(as.ts(fits.model_NN[,i]),trainingnew)[1,2]
  akurasi[i,2]=accuracy(as.ts(fits.model_NN[,i]),trainingnew)[1,3]
  akurasi[i,3]=accuracy(as.ts(fits.model_NN[,i]),trainingnew)[1,5]
  akurasi[i,4]=accuracy(fore.model_NN[,i],datatest[,1])[1,2]
  akurasi[i,5]=accuracy(fore.model_NN[,i],datatest[,1])[1,3]
  akurasi[i,6]=accuracy(fore.model_NN[,i],datatest[,1])[1,5]
}
akurasi


#MEMBUAT PLOT PERBANDINGAN RMSE, MAE, MAPE#
RMSE_a=min(c(akurasi[,1],akurasi[,4]))        #batas bawah plot RMSE
RMSE_b=max(c(akurasi[,1],akurasi[,4]))        #batas atas plot RMSE
MAE_a=min(c(akurasi[,2],akurasi[,5]))         #batas bawah plot MAE
MAE_b=max(c(akurasi[,2],akurasi[,5]))         #batas atas plot MAE
MAPE_a=min(c(akurasi[,3],akurasi[,6]))        #batas bawah plot MAPE
MAPE_b=max(c(akurasi[,3],akurasi[,6]))        #batas atas plot MAPE

par(mfrow=c(3,1),mar=c(2.7,2.9,1.2,0.4))  #banyaknya gambar dan ukuran margin
par(mgp=c(1.7,0.5,0))                     #jarak judul label ke axis

#RMSE
plot(as.ts(akurasi[,1]),ylab="RMSE",xlab="Neuron",lwd=2,axes=F,ylim=c(RMSE_a*1.1,RMSE_b*1.1))
box()
axis(side=2,lwd=0.5,cex.axis=0.8,las=2)
axis(side=1,lwd=0.5,cex.axis=0.8,las=0,at=c(1:10),labels=neuron)
lines(akurasi[,4],col="red2",lwd=2)
legend("topright",c("Data training","Data testing"),
       col=c("black","red2"),
       lwd=2,cex=0.7)
#MAE
plot(as.ts(akurasi[,2]),ylab="MAE",xlab="Neuron",lwd=2,axes=F,ylim=c(MAE_a*1.1,MAE_b*1.1))
box()
axis(side=2,lwd=0.5,cex.axis=0.8,las=2)
axis(side=1,lwd=0.5,cex.axis=0.8,las=0,at=c(1:10),labels=neuron)
lines(akurasi[,5],col="red2",lwd=2)
legend("topright",c("Data training","Data testing"),
       col=c("black","red2"),
       lwd=2,cex=0.7)
#MAPE
plot(as.ts(akurasi[,3]),ylab="MAPE",xlab="Neuron",lwd=2,axes=F,ylim=c(MAPE_a*1.1,MAPE_b*1.1))
box()
axis(side=2,lwd=0.5,cex.axis=0.8,las=2)
axis(side=1,lwd=0.5,cex.axis=0.8,las=0,at=c(1:10),labels=neuron)
lines(akurasi[,6],col="red2",lwd=2)
legend("topright",c("Data training","Data testing"),
       col=c("black","red2"),
       lwd=2,cex=0.7)

#MEMBUAT PLOT PERBANDINGAN DATA AKTUAL DAN RAMALAN SEMUA NEURON#
a=min(min(fits.model_NN),min(trainingnew))   #batas bawah plot data training
b=max(max(fits.model_NN),max(trainingnew))   #batas atas plot data training
c=min(min(fore.model_NN),min(testing))    #batas bawah plot data testing
d=max(max(fore.model_NN),max(testing))    #batas atas plot data testing
#colors()                                  #warna yang tersedia di R

par(mfrow=c(1,2),mar=c(2.3,2.7,1.2,0.4))  #banyaknya gambar dan ukuran margin
par(mgp=c(1.3,0.5,0))                     #jarak judul label ke axis
warna=c("red2","blue2","pink2","green3","grey88","yellow2","skyblue")

#PLOT DATA TRAINING#
plot(as.ts(trainingnew),ylab="Yt",xlab="t",lwd=2,axes=F,ylim=c(a*1.1,b*1.1))
box()
title("Data training",line=0.3,cex.main=0.9)
axis(side=2,lwd=0.5,cex.axis=0.8,las=2)
axis(side=1,lwd=0.5,cex.axis=0.8,las=0,at=c(1,seq(35,400,35)))
for (i in 1:nrow(neuron))
{lines(as.ts(fits.model_NN[,i]),col=warna[i],lwd=2)}

#PLOT DATA TESTING#
plot(as.ts(testing),ylab="Yt",xlab="t",lwd=2,ylim=c(c*1.1,d*1.2),cex.lab=0.8,axes=F)
box()
title("Data testing",line=0.3,cex.main=0.9)
axis(side=2,lwd=0.5,cex.axis=0.8,las=2)
axis(side=1,lwd=0.5,cex.axis=0.8,las=0,at=c(1:24),labels=c(145:168))
for (i in 1:nrow(neuron))
{lines(as.ts(fore.model_NN[,i]),col=warna[i],lwd=2)}

#MEMBERI NAMA LEGEND#
legend("topright",c("Data aktual","Neuron 1","Neuron 2","Neuron 3","Neuron 4",
                    "Neuron 5","Neuron 6","Neuron 7","Neuron 8"),
       col=c("black",warna),
       lwd=2,cex=0.7)