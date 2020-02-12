#[S2 Statistics Machine Learning - Project 2]
#perhatikan pagar2 di akhir baris


library(readxl)
library(neuralnet)
library(forecast)
library(lmtest)
library(tseries)
library(DataCombine)

setwd("D:/STATISTIKA/S2/COLLEGE/STAT MACHINE LEARNING/week_12") #setwdnya diganti ya :)
data_waduk_wonorejo <- read_excel("data waduk wonorejo.xlsx")
data_waduk_wonorejo <- data.frame(data_waduk_wonorejo)

y1t=as.ts(data_waduk_wonorejo[,4])
training=as.ts(y1t[1:228])   #ganti data training
testing=as.ts(y1t[229:252])  #ganti data testing


#diff seasonal
#Wtrain=diff(training,lag=12) #karena diff 12, jadi lag=12
#par(mfrow=c(1,1))
#plot(Wtrain)

#----PACF------
#ORDER IDENTIFICATION USING ACF AND PACF FROM STATIONARY DATA
#dari minitab lebih bagus sakjane
#tick=c(1,12,24,36)
#par(mfrow=c(1,1),mar=c(2.8,3,1.2,0.4))
#par(mgp=c(1.7,0.5,0))
#PACF
#pacf(Wtrain,lag.max=40,axes=F,ylim=c(-1,1))
#box()
#axis(side=1,at=tick,label=tick,lwd=0.5,las=0,cex.axis=0.8)
#abline(v=tick,lty="dotted", lwd=2, col="pink")
#axis(side=2,lwd=0.5,las=2,cex=0.5,cex.axis=0.8)
###############################################

#preprocessing
mean.Yt = mean(training)
sd.Yt   = sd(training)
Yt_std <- scale(y1t)
Yt_std1 <- as.data.frame(scale(y1t))

##======================menentukan input lag signifikan====================##

lag=c(1,3,5,12,13,15,17,24) #lag signifikan, ARIMA, stepwise
data1=slide(Yt_std1, Var = 'V1', slideBy = -lag)   
colnames(data1)=c("yt","yt1","yt3","yt5","yt12","yt13","yt15","yt17","yt24") #ganti sesuai di variabel lag
mlag=max(lag)+1
mtes=length(training)+1
data=data1[mlag:length(training),]
datatest=data1[mtes:length(y1t),]
head(data) #pastikan semua variabel punya colname

allvars=colnames(data)
predictorvarss=allvars[!allvars%in%"yt"]
predictorvarss=paste(predictorvarss,collapse = "+")
form=as.formula(paste("yt~",predictorvarss,collapse="+"));form #pastikan formulanya sesuai dg lag

###===========menentukan jumlah neuron===================###

neuron=c(1:10) #neuron yang akan digunakan
n_fore=24 #forecast berapa periode kedepan
rep = 10 #replikasi
#aa=c("rprop+","rprop-","sag","slr") #ganti yg mau dibandingkan apa (satu2, misal af aja, atau alg aja, cuma yg backprop blm aku ganti kayake)
aa = c("tanh","logistic")
af=length(aa)
best.model_NN=list
fits.model_NN=matrix(0,nrow(data),length(neuron)*rep*af)
fore.model_NN=matrix(0,n_fore,length(neuron)*af)
akutrain = matrix(0,rep,length(neuron)*af)
baris=length(training)-max(lag)

#note
#ketika di algorithm sag dan slr harus dicoba menggunakan learningrate.limit, 
#learningrate biasa di backprop
#syntax ini hanya bisa ngerun 3 algoritma (backprop, rprop+,rprop-)
#algoritma konstan dengan merubah neuron dan fungsi aktivasi

k1=0;mm=0;for (k in seq_along(neuron)){
  for (a1 in seq(aa)){
    set.seed(123456)
    mm <- mm+1
    best.model_NN=neuralnet(formula=form,data=data,hidden=neuron[k], algorithm = "rprop+",
                            act.fct=aa[a1],linear.output=TRUE,likelihood=TRUE, 
                            learningrate = 0.01,
                            err.fct="sse",
                            stepmax=100000,
                            rep=rep)
                            #kalo lg ngebandingkan af maka alg constant
    for (repl in 1:rep){
      fits.model=(as.ts(unlist(best.model_NN$net.result[[repl]])))*sd.Yt+mean.Yt
      fits.model_NN[,k1+repl]=fits.model
      akutrain[repl,mm]=accuracy(fits.model_NN[,k1+repl],training[1:baris])[1,2]
    }

    Ytest=c(data[,1],rep(0,n_fore))
    for(i in (nrow(data)+1):(nrow(data)+n_fore)){
      lagtest <- matrix(0,1,length(lag))
      for (m in 1:length(lag)){ lagtest[[m]] <- Ytest[(i-lag[m])]}
      input_pred <- cbind(lagtest)
      Ytest[i] <- compute(best.model_NN,input_pred, rep=which.min(best.model_NN$result.matrix[1,]))$net.result
      #punya hendri:Xtest=datatest[,2:ncol(datatest)]
      #punya hendri:Ytest[i] <- compute(best.model_NN,Xtest, rep=which.min(best.model_NN$result.matrix[1,]))$net.result
    }
    fore.model_NN[,mm] <- Ytest[(nrow(data)+1):(nrow(data)+n_fore)]*sd.Yt+mean.Yt
    k1 <- k1+rep
  }
  
}
#plot(best.model_NN,rep="best")
fore.model_NN #didapatkan testing dari setiap neuron di dua fungsi aktivasi 
akutrain #muncul setiap replikasi

#menghitung rata-rata setiap neuron di semua replikasi

e=0;meanakutrain = matrix(0,length(neuron),af)
akutest = matrix(0,length(neuron),af)
for (i in 1:length(neuron)){
  for (j in 1:af){
    meanakutrain[i,j] = mean(akutrain[,j+e])
    akutest[i,j]=accuracy(fore.model_NN[,j+e],as.vector(testing))[1,2]
  }
  e=e+af
}
colnames(meanakutrain)=c("train_tanh","train_log")
rownames(meanakutrain)=c("Neuron 1","Neuron 2","Neuron 3","Neuron 4","Neuron 5")
colnames(akutest)=c("test_tanh","test_log")
rownames(akutest)=rownames(meanakutrain)

akurasi = cbind(meanakutrain,akutest)
akurasi




#----ga penting sakjane-----

#MEMBUAT PLOT PERBANDINGAN DATA AKTUAL DAN RAMALAN SEMUA NEURON#
a=min(min(fits.model_NN),min(training[1:baris]))   #batas bawah plot data training
b=max(max(fits.model_NN),max(training[1:baris]))   #batas atas plot data training
c=min(min(fore.model_NN),min(testing))    #batas bawah plot data testing
d=max(max(fore.model_NN),max(testing))    #batas atas plot data testing
#colors()                                  #warna yang tersedia di R

par(mfrow=c(1,2),mar=c(2.3,2.7,1.2,0.4))  #banyaknya gambar dan ukuran margin
par(mgp=c(1.3,0.5,0))                     #jarak judul label ke axis
warna=c("red2","blue2","pink2","green3","grey88","yellow2","#ffa372","#ffa372","#2d334a","skyblue")

#PLOT DATA TRAINING#
plot(as.ts(training[1:baris]),ylab="Yt",xlab="t",lwd=2,axes=F,ylim=c(a*1.1,b*1.1))
box()
title("Data training",line=0.3,cex.main=0.9)
axis(side=2,lwd=0.5,cex.axis=0.8,las=2)
axis(side=1,lwd=0.5,cex.axis=0.8,las=0,at=c(1,seq(35,400,35)))
for (i in 1:length(neuron))
{lines(as.ts(fits.model_NN[,i]),col=warna[i],lwd=2)}

#PLOT DATA TESTING#
plot(as.ts(testing),ylab="Yt",xlab="t",lwd=2,ylim=c(c*1.1,d*1.2),cex.lab=0.8,axes=F)
box()
title("Data testing",line=0.3,cex.main=0.9)
axis(side=2,lwd=0.5,cex.axis=0.8,las=2)
axis(side=1,lwd=0.5,cex.axis=0.8,las=0,at=c(1:24),labels=c(145:168))
for (i in 1:length(neuron))
{lines(as.ts(fore.model_NN[,i]),col=warna[i],lwd=2)}

#MEMBERI NAMA LEGEND#
legend("topright",c("Data aktual","Neuron 1","Neuron 2","Neuron 3","Neuron 4",
                    "Neuron 5","Neuron 6","Neuron 7","Neuron 8","Neuron 9","Neuron 10"),
       col=c("black",warna),
       lwd=2,cex=0.7)

#MEMBUAT PLOT PERBANDINGAN DATA AKTUAL DAN RAMALAN#
a=min(min(fits.model_NN),min(training[1:baris]))   #batas bawah plot data training
b=max(max(fits.model_NN),max(training[1:baris]))   #batas atas plot data training
c=min(min(fore.model_NN),min(testing))    #batas bawah plot data testing
d=max(max(fore.model_NN),max(testing))    #batas atas plot data testing
#colors()                                  #warna yang tersedia di R

par(mfrow=c(2,1),mar=c(2.3,2.7,1.2,0.4))  #banyaknya gambar dan ukuran margin
par(mgp=c(1.3,0.5,0))                     #jarak judul label ke axis

#PLOT DATA TRAINING#
plot(as.ts(training[1:baris]),ylab="Yt",xlab="t",lwd=2,axes=F,ylim=c(a*1.1,b*1.5))
box()
title("Data training neuron 3",line=0.3,cex.main=0.9)
axis(side=2,lwd=0.5,cex.axis=0.8,las=2)
axis(side=1,lwd=0.5,cex.axis=0.8,las=0,at=c(1,seq(35,400,35)))
lines(as.ts(fits.model_NN[,3]),col="red2",lwd=2)
legend("topleft",c("Data aktual","Data ramalan"),
       col=c("black","red2"),lwd=2,cex=0.7)


#PLOT DATA TESTING#
plot(as.ts(testing),ylab="Yt",xlab="t",lwd=2,ylim=c(c*1.1,d*1.5),cex.lab=0.8,axes=F)
box()
title("Data testing neuron 3",line=0.3,cex.main=0.9)
axis(side=2,lwd=0.5,cex.axis=0.8,las=2)
axis(side=1,lwd=0.5,cex.axis=0.8,las=0,at=c(1:24),labels=c(145:168))
lines(as.ts(fore.model_NN[,3]),col="red2",lwd=2)
legend("topleft",c("Data aktual","Data ramalan"),
       col=c("black","red2"),lwd=2,cex=0.7)