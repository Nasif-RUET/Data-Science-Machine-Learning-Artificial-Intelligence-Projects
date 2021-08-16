## ==== ANN Classification for Climate Data G3_Data.csv =====#
rm(list=ls())

#install.packages("neuralnet")
library(neuralnet)

DT=read.csv("F:/Village of Study/PMASDS/PM-ASDS 22 MACHINELEARNING FOR DATA SCIENCE/Mymensingh.csv",header=TRUE)
DT[,c(1:4,10:11)]=NULL
head(DT)

DD=na.omit(DT,1)
head(DD)

#==Check that no data is missing ===#
##==HERE 1 for row, and 2 for column ===#
apply(DD,2,function(x) sum(is.na(x)))
#apply(DD,1,function(x) sum(is.na(x)))



Scale=function(x){(x - min(x))/(max(x) - min(x))}
SI=as.data.frame(lapply(DD[1:5], Scale))
head(SI)

D=cbind(SI,DD$RAN)
head(D)
names(D)[6]='RAN'
head(D)



set.seed(12357)
ID = sample(1:nrow(D), round(nrow(D)*0.75,0), replace=FALSE)

length(ID)
TR=D[ID,]   # Training Data
TS=D[-ID,]  # Test Data
nrow(TR)


nn1_TR=TR
nn1_TR=cbind(nn1_TR,	TR$RAN=="NRT")
nn1_TR=cbind(nn1_TR,	TR$RAN	=="LTR")
nn1_TR=cbind(nn1_TR,	TR$RAN	=="MHR")

head(nn1_TR)


names(nn1_TR)[7]="NRT"
names(nn1_TR)[8]="LTR"
names(nn1_TR)[9]="MHR"
head(nn1_TR[,7:9])
	
nn1	=neuralnet(NRT+LTR+MHR~TEM+DPT+WIS+HUM+SLP,
           data=nn1_TR,	hidden=c(4,3),stepmax = 1e+50,
           algorithm = "rprop+", err.fct = "sse",
           act.fct = "logistic", linear.output = FALSE)
print(nn1)
plot(nn1)


PR=compute(nn1, TS[-6])
PR1=PR$net.result

round(PR1,2)

PR_ID = function(x){return(which(x==max(x)))}
IDX	=apply(PR1, c(1), PR_ID)

PR_nn=c('NRT', 'LTR','MHR')[IDX]
T=table(TS$RAN,PR_nn)
T


ATS=sum(diag(T))/sum(T)
ATS
1-ATS


