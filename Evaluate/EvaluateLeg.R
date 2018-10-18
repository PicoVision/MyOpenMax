rm(list=ls())  
library(data.table)
library(xlsx)
library(ggplot2)
#This function is calculate AP per class
#input is Recall and Precision
VOCMap = function(rec,pre){
  MRec = c(0,rec,1)
  MPre = c(0,pre,0)
  
  for (i in (length(MPre)-1):1){
    Amax = max(MPre[i], MPre[i+1])
    MPre[i] = Amax
    #cat("xx->",i,"xx->",MPre[i],'\n')
  }
  
  ind = numeric()
  for (ind_1 in 2:(length(MRec))){
    if(MRec[ind_1] != MRec[ind_1-1]){
      ind = c(ind,ind_1)
    }
  }
  ap = 0.0
  for (i in ind){
    ap = ap+ ((MRec[i]-MRec[i-1])*MPre[i])
  }
  result = data.frame("Mrec"=MRec,"MPre"=MPre)
  
  return (list("ap"=ap,"data"=result))
}

Data_CloseSet <- read.csv("../Lecog_closeResult.txt",header = TRUE)
DataGT = Data_CloseSet[0:6]
plot(DataGT$eak)
