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

checkCloseset = function(x){
  label = "Close"
  if(x>=1000){
    label = "Open" 
  }
  return (label)
}


Data_CloseSet <- read.csv("../Result_CloseSet.txt",header = TRUE)
Data_OpenSet <- read.csv("../Result_OpenSet.txt",header = TRUE)

#Evaluate With ACC
cat("Evaluate with ACC")
length(which(Data_CloseSet$GT_Index == Data_CloseSet$SoftMaxPredict))/50000


Data_OpenSet = Data_OpenSet[1:15000,]    #Force select 15k dataset

#merge Closeset and Openset
Data_all  = rbind(Data_CloseSet, Data_OpenSet) 
Data_result = data.frame()
Data_ACC = data.frame()
# Threshold Loop
threshold = 0.9
for(threshold in seq(0,0.4,by=0.01)){
  
  Data_select_closeset = copy(Data_CloseSet)
  Data_select_openset = copy(Data_OpenSet)
  #cutoff by threshold if some Probability prediction is lower than threshold then set to 1000 that is unseen class 
  Data_select_closeset[which(Data_select_closeset$Prob_Of_Openmax<threshold),"OpenMaxPredict"] = 1000
  Data_select_openset[which(Data_select_openset$Prob_Of_Openmax<threshold),"OpenMaxPredict"] = 1000
  
  
  #select set new parameter
  Data_select_closeset["TP"] = 0
  Data_select_closeset["FP"] = 0
  Data_select_closeset["TN"] = 0
  Data_select_closeset["FN"] = 0
  Data_select_closeset["CTP"] = 0
  Data_select_closeset["CFP"] = 0
  Data_select_closeset["Pre"] = 0
  Data_select_closeset["Rec"] = 0
  
  
  #select set new parameter
  Data_select_openset["TP"] = 0
  Data_select_openset["FP"] = 0
  Data_select_openset["TN"] = 0
  Data_select_openset["FN"] = 0
  Data_select_openset["CTP"] = 0
  Data_select_openset["CFP"] = 0
  Data_select_openset["Pre"] = 0
  Data_select_openset["Rec"] = 0
  
  
  #remap closetset
  Data_select_closeset <- within(Data_select_closeset, GT_Re <- (Data_select_closeset$GT_Index>=1000)*1000)  
  Data_select_closeset <- within(Data_select_closeset, ST_Re <- (Data_select_closeset$OpenMaxPredict>=1000)*1000)  
  
  #remap openset
  Data_select_openset <- within(Data_select_openset, GT_Re <- (Data_select_openset$GT_Index>=1000)*1000)  
  Data_select_openset <- within(Data_select_openset, ST_Re <- (Data_select_openset$OpenMaxPredict>=1000)*1000)  
  
  
  
  #Evaluate closetset
  Data_select_closeset[Data_select_closeset$GT_Re == Data_select_closeset$ST_Re,c("TN")] = 1
  Data_select_closeset[Data_select_closeset$GT_Re != Data_select_closeset$ST_Re,c("FN")] = 1
  
  Data_select_openset[Data_select_openset$GT_Re == Data_select_openset$ST_Re,c("TP")] = 1
  Data_select_openset[Data_select_openset$GT_Re != Data_select_openset$ST_Re,c("FP")] = 1
  
  
  
  #calculate ACC
  sTN = sum(Data_select_closeset$TN) 
  sFN = sum(Data_select_closeset$FN)
  sTP = sum(Data_select_openset$TP) 
  sFP = sum(Data_select_openset$FP)
  ACC = (sTP+sTN)/(sTP+sTN+sFP+sFN)
  
  cat("ACC = ",ACC,"\n")
  Data_ACC = rbind(Data_ACC,data.frame("mAP"=ACC,"Threshold"=threshold))
}
write.csv(Data_ACC,"result_OPenACC.csv")
#read Data section
Data_OACC = read.csv("result_OPenACC.csv")
Data_OACC["class"] = "OpenMax"
p = ggplot(Data_OACC,aes(x=Threshold,y=mAP))+geom_point(color='blue')
p  


Data_SACC = read.csv("result_SoftACC.csv")
Data_SACC["class"] = "SoftMax"

Data_ACC = rbind(Data_OACC,Data_SACC)
p = ggplot(Data_ACC,aes(x=Threshold,y=mAP),group=class)+geom_point(aes(color=class))
p = p+geom_line(aes(color=class))
p
