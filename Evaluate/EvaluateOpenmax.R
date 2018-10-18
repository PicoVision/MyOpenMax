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


Data_CloseSet <- read.csv("./ResultCloseSet.txt",header = TRUE)
Data_OpenSet <- read.csv("./ResultOpenSet.txt",header = TRUE)

Data_OpenSet = Data_OpenSet[1:15000,]    #Force select 15k dataset

#merge Closeset and Openset
Data_all  = rbind(Data_CloseSet, Data_OpenSet) 
Data_result = data.frame()
Data_mAP = data.frame()
# Threshold Loop
for(threshold in seq(0,0.4,by=0.05)){
  
  
  Data_select = copy(Data_all)
  
  #cutoff by threshold if some Probability prediction is lower than threshold then set to 1000 that is unseen class 
  Data_select[which(Data_select$Prob_Of_Openmax<threshold),"OpenMaxPredict"] = 1000
  
  #sortData order by class and prob of openmax
  SortData_Class_byOpenmax = Data_select[with(Data_select, order(-OpenMaxPredict, -Prob_Of_Openmax)), ]
  
  #extract allClass of openmax
  openmaxClass = unique(Data_select[,"OpenMaxPredict"])
  openmaxClass = openmaxClass[order(openmaxClass)]
  
  #select set new parameter
  SortData_Class_byOpenmax["TP"] = 0
  SortData_Class_byOpenmax["FP"] = 0
  SortData_Class_byOpenmax["CTP"] = 0
  SortData_Class_byOpenmax["CFP"] = 0
  SortData_Class_byOpenmax["Pre"] = 0
  SortData_Class_byOpenmax["Rec"] = 0
  
  #if GT doesn't match prediction set FP =1 and if GT match with prediction set TP = 1
  SortData_Class_byOpenmax[which(SortData_Class_byOpenmax$GT_Index == SortData_Class_byOpenmax$OpenMaxPredict),c("TP")] = 1
  SortData_Class_byOpenmax[which(SortData_Class_byOpenmax$GT_Index != SortData_Class_byOpenmax$OpenMaxPredict),c("FP")] = 1
  
  #select openmax which predict interesting class
  
  AP =0
  for (i in 0:1000){
    #cat("Class : ",i,"\n")
    predict_class_selected = 
      copy(SortData_Class_byOpenmax[
        which(SortData_Class_byOpenmax$OpenMaxPredict ==i),])
    
    
    #cumulative sum of TP
    predict_class_selected = within(predict_class_selected, CTP <- cumsum(TP))
    
    #cumulative sum of FP
    predict_class_selected = within(predict_class_selected, CFP <- cumsum(FP))
    
    # Total Number object in GT
    TotalGT = nrow(SortData_Class_byOpenmax[which(SortData_Class_byOpenmax$GT_Index ==i),])
    
    predict_class_selected = within(predict_class_selected, Pre <- CTP/(CTP+CFP))
    predict_class_selected = within(predict_class_selected, Rec <- CTP/TotalGT)
    
    mPre = predict_class_selected$Pre
    mRec = predict_class_selected$Rec
    
    #plot(mRec,mPre,type="l")
    result = VOCMap(mRec,mPre)
    result_detail = result$data
    result_detail["class"] = i
    result_detail["threshold"] = threshold
    
    #show example plot
    #plot(mRec,mPre,type="l",lty=2)
    #lines(result_detail$Mrec,result_detail$MPre)
    
    
    AP = AP + result$ap
    Data_result = rbind.data.frame(Data_result,result_detail)
    #cat("class = ",i," AP= ",result$ap, "map = ",AP/(i+1),"\n")
  }
  print(AP/1001)
  Data_mAP = rbind(Data_mAP,data.frame("mAP"=AP/1001,"Threshold"=threshold))
}
write.csv(Data_mAP,"result_OMap.csv")
write.csv(Data_result,"result_OAP.csv")
#read Data section
Data_OmAP = read.csv("result_OMap.csv")
Data_OmAP["class"] = "Openmax"

Data_result_AP = read.csv("result_OAP.csv")
p = ggplot(Data_OmAP,aes(x=Threshold,y=mAP))+geom_point(color='blue')
p  


Data_SmAP = read.csv("result_SMap.csv")
Data_SmAP["class"] = "Softmax"
Data_map = rbind(Data_OmAP,Data_SmAP)

p = ggplot(Data_map,aes(x=Threshold,y=mAP,group=class))+
  geom_point(aes(color=class))
p 
