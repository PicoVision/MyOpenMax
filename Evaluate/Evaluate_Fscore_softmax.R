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


Data_CloseSet <- read.csv("./ResultCloseSet.txt",header = TRUE)
Data_OpenSet_raw <- read.csv("./ResultOpenSet.txt",header = TRUE)
Data_HackImage_raw <- read.csv("./ResultHackset.txt",header = TRUE)


#Evaluate With ACC
cat("Evaluate with ACC")
length(which(Data_CloseSet$GT_Index == Data_CloseSet$SoftMaxPredict))/50000
Data_result = data.frame()
for (Stepdataamount in seq(30000,75000,by=15000)){
  
  #shuffle openSet
  Data_OpenSet = Data_OpenSet_raw[sample(nrow(Data_OpenSet_raw)),]
  Data_OpenSet = Data_OpenSet[1:Stepdataamount,]    #Force select 15k first
  
  Data_ACC = data.frame()
  # Threshold Loop
  threshold = 0.9
  for(threshold in seq(0,0.9,by=0.05)){
    
    Data_select_closeset = copy(Data_CloseSet)
    Data_select_openset = copy(Data_OpenSet)
    Data_select_Hacked = copy(Data_HackImage_raw)
    
    #cutoff by threshold if some Probability prediction is lower than threshold then set to 1000 that is unseen class 
    Data_select_closeset[which(Data_select_closeset$Prob_Of_Softmax<threshold),"SoftMaxPredict"] = 1000
    Data_select_openset[which(Data_select_openset$Prob_Of_Softmax<threshold),"SoftMaxPredict"] = 1000
    Data_select_Hacked[which(Data_select_Hacked$Prob_Of_Softmax<threshold),"SoftMaxPredict"] = 1000
    
    
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
    
    #select set new parameter
    Data_select_Hacked["TP"] = 0
    Data_select_Hacked["FP"] = 0
    Data_select_Hacked["TN"] = 0
    Data_select_Hacked["FN"] = 0
    Data_select_Hacked["CTP"] = 0
    Data_select_Hacked["CFP"] = 0
    Data_select_Hacked["Pre"] = 0
    Data_select_Hacked["Rec"] = 0
    
    
    #closetset
    Data_select_closeset[Data_select_closeset$SoftMaxPredict == Data_select_closeset$GT_Index,c("TP")]=1
    Data_select_closeset[Data_select_closeset$SoftMaxPredict != Data_select_closeset$GT_Index,c("FP")]=1
    
    
    #remap openset
    Data_select_openset[Data_select_openset$SoftMaxPredict != Data_select_openset$GT_Index,c("FN")]=1
    Data_select_Hacked[Data_select_Hacked$SoftMaxPredict != Data_select_Hacked$GT_Index,c("FN")]=1
    
    
    #calculate ACC
    sTP = sum(Data_select_closeset$TP) 
    sFP = sum(Data_select_closeset$FP)
    sFN = sum(Data_select_openset$FN)+sum(Data_select_Hacked$FN)
    P = sTP / (sTP+sFP)
    R1 = sTP/(sTP+sFN)
    R2 = sTP/(sTP+sFN+sFP)
    R3 = sTP/65000
    F1 = 2*(P*R1)/(P+R1)
    F2 = 2*(P*R2)/(P+R2)
    F3 = 2*(P*R3)/(P+R3)
    
    cat("F1 = ",F1,"F2 = ",F2,"F3 = ",F3,"\n")
    Data_ACC = rbind(Data_ACC,data.frame("F1"=F1,"F2"=F2,"F3"=F3,  "Threshold"=threshold,"OpensetElement" = Stepdataamount))
  }
  Data_result = rbind(Data_result,Data_ACC)
}

write.csv(Data_result,"result_sFscore.csv")
#read Data section
Data_result = read.csv("result_sFscore.csv")

p = ggplot(Data_result,aes(x=Threshold,y=F1,group=OpensetElement))+geom_point(aes(color=OpensetElement))
p  





