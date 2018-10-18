rm(list=ls())  
library(data.table)
library(xlsx)
library(ggplot2)
library(tidyr)
dev.off()
Cog_CloseSet <- read.csv("./Lecog_ResultCloseSet.txt",header = TRUE)
Cog_OpenSet <- read.csv("./Lecog_ResultOpenSet.txt",header = TRUE)
Cog_Hackset<- read.csv("./Lecog_ResultHackeset.txt",header = TRUE)

Cog_CloseSet["Class_index"] = "CloseSet"
Cog_OpenSet["Class_index"] = "OpenSet"
Cog_Hackset["Class_index"] = "HackSet"


Cog_CloseSet_CO = Cog_CloseSet[which(Cog_CloseSet$GT_Index==Cog_CloseSet$SoftMaxPredict),]
Cog_CloseSet_IN = Cog_CloseSet[which(Cog_CloseSet$GT_Index!=Cog_CloseSet$SoftMaxPredict),]

All_data  = data.frame()
All_data = rbind(All_data,Cog_CloseSet_CO,Cog_OpenSet,Cog_Hackset)

p <- ggplot(All_data, aes(x=factor(Class_index), y=eak_5))
p = p+ylim(low=0, high=1000000)
p = p+geom_violin()+ geom_boxplot(width=0.1) 
p


#library(tidyr) use tidyr to combile multicoulm to one column
Reformat_All_data = All_data[,c(1,2,3,4,15,16,17,18,19,20,21,22,23,24,45)] %>% gather(Formula_x, value, 5:14)

p <- ggplot(Reformat_All_data, aes(x=factor(Formula_x), y=value,colour=factor(Formula_x)))
p = p+ylim(low=0, high=500000)+theme(axis.text.x=element_text(angle=90, hjust=1))
p = p+geom_violin()+geom_boxplot(width=0.1)+facet_wrap(~ Class_index) 
p


p <- ggplot(Reformat_All_data, aes(x=factor(Class_index), y=value,colour=factor(Formula_x)))
p = p+ylim(low=0, high=500000)+theme(axis.text.x=element_text(angle=90, hjust=1))
p = p+geom_violin(width=1.5)+geom_boxplot(width=0.1)+facet_wrap(~ Formula_x) 
p





#Kruskal-Wallis rank sum test
name = "F10"

Va_Ty = ALL[c(2,3)]
CloseSet = All_data[All_data$Class_index=="CloseSet",]
OpenSet = All_data[All_data$Class_index=="OpenSet",]
HackSet = All_data[All_data$Class_index=="HackSet",]


summaryAll <- ddply(ALL, "Type", summarise, Value.sd=sd(Value),Value.max=max(Value),Value.min=min(Value),Value.mean=mean(Value),Value.median=median(Value))
summaryAll


library(plyr)
library(caret)
library(nortest)
library(scales)
print("Is normal??")
lillie.test(CloseSet$pow3Ak)
lillie.test(OpenSet$pow3Ak)
lillie.test(OpenSet$pow3Ak)

print("Non Para-test")


Va_Ty$Type = as.factor(All_data$Class_index)
kruskal.test(Value ~ Type, data = Va_Ty)

posthoc.kruskal.nemenyi.test(list(CP$Value,IP$Value,NS$Value), g=origin, dist='Tukey')



wilcox.test(CP$Value,IP$Value)
wilcox.test(CP$Value,NS$Value)
wilcox.test(NS$Value,IP$Value)


kruskalmc(Va_Ty$Value ~ Va_Ty$Type, probs=.05, cont=NULL)

