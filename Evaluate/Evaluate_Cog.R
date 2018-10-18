rm(list=ls())  
library(data.table)
library(xlsx)
library(ggplot2)
dev.off()
Cog_CloseSet <- read.csv("./Lecog_ResultCloseSet.txt",header = TRUE)
Cog_OpenSet <- read.csv("./Lecog_ResultOpenSet.txt",header = TRUE)
Cog_Hackset<- read.csv("./Lecog_ResultHackeset.txt",header = TRUE)

Cog_CloseSet["Class_index"] = "CloseSet"
Cog_OpenSet["Class_index"] = "OpenSet"
Cog_Hackset["Class_index"] = "HackSet"

Cog_OpenSet = Cog_OpenSet[1:50000,]

Cog_CloseSet = Cog_CloseSet[which(Cog_CloseSet$GT_Index==Cog_CloseSet$SoftMaxPredict),]

All_data  = data.frame()
All_data = rbind(All_data,Cog_CloseSet,Cog_OpenSet,Cog_Hackset)

p <- ggplot(All_data, aes(x=Class_index, y=eak)) + geom_boxplot()+ylim(low=0, high=100000)
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

