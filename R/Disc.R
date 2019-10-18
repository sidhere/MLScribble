library(ggplot2)
library(readr)
library(rpart)
library(randomForest)
library(magrittr)
library(dplyr)
library(tm)
library(stringr)
library("recommenderlab")
library(DAAG)
library(readxl)
Disc.frame <- read_excel("C:/ML/data/Disc.xlsx", 
                          col_types = c("blank", "text", "text", 
                                          "date", "date", "text", "date", "numeric", 
                                          "numeric", "date", "numeric", "date", 
                                          "numeric", "date", "numeric", "numeric", 
                                          "numeric", "numeric", "numeric", 
                                          "numeric", "text", "date", "text"))
Disc.frame$Disc.rsn <- ifelse(is.na(Disc.frame$Disc.rsn),"BLK",Disc.frame$Disc.rsn)
Disc.frame$Delq.status <- ifelse(is.na(Disc.frame$Delq.status),"BL",Disc.frame$Delq.status)
Disc.frame$Svc.chng <- ifelse(is.na(as.Date(Disc.frame$Svc.chng)),"0001-01-01",as.character(Disc.frame$Svc.chng))
Disc.frame$Svc.chng <- as.Date(Disc.frame$Svc.chng)
Disc.frame$Lst.PPV.date <- ifelse(is.na(as.Date(Disc.frame$Lst.PPV.date)),"0001-01-01",as.character(Disc.frame$Lst.PPV.date))
Disc.frame$Lst.PPV.date <- as.Date(Disc.frame$Lst.PPV.date)
Disc.frame$Disc.date <- ifelse(is.na(as.Date(Disc.frame$Disc.date)),"0001-01-01",as.character(Disc.frame$Disc.date))
Disc.frame$Disc.date <- as.Date(Disc.frame$Disc.date)
Disc.frame$Lst.Pay.date <- ifelse(is.na(as.Date(Disc.frame$Lst.Pay.date)),"0001-01-01",as.character(Disc.frame$Lst.Pay.date))
Disc.frame$Lst.Pay.date <- as.Date(Disc.frame$Lst.Pay.date)
Disc.frame$Type <- as.factor(Disc.frame$Type)
Disc.frame$Delq.status <- as.factor(Disc.frame$Delq.status)
Disc.frame$Disc.rsn <- as.factor(Disc.frame$Disc.rsn)
train.frame <- Disc.frame[1:750,]
test.frame <- Disc.frame[751:1000,]

#rf.model <- randomForest(Delq.status ~ Pre.stmt.bal + Lst.stmt.bal + Lst.Pay.date + PPY.Bal + Lst.pay.amt  + Delq.amt + Disc.rsn + Connect.date + Create.date + Delq.1 + Delq.2 + Delq.3 + Delq.4 + Delq.5 + Svc.chng ,data = train.frame[,3:22], ntree = 1000, importance = TRUE)
Dt.model <- rpart(Delq.status ~ ., data = train.frame[,-c(1,2,21)], method="anova",control = rpart.control(minsplit = 30))

rf.model <- randomForest(Delq.status ~ ., data = train.frame[,-c(1,2,21,23)], ntree = 500, importance = TRUE, mtry = 10)



train.frame$Pred.Acc <- rf.model$predicted
rf.rsn.model <- randomForest(Disc.rsn ~ ., data = train.frame[,-c(1,2,21)], ntree = 500, importance = TRUE, mtry = 10)
train.frame$Pred.rsn <- rf.rsn.model$predicted
varImpPlot(rf.model)
plot(rf.model)
table(train.frame$Delq.status,rf.model$predicted)
table(train.frame$Delq.status,Dt.model$predicted)
my.data <- predict(rf.model,test.frame)
DT.predit <- predict(Dt.model,test.frame)
summary(my.data)
summary(test.frame$Delq.status)
sucess.rate <- mean(test.frame$Delq.status == my.data)*100
sucess.rate.dt <- mean(test.frame$Delq.status == DT.predit)*100
varUsed(rf.model)
importance(rf.model)
fgl.res <- tuneRF(train.frame[,-c(1,2,20,21)], train.frame[,20], stepFactor=1.5)
table(rf.model$predicted,train.frame$Delq.status)
Pred.Acc <- ifelse(rf.model$predicted != train.frame$Delq.status & rf.model$predicted == 'C',train.frame$AcctNumber,NA)
a<-as.data.frame(na.omit(Pred.Acc))
Pred.test.Acc <- ifelse(my.data != test.frame$Delq.status & my.data == 'C',test.frame$AcctNumber,NA)
a.test<-as.data.frame(na.omit(Pred.test.Acc))
