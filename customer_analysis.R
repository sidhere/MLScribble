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
loadData <- read_delim("C:/ML/data/Comcast.csv", 
                       "\t", escape_double = FALSE, col_types = cols(Account_num = col_character()), 
                       trim_ws = TRUE)
#View(loadData)

library(readr)
Comcast.amount <- read_delim("C:/ML/data/Comcast_amount.csv", 
                             "\t", escape_double = FALSE, col_types = cols(account_num = col_character()), 
                             trim_ws = TRUE)
#View(Comcast_amount)
#table(loadData$Account_num,loadData$Category)
# Itun[ , colSums(is.na(Itun)) == 0]
# loadData$Category[loadData$Category == 'BUNDLE_NAME'] <- loadData$Item_desc[loadData$Category == 'BUNDLE_NAME']
loadData$NormCategory <- ifelse( loadData$Category == 'BUNDLE_NAME',loadData$Item_desc,loadData$Category )
#table(loadData$Account_num,loadData$NormCategory)

#NormData <- data.frame("acc_num" = unique(loadData$Account_num),
#                       "type" = loadData$NormCategory[loadData$Category == 'BUNDLE_NAME']
 #                      )

# Identifying the LoBs of each customer
NormData <- loadData %>%
  group_by(Account_num) %>%
  summarise(LoBs = paste0(unique(NormCategory),collapse=","),Subscription = paste0(unique(Item_desc),collapse=","))
NormData$LoBs <- gsub('.*(Triple Play).*','\\1',NormData$LoBs)
#NormData$LoBs <- gsub('.*(Double Play).*','\\1',NormData$LoBs)

# removal of fee items
stopwords <- c("Bundle Discount","Nonpublished Number","Deposit","Shipping and Handling","Service Protection Plan","HD Technology Fee","Broadcast TV Fee","Franchise Fees","Federal Regulatory Fee","Network Support Fee","PEG Support Fee","State and Local Taxes","911 Fee(s)","Additional Outlet","Regional Sports Fee","Late Fee","Returned Payment Fee","Reactivate Fee- Internet","Reactivate Fee- Video","Unreturned Digital Equipment","Unreturned Cablecard","Unreturned Voice Equipment","Returned Payment Fee","Regulatory Recovery Fees")
#stopwords <- c(".*Partial Month(s)",".*Fee",".*Taxes","HD Technology Fee","Broadcast TV Fee","Franchise Fees","Federal Regulatory Fee","Network Support Fee","PEG Support Fee","State and Local Taxes","911 Fee(s)","Additional Outlet","Regional Sports Fee","Late Fee","Returned Payment Fee","Reactivate Fee- Internet","Reactivate Fee- Video","Unreturned Digital Equipment","Unreturned Cablecard","Unreturned Voice Equipment","Returned Payment Fee","Regulatory Recovery Fees","")

NormData$Subscription <- removeWords(NormData$Subscription,stopwords)

# Netflix accounts Extraction
NormData$Netflix <- sapply('+Netflix+',grepl,NormData$Subscription)

#on demands purchased customers
#NormData$VoDs <- sapply('+HD,',grepl,NormData$Subscription)

# temp$AccType <- gsub('.*(Triple)','\\1',temp$AccType)
# temp$AccType <- gsub('.*(Double)','\\1',temp$AccType)
# temp$AccType <- gsub('(triple).*','\\1',temp$AccType)
# temp <- loadData %>%
#   group_by(Account_num) %>%
#   summarise(result = paste0(unique(NormCategory),collapse=" "),res2 = paste0(unique(Item_desc),collapse=" ") )

# stopwords <- c("Franchise Fees","Federal Regulatory Fee","Network Support Fee","PEG Support Fee","State and Local Taxes","911 Fee(s)"," Additional Outlet")
# temp$res2 <- removeWords(temp$res2,stopwords)
# 
# temp$net <- ifelse(match('netflix',temp$res2)

#library(xlsx)
#write.xlsx(NormData, "C:/ML/data/output.xlsx")



# dealing with amount

NormAmt <- Comcast.amount %>%
  group_by(account_num) %>%
  summarise(item = paste0(item_desc,collapse=","),Amt = paste0(totalAmt,collapse=","))

# converting comma seperated value in to a seperate column

NormAmt$previousBalance <- str_split_fixed(NormAmt$Amt , ",", 5)[,1]
NormAmt$payment <- str_split_fixed(NormAmt$Amt , ",", 5)[,2]
NormAmt$unpaidAmount <- ifelse((str_split_fixed(NormAmt$item, ",", 5)[,3]) == 'UnpaidAmount',(str_split_fixed(NormAmt$Amt, ",", 5)[,3]),'0')
NormAmt$NewChargesAmount <- ifelse((str_split_fixed(NormAmt$item, ",", 5)[,3]) == 'NewChargesAmount',(str_split_fixed(NormAmt$Amt, ",", 5)[,3]),(str_split_fixed(NormAmt$Amt, ",", 5)[,4]))
NormAmt$totalAmountDue <- ifelse((str_split_fixed(NormAmt$item, ",", 5)[,4]) == 'TotalAmountDue',(str_split_fixed(NormAmt$Amt, ",", 5)[,4]),(str_split_fixed(NormAmt$Amt, ",", 5)[,5]))
# merging NormData and NormAmt
NormData <- NormData[-c(91,101),]
NormData <- cbind(NormData, NormAmt[!names(NormAmt) %in% c("item","Amt")])

FeaturedData <- NormData[,-c(5)]
# converting the mode of the data frame

FeaturedData$payment = as.numeric(FeaturedData$payment)
FeaturedData$NewChargesAmount = as.numeric(FeaturedData$NewChargesAmount)
FeaturedData$previousBalance = as.numeric(FeaturedData$previousBalance)
FeaturedData$unpaidAmount = as.numeric(FeaturedData$unpaidAmount)
FeaturedData$totalAmountDue = as.numeric(FeaturedData$totalAmountDue)
#FeaturedData$Lobcount = as.numeric(FeaturedData$Lobcount)
FeaturedData$Subscriptionrefine <- gsub("(?!,)[[:punct:]]", "", FeaturedData$Subscription,perl=TRUE) # remove all punc except comma
FeaturedData$Subscriptionrefine <- gsub(",+", ",", FeaturedData$Subscriptionrefine) # replace multiple commas with single comma
FeaturedData$Subscriptionrefine <- gsub("(Netflix.*?),","Netflix,",FeaturedData$Subscriptionrefine)
FeaturedData$Subscriptionrefine <- gsub("Xfinity.*Latino","Xfinity Latino",FeaturedData$Subscriptionrefine)
FeaturedData$Subscriptionrefine <- removeWords(FeaturedData$Subscriptionrefine,".*Partial Months")
FeaturedData$Subscriptionrefine <- gsub(",$", "", FeaturedData$Subscriptionrefine) # remove the last comma
FeaturedData$Subscriptionrefine <- gsub("^,", "", trimws(FeaturedData$Subscriptionrefine)) # remove first comma


# extracting unique subs and featuring it
#recomd.subs<-matrix(unique(unlist(str_split(FeaturedData$Subscriptionrefine,","))),,byrow=TRUE)
ls.sub<-str_split(FeaturedData$Subscriptionrefine,",")
recomd.subs<-matrix(unique(unlist(ls.sub)),ncol=1,byrow=TRUE)
recomd.subs<-removeWords(recomd.subs,".*Partial Months")
recomd.subs[trimws(recomd.subs)==""] <- NA
recomd.subs <- gsub("Netflix.*","Netflix",recomd.subs)
recomd.subs <- gsub("Xfinity.*Latino","Xfinity Latino",recomd.subs)
recomd.subs <- gsub(".*Fee.*",NA,recomd.subs)
recomd.subs <- gsub(".*Install.*",NA,recomd.subs)
recomd.subs<-na.omit(recomd.subs)
recomd.subs <- unique(recomd.subs)


# preparing data for recommedor
set.seed(200)
recomd.subs[,] <-recomd.subs[sample(nrow(recomd.subs)),]
recomd.data <- data.frame(FeaturedData$Account_num)
recomd.data[,as.character(unlist(recomd.subs[,]))]<-NA
#colnames(recomd.data) <- as.character(unlist(recomd.subs[,]))
#recomd.data[recomd.subs[,]]<- NA 

i<-1
for(item in FeaturedData$Subscriptionrefine)
{
  eachAcc <- unlist(str_split(item,","))
  res <- numeric(92)
  for (value in eachAcc)
  {
    temp <- ifelse(grepl(value,colnames(recomd.data[-1])) & value != "",1,0)
    res<-res+temp
  }
  recomd.data[i,-1] = t(res)
  i<-i+1
}

# Featuring  Lobs



FeaturedData$LoBRefine <- FeaturedData$LoBs
FeaturedData$LoBRefine <- removeWords(gsub("[()]", "", FeaturedData$LoBRefine),".*Partial Months,")
FeaturedData$LoBRefine <- removeWords(FeaturedData$LoBRefine,"Bundle Discount,")
FeaturedData$LoBRefine <- removeWords(FeaturedData$LoBRefine,",NA")
FeaturedData$LoBRefine[which(FeaturedData$LoBRefine == "Triple Play")] <- "TV,Internet,Voice"
FeaturedData$LoBRefine <- gsub(",.+Double Play",",D", FeaturedData$LoBRefine)
FeaturedData$LoBRefine <- gsub(".+Double Play","D", FeaturedData$LoBRefine)
FeaturedData$LoBRefine <- gsub("Double Play.+,","D,", FeaturedData$LoBRefine)
DouplePlay.location <-  grep("D,|,D",FeaturedData$LoBRefine)
FeaturedData$LoBRefine[DouplePlay.location] <- gsub("D,","",FeaturedData$LoBRefine[DouplePlay.location])
FeaturedData$LoBRefine[DouplePlay.location] <- ifelse(str_count(FeaturedData$LoBRefine[DouplePlay.location],",") == 0,paste(FeaturedData$LoBRefine[DouplePlay.location],"Internet",sep = ","),FeaturedData$LoBRefine[DouplePlay.location])


#removing all except Tv Internet Home and video
coll.words <- c("TV","Internet","Voice","Home")
for (item in 1:length(FeaturedData$LoBRefine))
{
  if(str_count(FeaturedData$LoBRefine[item],",") != 0)
  {
    iter <- str_count(FeaturedData$LoBRefine[item],",")+1
    for(i in 1:iter)
    {
      if(all(str_split(FeaturedData$LoBRefine[item],",")[[1]][[i]] != coll.words))
      {
        FeaturedData$LoBRefine[item] <- gsub(str_split(FeaturedData$LoBRefine[item],",")[[1]][[i]],"",FeaturedData$LoBRefine[item])
      }
    }
  }
}
FeaturedData$LoBRefine <- gsub(",+", ",", FeaturedData$LoBRefine) # replace multiple commas with single comma
FeaturedData$LoBRefine <- gsub(",$", "", FeaturedData$LoBRefine) # remove the last comma
FeaturedData$LoBRefine <- gsub("^,", "", trimws(FeaturedData$LoBRefine)) # remove first comma
FeaturedData$LoBRefine <- gsub("Internet,TV","TV,Internet", FeaturedData$LoBRefine)
FeaturedData$LoBRefine <- gsub("Voice,TV","TV,Voice", FeaturedData$LoBRefine)
FeaturedData$LoBRefine <- gsub("Voice,TV,Internet","TV,Internet,Voice", FeaturedData$LoBRefine)
FeaturedData$LoBRefine <- gsub("TV,Voice,Internet","TV,Internet,Voice", FeaturedData$LoBRefine)

FeaturedData$Lobcount <- str_count(FeaturedData$LoBRefine ,",")+1


FeaturedData$Lobcount <- as.numeric(FeaturedData$Lobcount)
FeaturedData$LoBRefine <- as.factor(FeaturedData$LoBRefine)
FeaturedData <- FeaturedData[2:99,]

# Finding out a good credit customer

FeaturedData$goodcust <- ifelse(FeaturedData$previousBalance + FeaturedData$payment == 0 , TRUE,FALSE)


# extraction of Bundle details
Bundle.frame <- subset(loadData,loadData$Category == "BUNDLE_NAME")
FeaturedData$Bundle <- Bundle.frame$Item_desc[match(FeaturedData$Account_num, Bundle.frame$Account_num)]
FeaturedData$Bundle.Amt <- Bundle.frame$TotalAmt[match(FeaturedData$Account_num, Bundle.frame$Account_num)]
FeaturedData$Bundle.Amt <- ifelse(is.na(FeaturedData$Bundle.Amt),0,FeaturedData$Bundle.Amt)
HD.frame <- subset(loadData,loadData$Item_desc == "HD Technology Fee")
FeaturedData$HD <-  HD.frame$Item_desc[match(FeaturedData$Account_num, HD.frame$Account_num)]
FeaturedData$HD <- ifelse(grepl("HD",FeaturedData$Bundle) | grepl("HD",FeaturedData$HD),TRUE,FALSE)

#VoDs extraction

movies.frame <- subset(loadData,loadData$GroupName == "Onetime_Extras")
Normovies.frame <- movies.frame %>%
  group_by(Account_num) %>%
  summarise(item = paste0(Item_desc,collapse=","), Amt = sum(TotalAmt))

FeaturedData$VoDs <- FALSE
FeaturedData$VoDs <- as.logical(match(FeaturedData$Account_num,Normovies.frame$Account_num)) # True for the one who rented a movie
for(iter in 1:length(Normovies.frame$Account_num))
{
  print(iter)
  FeaturedData$VoDs[FeaturedData$Account_num == na.omit(FeaturedData$Account_num[FeaturedData$VoDs == TRUE])[1]] <- Normovies.frame$Amt[iter]
}
FeaturedData$VoDs <- ifelse(is.na(FeaturedData$VoDs),0,FeaturedData$VoDs) # converting NAs to false


# Tv Subscription


Channel.frame <- subset(loadData,loadData$GroupName == "Regular_Bundle" & loadData$Category == "TV")
Channel.frame <- Channel.frame[!grepl("Fee",Channel.frame$Item_desc),]
Channel.frame <- subset(Channel.frame,as.numeric(Channel.frame$TotalAmt) > 0)
NormTV.frame <- Channel.frame %>%
  group_by(Account_num) %>%
  summarise(item = paste0(Item_desc[as.numeric(Channel.frame$TotalAmt) > 0],collapse=","))

NormTV.frame$item <- gsub("NA,","",NormTV.frame$item)

NormTV.frame$item <- gsub(",NA","",NormTV.frame$item)
FeaturedData$addlsubs <- as.logical(match(FeaturedData$Account_num,NormTV.frame$Account_num)) # finding out additional subscribed users

FeaturedData$addlsubs <- ifelse(is.na(FeaturedData$addlsubs),FALSE,FeaturedData$addlsubs) # converting NAs to false




# Additional Outlet

Addotl.frame <- loadData %>%
  filter(Item_desc == "Additional Outlet")  %>%
  group_by(Account_num) %>%
  summarise(item = paste0(Item_desc,collapse=","),Amt = sum(TotalAmt))
Addotl.frame$ao.count <- str_count(Addotl.frame$item,",")+1
Ao.location <- match(Addotl.frame$Account_num,FeaturedData$Account_num)
FeaturedData$AO.Amt<-0
FeaturedData$AO.count <- 0
FeaturedData$AO.Amt[Ao.location] <- Addotl.frame$Amt 
FeaturedData$AO.count[Ao.location] <- Addotl.frame$ao.count
# Anyroom DVR

Anydvr.frame <- loadData %>%
  filter(Item_desc == "Anyroom DVR")  %>%
  group_by(Account_num) %>%
  summarise(item = paste0(Item_desc,collapse=","),Amt = sum(TotalAmt))
Anydvr.frame$ad.count <- str_count(Anydvr.frame$item,",")+1
Ad.location <- match(Anydvr.frame$Account_num,FeaturedData$Account_num)
FeaturedData$Ad.Amt<-0
FeaturedData$Ad.count <- 0
FeaturedData$Ad.Amt[Ad.location] <- Anydvr.frame$Amt 
FeaturedData$Ad.count[Ad.location] <- Anydvr.frame$ad.count



# Equipment
Equip.frame <- loadData %>%
  filter(GroupName == "Regular_Equipment"  & Category != "TV") %>%
  group_by(Account_num) %>%
  summarise(item = paste0(Item_desc,collapse=","),Amt = sum(TotalAmt),Category = paste0(Category,collapse=","))
Equip.frame$eq.count <- str_count(Equip.frame$item,",")+1
eq.location <- match(Equip.frame$Account_num,FeaturedData$Account_num)
FeaturedData$eq.Amt<-0
FeaturedData$eq.count <- 0
FeaturedData$eq.category <- "NIL"
FeaturedData$eq.Amt[eq.location] <- Equip.frame$Amt 
FeaturedData$eq.count[eq.location] <- Equip.frame$eq.count
FeaturedData$eq.category[eq.location] <- Equip.frame$Category
FeaturedData$eq.category <- as.factor(FeaturedData$eq.category)
#unique.Subscription <- str_split_fixed((unique(FeaturedData$Subscription)),",",5
#split.unique <- strsplit(unique.Subscription,",")



#ggplot(FeaturedData, aes(LoBs, Subscription, color = VoDs)) + geom_point()#
# ComcastCluster <- kmeans(FeaturedData[, c(2,6)], 5)
# 
# ComcastCluster$cluster <- as.factor(ComcastCluster$cluster)
# ggplot(FeaturedData, aes(previousBalance, LoBs, color = ComcastCluster$cluster)) + geom_point()


# desicion tree
# 
# Pred.output <- rpart(unpaidAmount ~ LoBs + previousBalance + payment, data=FeaturedData, method="anova",control = rpart.control(minsplit = 30))
# printcp(Pred.output)
# plotcp(Pred.output)
# summary((Pred.output))
# plot(Pred.output, uniform=TRUE, 
# main="Classification Tree for subcription")
# text(Pred.output, use.n=TRUE, all=TRUE, cex=.8)
# par(mfrow=c(1,2)) # two plots on one page 
# rsq.rpart(Pred.output) # visualize cross-validation results 


# recommendar Algorithm
# affinity.data<-data.matrix(recomd.data[-1])
# rownames(affinity.data) <- paste(recomd.data$FeaturedData.Account_num)
# affinity.matrix<- as(affinity.data,"realRatingMatrix")
# 
# Rec.model<-Recommender(affinity.matrix, method = "UBCF")
# recommended.items.newcust <- predict(Rec.model, affinity.matrix[1:5], n=5)
# #b<- as(affinity.matrix, "data.frame")
# as(recommended.items.newcust, "list")
# recommended.items.newcust.top3 <- bestN(recommended.items.newcust, n = 3)
# as(recommended.items.newcust.top3, "list")
# 
# # Predict list of product which can be recommended to given users	 	
# #to predict affinity to all non-rated items 
# predicted.affinity.newcust <- predict(Rec.model, affinity.matrix[1], type="ratings")
# # to see the user "u15348"'s predicted affinity for items we didn't have any value for
# as(predicted.affinity.newcust, "list")
# # .. and the real affinity for the items obtained from the affinity.matrix
# as(affinity.matrix["8220202108604488",], "list")


# linear Regression
linear.model <- lm(NewChargesAmount ~ previousBalance + unpaidAmount + payment + totalAmountDue , data = FeaturedData )
linear.model1 <- lm(NewChargesAmount ~ LoBRefine + HD + addlsubs , data = FeaturedData )
x <- as.numeric(FeaturedData$totalAmountDue)
y <- as.numeric(FeaturedData$NewChargesAmount)
m<-nls(y ~ a*x/(b+x)) 
a_start<- -180 #param a is the y value when x=0
b_start<-2/a_start #b is the decay rate
m1<-nls(y ~ a*x/(b+x),start=list(a=a_start,b=b_start))
summary(linear.model)
xyplot(resid(linear.model1) ~ fitted(linear.model1),
       xlab = "Fitted Values",
       ylab = "Residuals",
       main = "Residual Diagnostic Plot",
       panel = function(x, y, ...)
       {
         panel.grid(h = -1, v = -1)
         panel.abline(h = 0)
         panel.xyplot(x, y, ...)
       }
)

cor(y,predict(m))
cor(y,predict(m1))
plot(x,y)
lines(x,predict(m),lty=2,col="red",lwd=3)
y.predict <- predict(linear.model1,FeaturedData[3,6:8],interval = "confidence")
cv.lm(FeaturedData , linear.model, m=10)


# GLM logit regression
model <- glm(VoDs ~ Lobcount + NewChargesAmount,family=binomial(link='logit'),data=FeaturedData)
summary(model)
print(paste('Accuracy', 1-mean(model$fitted.values != FeaturedData$Netflix)))



# Random forest
train.data <- FeaturedData[1:70,]
test.data <- FeaturedData[71:98,]
rf.model <- randomForest(factor(Lobcount) ~ NewChargesAmount + LoBRefine+ previousBalance + HD + goodcust , data = train.data,  ntree = 30, importance = TRUE)
rf.model.linear <- randomForest(NewChargesAmount ~ LoBRefine + HD + addlsubs, data = train.data,  ntree = 1000, importance = TRUE)
plot(rf.model.linear)
varImpPlot(rf.model.linear)
MDSplot(rf.model, FeaturedData$LoBRefine)
predict(rf.model.linear,test.data)




# K-means Clustering

cluster.frame <- FeaturedData[,c(4,8,15:24)]
cluster.frame$Netflix <- as.numeric(cluster.frame$Netflix)
cluster.frame$HD <- as.numeric(cluster.frame$HD)
cluster.frame$VoDs <- as.numeric(cluster.frame$VoDs)
cluster.frame$addlsubs <- as.numeric(cluster.frame$addlsubs)


#ggplot(FeaturedData, aes(NewChargesAmount, previousBalance, color = Lobcount)) + geom_point()
set.seed(420)
unsup.model <-kmeans(cluster.frame,3,iter.max=500)
table(unsup.model$cluster,FeaturedData$Lobcount)
plot(FeaturedData$Bundle.Amt,FeaturedData$NewChargesAmount)