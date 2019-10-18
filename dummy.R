dummy = recomd.data
i<-1
for(item in FeaturedData$Subscriptionrefine)
{
  a <- unlist(str_split(item,","))
  d <- numeric(92)
  for ( value in a)
  {
    c <- ifelse(grepl(value,colnames(dummy[-1])) & value != "",1,0)
    d<-d+c
  }
  dummy[i,-1] = t(d)
  i<-i+1
}


m <- matrix(sample(c(as.numeric(0:5), NA), 50,
                   replace=TRUE, prob=c(rep(.4/6,6),.6)), ncol=10,
            dimnames=list(user=paste("u", 1:5, sep=''),
                            item=paste("i", 1:10, sep='')))
#s<-data.matrix(recomd.data[-1])


a.m<- as(m,"realRatingMatrix")

R.model<-Recommender(a.m, method = "UBCF")
r.i.n<- predict(R.model, a.m[1:2], n=5)
b2<- as(a.m, "data.frame")
as(r.i.n, "list")
r.i.n.t <- bestN(r.i.n, n = 3)
as(r.i.n.t, "list")

# Predict list of product which can be recommended to given users	 	
#to predict affinity to all non-rated items 
predicted.affinity.newcust <- predict(R.model, a.m[3], type="ratings")
# to see the user "u15348"'s predicted affinity for items we didn't have any value for
as(predicted.affinity.newcust, "list")
# .. and the real affinity for the items obtained from the affinity.matrix
as(a.m[3], "list")
