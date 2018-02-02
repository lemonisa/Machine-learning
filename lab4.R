####### Assignment 1. Uncertainty estimation
# sep = ";", read.csv2
mydata <- read.csv2("C:/Users/Sam/Desktop/machine learning/lab4/State.csv")

### 1.1 Reorder data with respect to the increase of MET
ind <- order(mydata$MET,decreasing = FALSE)
for(i in 1:ncol(mydata)){
  mydata[, i] = mydata[, i][ind]
}
mydata

# Plot MET versus EX
plot(mydata$MET, mydata$EX, col = "blue", xlab = "MET", ylab = "EX")

### 1.2 Fit a regression tree model with target EX~feature MET 
# the number of the leaves is selected by cross-validation #######

# Selecting optimal tree by penalizing
# use the entire data set, nrow(mydata)=48
# set minimum number of observations in a leaf = 8(setting minsize in tree.control)
library(tree)
fit = tree(EX~MET, data = mydata, control = tree.control(nobs = 48, minsize = 8))
predict(fit)
set.seed(12345)
cv.res = cv.tree(fit) #cross-validation
plot(cv.res$size, cv.res$dev, type = "b", col = "red")
plot((cv.res$k), cv.res$dev, type = "b", col = "red")
plot(cv.res)

# Report optimal tree
# gives the deviance for each K (small is better)
cv.res$dev
# [1] 192109.0 186515.5 188981.5
index <- which.min(cv.res$dev) #5
bestsize <- cv.res$size[index]
bestsize  # 3

# refit the model with the number of leafs of best size
finalTree = prune.tree(fit, best = bestsize) ##########choose from above plot, min deviance
Ffit = predict(finalTree)
plot(finalTree)
text (finalTree , pretty =0)

# Plot the original and the fitted data
plot(mydata$MET, mydata$EX, col = "blue", xlab = "MET", ylab = "EX")
points(mydata$MET, Ffit, col = "red")

# Plot histogram of residuals.  
residuals = residuals(finalTree)
hist(residuals)

### 1.3 Compute and plot the 95%(default) confidence bands for the regression tree model from step 2
# non-parametric bootstrap.
library(boot)
#reordering data according to feature:MET
data2=mydata[order(mydata$MET),]

# computing bootstrap samples
f=function(data, ind){
  # extract bootstrap sample
  data1=data[ind,]
  # fit regression tree model
  fit = tree(EX~MET, data = data1, control = tree.control(nobs = 48, minsize = 8))
  res = prune.tree(fit, best = bestsize)  
  #predict values for all Area values from the original data
  EXP=predict(res,newdata=data2) 
  return(EXP)
}

# make bootstrap
res=boot(data2, f, R=2000) 

# Bootstrap cofidence bands for linear model
# compute confidence bands
e=envelope(res)

# compute fitted line
fit2=tree(EX~MET, data=data2, control = tree.control(nobs = 48, minsize = 8)) 
fianlTree = prune.tree(fit2, best = bestsize) #######
EXP=predict(fianlTree)

#plot fitted line
plot(mydata$MET, mydata$EX, pch=21, bg="orange", xlab = "MET", ylab = "EX")
points(data2$MET,EXP,type="l") 

# plot cofidence bands
points(data2$MET,e$point[2,], type="l", col="blue")
points(data2$MET,e$point[1,], type="l", col="blue")


### 1.4 Compute and plot the 95% confidence bands && prediction bands
# by using parametric bootstrap.
# assume Y~N(mu, s^2)
library(boot)
#reordering data according to feature:MET
data2=mydata[order(mydata$MET),]
fit2 = tree(EX~MET, data = data2, control = tree.control(nobs = 48, minsize = 8))
mle = prune.tree(fit2, best = bestsize) #####

rng=function(data, mle) {
  data1=data.frame(EX=mydata$EX, MET=mydata$MET)
  n=length(mydata$EX)
  #generate new target:EX
  data1$EX=rnorm(n,predict(mle, newdata=data1),sd(residuals(mle)))
  return(data1)
}

f1=function(data1){
  #fit regression tree model
  res=tree(EX~MET, data= data1, control = tree.control(nobs = 48, minsize = 8))
  res = prune.tree(res, best = bestsize) ######
  #predict values for all Area values from the original data
  EXP=predict(res,newdata=data2) 
  return(EXP)
}
# make bootstrap
res2=boot(data2, statistic=f1, R=1000, mle=mle, ran.gen=rng, sim="parametric")

# Bootstrap cofidence bands for tree model
# compute confidence bands
e2=envelope(res2)

# compute fitted line
fit2=tree(EX~MET, data=data2, control = tree.control(nobs = 48, minsize = 8)) 
fianlTree = prune.tree(fit2, best = bestsize) #######
EXP=predict(fianlTree)

# plot fitted line
plot(mydata$MET, mydata$EX, pch=21, bg="orange", xlab = "MET", ylab = "EX",
     ylim = c(100,450), xlim = c(0,90))
points(data2$MET,EXP,type="l") 

# plot cofidence bands
points(data2$MET,e2$point[2,], type="l", col="blue")
points(data2$MET,e2$point[1,], type="l", col="blue")

# compute Prediction bands
data2=mydata[order(mydata$MET),]
fit2 = tree(EX~MET, data = data2, control = tree.control(nobs = 48, minsize = 8))
mle = prune.tree(fit2, best = bestsize)

f2 = function(data1){
  #fit model
  temp = tree(EX~MET, data = data1, control = tree.control(nobs = 48, minsize = 8))
  res = prune.tree(temp, best = bestsize) #####
  #predict values for all Area values from the original data
  EXP = predict(res, newdata = data2)
  n = length(data2$EX)
  predictedP = rnorm(n, EXP, sd(residuals(mle)))
  return(predictedP)
  
}
# R=2000, more number
res3 = boot(data2, statistic = f2, R = 2000, mle = mle, ran.gen = rng, sim = "parametric") 

e3 = envelope(res3)

# Plot prediction bands
points(data2$MET, e3$point[2,], type = "l", col = "green")
points(data2$MET, e3$point[1,], type = "l", col = "green")





####### Assignment 2. Principal components
data <- read.csv2("C:/Users/Sam/Desktop/machine learning/lab4/NIRSpectra.csv")

### 2.1 Conduct a standard PCA
# by using the feature space
# plot to explain how much variation is explained by each feature.

data1=data
data1$Viscosity=c()
res=prcomp(data1)
lambda=res$sdev^2
#eigenvalues
lambda
#proportion of variation
# Select the minimal number of components 
# explaining at least 99% of the total variance.
sprintf("%2.3f",lambda/sum(lambda)*100)
screeplot(res)

# Select the minimal number of components 
# explaining at least 99% of the total variance.
# scores plot in the coordinates (PC1, PC2)
plot(res$x[,1], res$x[,2])


### 2.2 Trace plots for the loadings of PC1 & PC2. 

#U = loadings(res)
U=res$rotation
head(U)

plot(U[,1], main="Traceplot, PC1")
plot(U[,2],main="Traceplot, PC2")

### 2.3 ICA
# Perform Independent Component Analysis of PC1 & PC2, n.comp=2
set.seed(12345) 
library(fastICA)
a = fastICA(data1, n.comp=2,alg.typ = "parallel", fun = "logcosh", alpha = 1, 
            method = "R", row.norm = FALSE, maxit = 200, tol = 0.0001, verbose = TRUE)
# Report matrix W
a$W

# Explain the roll of the matrix w'= k.w
Wp = a$K%*%a$W
Wp

# present its columns in form of the trace plots.
plot(Wp[,1], main = "Traceplot, PC1")
plot(Wp[,2], main = "Traceplot, PC2")

# Make the score plot of the first two latent features
plot(a$S[,1], a$S[,2])


### 2.4 Fit a PCR model to the training data 
# where number of components is selected by cross validation.
library(pls)
set.seed(12345)

# where number of components is selected by cross validation.
pcr.fit = pcr(Viscosity~., data = data, scale = FALSE, validation = "CV")
summary(pcr.fit)

# Plot MSPE, the dependence of the mean-square predicted error
validationplot(pcr.fit, val.type = "MSEP")




#####################################
#####################################
# Estimate MSE of the optimal model by using the test set. 
pcr.fit1 = pcr(Viscosity~., ncomp=40,  data = train, scale = FALSE, validation = "none")  # 40######## 30
summary(pcr.fit1)

y1 = predict(pcr.fit1, newdata = test)
mse = mean((y1-test$Viscosity)^2,na.rm= TRUE)
mse


### 2.5 Divide the data into training(50%) and test(50%) sets.
n=dim(data)[1]
set.seed(12345) 
id=sample(1:n, floor(n*0.5)) 
train=data[id,] 
test=data[-id,]

#### 2.6 use PLS model
library(pls)
set.seed(12345)

# Estimate MSE of the optimal model by using the test set.
pls.fit = plsr(train$Viscosity~., data = train, scale = FALSE, validation = "CV")
summary(pls.fit)

# Plot MSPE
validationplot(pls.fit, val.type = "MSEP")

# Estimate MSE of the optimal model by using the test set. 
pls.fit2 = plsr(Viscosity~., ncomp=9,  data = train, scale = FALSE, validation = "none")
summary(pls.fit2)
y2 = predict(pls.fit2, newdata = test)
mse2 = mean((y2-test$Viscosity)^2,na.rm= TRUE)
mse2



