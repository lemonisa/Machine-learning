#### Assignment 1. Spam classification with nearest neighbors

#### 1.1 Import data
data <- read.csv("C:/Users/Sam/Desktop/machine learning/lab1/spambase.csv")

# divide it into training and test sets (50%/50%)
n=dim(data)[1]
set.seed(12345)
id=sample(1:n, floor(n*0.5))
train=data[id,]
test=data[-id,]

traindata = as.matrix(train)
testdata = as.matrix(test)

#### 1.2 create K-nearest neighbors fun()
knearest <- function(data, k, newdata){
  X = data[,-ncol(data)]   #except the last col of spam
  Y = newdata[,-ncol(data)]
  
  # compute d(x,y) 
  Xhat = X/sqrt(rowSums(X^2))
  Yhat = Y/sqrt(rowSums(Y^2))
  
  # Compute distance 
  C = Xhat %*% t(Yhat)
  dist = 1-C 
  #View(dist) # 1370*1370
  
  # order index of the col of dist, 2 for col, increase, each col for each Y 
  order_dist = apply(dist,2,order) 
  #View(order_dist) # 1370*1370
  
  # Compute predicted class probabilities
  index_x = order_dist[1:k, ]
  #View(index_x) # 5*1370
  
  nearest = matrix(data[index_x,ncol(data)],nrow = k,ncol=ncol(order_dist)) 
  #View(nearest) # 5*1370
  ki = colSums(nearest)
  #print(ki)
  prob = ki/k
  return(prob)
}

knearest(traindata, k=5, testdata)

#### 1.3 Classify the training and test data by using K=5
# report the confusion matrix (use table())
# misclassification rate for training and test data

# test data
fit1 = knearest(traindata, k=5, testdata) # fit:probility %
predY1 = as.integer(fit1>0.5)   # pred: classify(0,1) as.integer:(TRUE FALSE)->(1,0) 
true1 = testdata[,ncol(data)]
conf1 = table(true1,predY1)
misc1 = (1-sum(diag(conf1))/length(true1))
conf1
misc1

# training data
fit2 = knearest(traindata, k=5, traindata)
predY2 = as.integer(fit2>0.5)
true2 = traindata[,ncol(data)]
conf2 = table(true2,predY2)
misc2 = (1-sum(diag(conf2))/length(true2))
conf2
misc2

#### 1.4 Repeat step 3 with K=1
# test data
fit3 = knearest(traindata, k=1, testdata)
predY3 = as.integer(fit3>0.5)
true3 = testdata[,ncol(data)]
conf3 = table(true3,predY3)
misc3 = (1-sum(diag(conf3))/length(true3))
conf3
misc3

# training data
fit4 = knearest(traindata, k=1, traindata)
predY4 = as.integer(fit4>0.5)
true4 = traindata[,ncol(data)]
conf4 = table(true4,predY4)
misc4 = (1-sum(diag(conf4))/length(true4))
conf4
misc4

#### 1.5 Use classifier kknn() fr package kknn  with K=5
# report the confusion matrix & misclassification rate for test data

library(kknn)

# test data
# Way 1: can compared with different value, not only 0.5
fit5 = kknn(as.factor(train$Spam)~., train, test, k = 5)
prob5 = fit5$prob # 2 cols: 1st col for 0,2nd col for 1
predY5 = as.integer(prob5[,2]>0.5)
true5 = testdata[, ncol(data)]
conf5 = table(true5,predY5)
misc5 = (1-sum(diag(conf5))/length(true5))
conf5
misc5

# way 2: if compared with 0.5
fit52 = kknn(as.factor(train$Spam)~., train, test, k = 5)
predY52 <- predict(fit52)
true52 = testdata[, ncol(data)]
conf52 = table(true52, predY52)
conf52

# package(class) knn() 
library(class)
fit6 = knn(train, test, as.factor(train$Spam), k=5,prob = TRUE)
prob6 = attributes(fit6)$prob
prob6 = ifelse(fit6=="1", prob6, 1-prob6)
predY6 = as.integer(prob6>0.5)
true6 = testdata[, ncol(data)]
conf6 = table(true6,predY6)
misc6 = (1-sum(diag(conf6))/length(true6))
conf6
misc6

# 1.6 Compute sensitivity & specificity values for the two methods
# Use knearest() and kknn() functions with K=5
# create pi = seq(from = 0.05, to = 0.95, by = 0.05)

# Use knearest()
fit7 = knearest(traindata, k=5, testdata)
true7 = testdata[,ncol(data)]

sens7 = c()
spc7 = c()
pi = seq(from = 0.05, to = 0.95, by = 0.05)
for (i in 1:length(pi)){
  prediction7 = as.integer(fit7>pi[i])
  confusion7 = table(true7, prediction7)
  sens7[i] = confusion7[2,2]/sum(confusion7[2,])
  spc7[i] = confusion7[1,1]/sum(confusion7[1,])
}
sens7
spc7

# use kknn()
library(kknn)
fit8 = kknn(as.factor(train$Spam)~.,train,test, k = 5)

prob8 = fit8$prob # probability %
true8 = testdata[, ncol(data)]

sens8 = c()
spc8 = c()
pi = seq(from = 0.05, to = 0.95, by = 0.05)
for (i in 1:length(pi)){
  prediction8 = as.integer(prob8[,2]>pi[i])  # classify(0,1)
  confusion8 = table(true8, prediction8)
  sens8[i] = confusion8[2,2]/sum(confusion8[2,])
  spc8[i] = confusion8[1,1]/sum(confusion8[1,])
}
sens8
spc8

# plot the corresponding ROC curves of 2 methods in 1 plot
x1  <- 1 - spc7 # FPR
y1 <- sens7     # TPR
x2 <- 1 - spc8
y2 <- sens8

# 1st plot
plot(x=x1, y=y1,col = "red", type = "l",  # type"l"for line
     xlab = "fpr", ylab = "tpr", 
     xlim = c(0, 1), ylim = c(0, 1))  #usually x,ylim=c(0,1) for probility
# 2nd plot
lines(x=x2, y=y2,col = "blue", type = "l")




#### Assignment 2. Inference about lifetime of machines
# 2.1 Import the data to R.
df <- read.csv("C:/Users/Sam/Desktop/machine learning/lab1/machines.csv")

#### 2.2
# computes the log-likelihood for a given theta & a given data vector x.
n = nrow(df)

# log-likelihood fun
log_likelihood <- function(n, theta, x){
  log_likelihood <- n*log(theta) - theta * sum(x)
  return(log_likelihood)
}

# loop in a given theta
theta <- seq(from = 0.1, to = 5, by = 0.1)
N = length(theta)

ll <- c()
for(i in 1:N){
  ll[i] = log_likelihood(n=n, theta = theta[i], x = df$Length)
}
ll

# Plot the curve showing the dependence of log-likelihood on theta
plot(x=theta,y=ll,col = "red", type = "l",
     xlab = "theta", ylab = "log-likelihood", 
     xlim = c(0.1,5), ylim = c(-150, -20))

# value of theta on maximum likelihood
index_max_y1 <- which.max(ll)
theta_optimal1 <- theta[index_max_y1]
theta_optimal1

#### 2.3 Repeat step 2, only 6 first observations from the data
ll2 <- c()
for(i in 1:N){
  ll2[i] = log_likelihood(n=6, theta = theta[i], x = df$Length[1:6])
}
ll2

# value of theta on maximum likelihood
index_max_y2 <- which.max(ll2)
theta_optimal2 <- theta[index_max_y2]
theta_optimal2

# Plot the curve showing the dependence of log-likelihood on theta
# put the two log-likelihood curves (from step 2 and 3) in the same plot
plot(x=theta,y=ll,col = "red", type = "l",
     xlab = "theta", ylab = "log-likelihood", 
     xlim = c(0.1,5), ylim = c(-150, 0))
lines(x=theta,y=ll2,col = "blue")

#### 2.4 log-likelihood fun for Bayesian model with a prior
theta <- seq(from = 0.1, to = 5, by=0.1)
n = nrow(df)
N = length(theta)
lamda = 10

l_theta <- function(n, theta, x, lamda){
  l_theta <- (n*log(theta) - theta*sum(x) + log(lamda*exp(-lamda*theta)))
  return(l_theta)
}

# loop in a given theta
ll3 <- c()
for(i in 1:N){
  ll3[i] = l_theta(n=n, theta = theta[i], x = df$Length, lamda)
}
ll3

# Plot the curve showing the dependence of l(theta)
plot(x= theta, y=ll3,col = "green", type = "l", xlab = "theta", 
     ylab = "l(theta)", xlim = c(0.1,5), ylim = c(-150, -20))

# value of theta on maximum likelihood
index_max_y3 <- which.max(ll3)
theta_optimal3 <- theta[index_max_y3]
theta_optimal3

#### 2.5 use theta value found in step 2 to generate 50 new observations
# use standard random number generators,random exponential distribution
random = rexp(50, rate = theta_optimal1)

# Create the histograms of the original and the new data
hist(random, xlab = "Lifetime", main = "Histogram of new data")
hist(df$Length, xlab = "Lifetime", main = "Histogram of original data")

