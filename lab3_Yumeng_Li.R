##### Assignment 1. LDA and logistic regression

data <- read.csv("C:/Users/Sam/Desktop/machine learning/lab3/australian-crabs.csv")

### 1.1 scatterplot of carapace length (CL) versus rear width (RW)
# observations are colored by Sex.
index = (data$sex=="Male")
plot(data$RW[index], data$CL[index], col = "red", 
     xlab = "Rear width", ylab = "Carapace length")
points(data$RW[!index], data$CL[!index], col = "blue")

### 1.2 implement LDA with proportional priors, (use only basic R functions)
# inputs RW and CL and output Sex

# class set for input male & female for RW & CL
index = (data$sex=="Male")

RW1 = data$RW[index]   #male
RW2 = data$RW[!index]  #female

CL1 = data$CL[index]   #male
CL2 = data$CL[!index]  #female

nc1 = length(CL1)  #length(CL1)=length(RW1)
nc2 = length(CL2)

k = 2

mu1 = c(mean(CL1),mean(RW1))
mu2 = c(mean(CL2),mean(RW2))

# compute sigma1 of male, size: 2*2 for cl & rw
sigma1 = matrix(0, nrow = 2, ncol = 2)   
for (i in 1:length(CL1)){
  xi = c(CL1[i], RW1[i])
  temp = (xi-mu1)%*%(t(xi-mu1))
  sigma1 = sigma1+temp
}
sigma1 = sigma1/nc1

# compute sigma2 of female, size: 2*2 for cl & rw
sigma2 = matrix(0, nrow = 2, ncol = 2)
for (i in 1:length(CL2)){
  xi = c(CL2[i], RW2[i])
  temp = (xi-mu2)%*%(t(xi-mu2))
  sigma2 = sigma2+temp
}
sigma2 = sigma2/nc2

sigma = (nc1*sigma1 + nc2*sigma2)/(nrow(data))

# compute coefficients for two classes
pihat1 = nc1/nrow(data) 

w0i1 = -0.5*t(mu1)%*%solve(sigma)%*%mu1+log(pihat1)
wi1 = solve(sigma)%*%mu1
w0i1
wi1

pihat2 = nc2/nrow(data)  

w0i2 = -0.5*t(mu2)%*%solve(sigma)%*%mu2+log(1-pihat2)
wi2 = solve(sigma)%*%mu2
w0i2
wi2

# Label data based on LDA 
label = c()
for (i in 1:nrow(data)){
  x = c(data$CL[i], data$RW[i])
  p1 = (w0i1 + t(wi1)%*%x)
  p2 = (w0i2 + t(wi2)%*%x)
  pmale = exp(p1)/(exp(p1)+exp(p2))
  label[i] = (pmale>=0.5)
}
label

### 1.3 plot classsification with a decision boundary
# Plot classifed data
plot(data$RW[label], data$CL[label], col="blue", 
     xlab = "Rear width", ylab = "Carapace length", 
     main = "Classified data by LDA")
points(data$RW[!label], data$CL[!label], col="red")

# plot decision boundary
range(data$CL)
# [1] 14.7 47.6
y = c(14:48)
# wi[1] for CL,y, wi[2] for RW,x
x = (w0i1-w0i2-(wi2[1]-wi1[1])*y)/(wi2[2]-wi1[2])

lines(x, y, lwd = 2)

### 1.4 plot classsification by logistic regression
# use function glm(), ##family=binomial##

sex = c()
class = glm(sex~CL+RW, family = "binomial", data = data)
p = predict(class, type = "response")
indexl = (p>=0.5)
# p[indexm] = 1, p[!indexm] = 0
# p = as.numeric(indexm)


# Plot classified data using logistic regression
# Plot classifed data
plot(data$RW[indexl], data$CL[indexl], col="blue", 
     xlab = "Rear width", ylab = "Carapace length", 
     main = "Classified by logistic regression")
points(data$RW[!indexl], data$CL[!indexl], col="red")

# plot decision boundary
range(data$RW)
#[1]  6.5 20.2
x1 = c(6:21) #xlab: RW

w = class$coefficients
#(Intercept)          CL          RW 
#13.616628      4.630747  -12.563893 
y1 = (-w[1]-w[3]*x1)/w[2]

lines(x1, y1, lwd = 2) #ylab: CL

##### Assignment 2. Analysis of credit scoring
df <- read.csv("C:/Users/Sam/Desktop/machine learning/lab3/creditscoring.csv")

### 2.1 Import the data to R and divide into training/validation/test as 50/25/25
n=dim(df)[1]
set.seed(12345) 
id=sample(1:n, floor(n*0.5)) 
train=df[id,] 

id1=setdiff(1:n, id)
set.seed(12345) 
id2=sample(id1, floor(n*0.25)) 
valid=df[id2,]

id3=setdiff(id1,id2)
test=df[id3,] 

### 2.2 Fit a decision tree to the training data
library(tree)

# Fit a decision tree using "Deviance" split =
fitDe = tree(good_bad~., data = train, split = "deviance")

# misclassification rates for the training and test data
yhat1D = predict(fitDe, newdata = train, type = "class")
tab1D = table(true=train$good_bad, pred=yhat1D)
mis1D = 1-sum(diag(tab1D))/sum(tab1D)
tab1D
mis1D 
Yfit2D = predict(fitDe, newdata = test, type = "class")
tab2D = table(true=test$good_bad, pred=Yfit2D)
mis2D = 1-sum(diag(tab2D))/sum(tab2D)
tab2D
mis2D 

# Fit a decision tree using "Gini" split = 
fitGi = tree(good_bad~., data = train, split = "gini")

yhat1G = predict(fitGi, newdata = train, type = "class")
tab1G = table(true=train$good_bad, pred=yhat1G)
mis1G = 1-sum(diag(tab1G))/sum(tab1G)
tab1G
mis1G
yhat2G = predict(fitGi, newdata = test, type = "class")
tab2G = table(true=test$good_bad, pred=yhat2G)
mis2G = 1-sum(diag(tab2G))/sum(tab2G)
tab2G
mis2G

### 2.3 Selecting optimal tree by train/validation
fitDe = tree(good_bad~., data = train, split = "deviance")

# rep(0,x) x decided by me, usually 9
trainScore = rep(0,15)
valiScore = rep(0,15)
for (i in 2:15){
  prunedTree = prune.tree(fitDe, best = i)
  pred = predict(prunedTree, newdata = valid, type = "tree")
  trainScore[i] = deviance(prunedTree)
  valiScore[i] = deviance(pred)
}

plot(2:15, trainScore[2:15], type = "b", col = "red", ylim = c(0, 600))
points(2:15, valiScore[2:15], type = "b", col = "blue")

# optimal no. of leaves
# as it is from 2 so, 3+1=4
which.min(valiScore[2:15]) + 1
# 4


# report its depth and the variables used by the tree.
# best = optimal no. of leaves
finalTree = prune.tree(fitDe, best = 4)
finalTree
summary(finalTree)

plot(finalTree)
text (finalTree , pretty = 0)

# Estimate the misclassification rate for the test data.
yhat = predict(finalTree, newdata = test, type = "class")
tab = table(ture=test$good_bad, pred=yhat)
mis = 1-sum(diag(tab))/sum(tab)
tab
mis

### 2.4 Classification using Naive Bayes
library(MASS)
library(e1071)

fitNB = naiveBayes(good_bad~., train)

# training data
yhatNB = predict(fitNB, newdata = train)
t1 = table(true=train$good_bad, pred=yhatNB)
misNB1 = 1-sum(diag(t1))/sum(t1)
t1
misNB1

# test data
yhatNBt = predict(fitNB, newdata = test)
t2 = table(true=test$good_bad,pred=yhatNBt)
misNB2 = 1-sum(diag(t2))/sum(t2)
t2
#      pred
#true   bad good
#bad   46   30
#good  49  125
misNB2

### 2.5 Repeat Naive Bayes by using loss matrix
# position of good/bad reversed!!!

# training data
library(MASS)
library(e1071)

fitNB = naiveBayes(good_bad~., train)

prob1 = predict(fitNB, newdata = train) #get class
prob1

classLoss <- function(fitNB, train){
  p <- predict(fitNB, newdata=train, type="raw") #get probabilty value
  #              bad         good
  #[1,] 7.317159e-01 0.2682840747
  #[2,] 4.136823e-02 0.9586317716
  return(factor(p[,2]/p[,1] > (10-0)/(1-0), labels=c("bad", "good")))
}
Yfit.train <- classLoss(fitNB, train)
Yfit.test <- classLoss(fitNB, test)
t3.train <- table(true=train$good_bad, pred=Yfit.train)
t3.test <- table(true=test$good_bad, pred=Yfit.test)
t3.train
t3.test

mis.train <- mean(Yfit.train != train$good_bad)
mis.test <- mean(Yfit.test != test$good_bad)
mis.train
mis.test
