##### Assignment 1. Feature selection by cross-validation in a linear model.
linear_regresion <- function(X, Y, Nfolds){
  n = length(Y)
  k = floor(n/Nfolds)
  
  # permute 1:k-1 subsets
  # (if n=51,Nfolds=10,k=5,first 4 subsets have 10 elements)
  index_folds <- c()
  for (i in 1:k-1){
    set.seed(12345)
    # 1:10, 11:20, 21:30, 31:40
    index_folds[((i-1)*Nfolds+1):(i*Nfolds)] = sample(c(1:Nfolds), Nfolds)
  }
  
  # permute k_th/last subset
  # (if n=51,Nfolds=10,k=5,last subset has 11 elements)
  # last 41:47 sample(1:10, 11, replace = T)
  index_folds[((k-1)*Nfolds+1):n] = sample(c(1:Nfolds), n-(k-1)*Nfolds, replace = TRUE)
  index_folds 
  #> index_folds
  #51 index
  #[1]  8 10  7  9  3  1  2  4  6  5  8 10  7  9  3  1  2  4  6  5  8 10  7  9  3  1  2
  #[28]  4  6  5  8 10  7  9  3  1  2  4  6  5  8  9  8  9  5  2  4  6  8 10  1
  
  
  #model fit by using each fold as test
  cv <- c()
  for (j in 1:Nfolds){
    # TRUE OR FALSE
    if_subset = (index_folds==j)
    #> valInd 51 elements if j=1
    #[27] FALSE FALSE FALSE FALSE FALSE  TRUE FALSE FALSE FALSE FALSE FALSE FALSE FALSE
    #[40] FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE  TRUE FALSE
    
    # divide data into training data and validation data
    # training: Remove subset #i, using remaining data.
    # test: Predict for subset #i 
    
    # assume matrix X has 5 columns
    ind=sample(n,n)
    X=X[ind,]
    Y=Y[ind]
    
    testx = X[if_subset, 5]
    trainx = X[!if_subset, 5]
    testy = Y[if_subset]
    trainy = Y[!if_subset]
    
    # Estimate w for linear regression
    w_hat = solve(t(trainx) %*% trainx) %*% t(trainx) %*% trainy
    
    # Estimate w for ridge regression
    # I = diag(dim(trainx)[2])
    # W_hat = ((solve((t(trainx) %*% trainx) + Lam *I)) %*% t(trainx) )%*%trainy
    
    # Prediction
    y_hat = testx %*% w_hat
    
    #calculate sum of cross-validation score
    sum_loss = sum((testy - y_hat)^2)
    cv[j] = sum_loss
  }  
  cv = cv/n
  
  pl <- plot(1:Nfolds, cv, col="red", xlab = "index of features of subsets", ylab = "CV scores")
  
  optimal_cv <- min(cv)
  optimal_index <- which.min(cv)
  
  result <- list(cv=cv, optimal=optimal_index, optimal_cv=optimal_cv, plot=pl)
  return(result)
}

### 2. Test your function on data set swiss
data("swiss")
# Fertility: Y
# All other variables: X
# Nfolds: 5
Y <- as.matrix(swiss[,1])
X <- as.matrix(swiss[,-1])
Nfolds <- 5

linear_regresion(X=X, Y=Y, Nfolds = Nfolds)




##### Assignment 2. Linear regression and regularization

### 1. Import data to R.
data <- read.csv("C:/Users/Sam/Desktop/machine learning/lab2/tecator.csv")

# create a plot of Moisture versus Protein.
attach(data)
plot(Protein, Moisture)

### 3. Divide the data into training & validation sets( 50%/50%)
n=dim(data)[1]
set.seed(12345) 
id=sample(1:n, floor(n*0.5)) 
train=data[id,] 
test=data[-id,] 

# fit models Mi,i=1...6.
# record the training & validation Mean Squared Error (MSE)

fit1 = lm(Moisture ~ Protein, data = train)
MSE_train1 = mean(fit1$residuals^2)
pred1 = predict(fit1, test)
MSE_val1 = mean((pred1-test$Moisture)^2)

fit2 = lm(Moisture ~ poly(Protein,2,raw=TRUE), data = train)
MSE_train2 = mean(fit2$residuals^2)
pred2 = predict(fit2, test)
MSE_val2 = mean((pred2-test$Moisture)^2)

fit3 = lm(Moisture ~ poly(Protein,3,raw=TRUE), data = train)
MSE_train3 = mean(fit3$residuals^2)
pred3 = predict(fit3, test)
MSE_val3 = mean((pred3-test$Moisture)^2)

fit4 = lm(Moisture ~ poly(Protein,4,raw=TRUE), data = train)
MSE_train4 = mean(fit4$residuals^2)
pred4 = predict(fit4, test)
MSE_val4 = mean((pred4-test$Moisture)^2)

fit5 = lm(Moisture ~ poly(Protein,5,raw=TRUE), data = train)
MSE_train5 = mean(fit5$residuals^2)
pred5 = predict(fit5, test)
MSE_val5 = mean((pred5-test$Moisture)^2)

fit6 = lm(Moisture ~ poly(Protein,6,raw=TRUE), data = train)
MSE_train6 = mean(fit6$residuals^2)
pred6 = predict(fit6, test)
MSE_val6 = mean((pred6-test$Moisture)^2)

# Plot showing how training and validation MSE depend on i
MSE_train <- c(MSE_train1, MSE_train2, MSE_train3, 
               MSE_train4, MSE_train5, MSE_train6)
MSE_val <- c(MSE_val1, MSE_val2, MSE_val3, 
             MSE_val4, MSE_val5, MSE_val6)

plot(c(1,2,3,4,5,6), MSE_train, col = "blue", xlim = c(1, 6), ylim = c(25,40),
     xlab = "Power i of polynomial", ylab = "MSE of training & validation")
points(c(1,2,3,4,5,6), MSE_val, col = "red")


### 4.0 Calculate AIC values for each model Mi i=1:6
# estimate sd as the residual MSE in the AIC formula
n = dim(data)[1]
y1 = data$Moisture
x1 = data$Protein
AIC = c() 
for (i in 1:6){
  f1 = lm(y1~poly(x1, i, raw=TRUE))
  MSE = mean(f1$residuals^2)
  AIC[i] = 2*(i+1) + n*(log(2*(pi)*MSE)+1) 
  AIC 
}
AIC
plot(c(1:6), AIC, xlab = "Model", ylab = "AIC value")



### 4. Perform variable selection of a linear model by stepAIC.
# Fat is response
# Channel1-Channel100 are predictors 

# select channel1-100 and Fat from data
data1 <- data[ ,2:102]

# variable selection
library(MASS)
fit <- lm(Fat~., data=data1)
step <- stepAIC(fit, direction="both")
step$anova 
summary(step)
        
# number of variables were selected
length(step$coefficients)


### 5. Fit a Ridge regression model 
# with the same predictor and response variables.

# Channel1-Channel100
covariates <- scale(data[, 2:101])
# Fat
response <- scale(data[, 102])

# use package glmnet with alpha=0 (Ridge regression)
library(glmnet)
model1=glmnet(as.matrix(covariates), response, alpha=0,family="gaussian")
plot(model1, xvar="lambda", label=TRUE)

### 6.LASSO model
# use package glmnet with alpha=1 (LASSO)
# Channel1-Channel100
covariates <- data[, 2:101]
# Fat
response <- data[, 102]
model2=glmnet(as.matrix(covariates), response, alpha=1,family="gaussian")
plot(model2, xvar="lambda", label=TRUE)

### 7.Use cross-validation to find the optimal LASSO model
# if ridge model alpha=0
# Channel1-Channel100
covariates <- scale(data[, 2:101])
# Fat
response <- scale(data[, 102])
model3=cv.glmnet(as.matrix(covariates), response, alpha=1,family="gaussian")
model3$lambda.min
plot(model3)
coef <- coef(model3, s="lambda.min")
coef
#101 x 1 sparse Matrix of class "dgCMatrix"
#1
#(Intercept)  2.430320e-15
#Channel1     .           
#Channel2     .           
#Channel3     .   

# counte variables chosen by the model
sum = 0
for(i in 2:100){
  if (coef[i,1] != 0){
    sum = sum + 1
    sum
  }
}
print(sum)
#[1] 13