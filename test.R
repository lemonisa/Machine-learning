### Question 1: Describing individual variables ###
dat = read.table("C:/732A97 Multivariate Statistical Methods/lab/T1-9.dat", 
           header=F)
## 1a
datMean = colMeans(dat[,2:8]) 
#sapply(dat[,2:8],FUN=mean)
datSd = sapply(dat[,2:8], sd)

### 1b
par(mfrow = c(4,2))
for(i in 2:8){
  hist(dat[,i], probability = TRUE, xlab = "Variable",
       breaks=30, main=colnames(dat)[i])
  lines(density(dat[,i]), col = "red")
} 
par(mfrow = c(1,1))

## parallel coordinates
library(lattice)
parallel(dat[2:8], main="Parallel Plot of 7 variables")

# fit normal distr to 1 & 2
par(mfrow = c(4,2))
for(i in 2:8){
  hist(dat[,i], probability = TRUE, xlab = "Variable",
       breaks=30, main=colnames(dat)[i])
  lines(density(dat[,i]), col = "red")
  
  varSeq <- seq(datMean[i-1] - 4*datSd[i-1],
                datMean[i-1] + 4*datSd[i-1],
                 length = 1000)
  lines(varSeq, dnorm(varSeq, datMean[i-1], datSd[i-1]), col = "blue")
} 
par(mfrow = c(1,1))

### Question 2: Relationships between the variables ###
### 2a
datCov = cov(dat[,2:8])
datCor = cor(dat[,2:8])

### 2b Scatterplot Matrix
plot(dat[,2:8], main="Scatterplot Matrix", 
     pch=3, cex=0.5, col="blue")

### 3c
stars(dat[,2:8], main="Stars Plot")

library(aplpack)
faces(dat[,2:8], main="Faces Plot")

### Question 3: Examining for extreme values ###

### 3b Euclidean distance
unit1 = as.matrix(rep(1, nrow(dat)))
datRes = dat[,2:8]- unit1 %*% datMean #residuals

dis = sqrt(colSums(t(datRes)^2))
dat[order(dis, decreasing=T)[1:5],1]

### 3c squared distance
# centring and scaling
datNor = scale(datRes)
dis2 = sqrt(colSums(t(datNor)^2))
dat[which.max(dis2),1]

#V = diag(x=1,nrow=7,ncol=7) * sapply(datRes, var)
#dis2 = datRes %*% t(V) %*% t(datRes)
#which.max(diag(dis2))
#[1] 40

### 3d Mahalanobis distance
C = cov(dat[,2:8])
dis3 = as.matrix(datRes) %*% solve(C) %*% as.matrix(t(datRes))
which.max(diag(dis3))

# c'k top5 ranking of 3 methods
dat[order(diag(dis3), decreasing=T)[1:5],1]
dat[order(dis, decreasing=T)[1:5],1]
dat[order(dis2, decreasing=T)[1:5],1]

# Find Sweden
alldis = data.frame(
           country = dat[,1],
           unnorm_eucd = dis,
           norm_eucd = dis2,
           mahal = diag(dis3))
parallel(alldis[,2:ncol(alldis)], 
         col = ifelse(alldis$country == 'SWE', 'red', '#AAAAAA'),
         main="Parallel Plot of Distances Using 3 Methods")
