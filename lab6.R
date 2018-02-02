##### LAB 6: NEURAL NETWORKS AND DEEP LEARNING
library(neuralnet)
set.seed(1234567890)

# Sample 50 points uniformly at random in the interval [0, 10]
Var <- runif(50, 0, 10)
# Apply the sin function to each point
trva <- data.frame(Var, Sin=sin(Var))

tr <- trva[1:25,] # Training
va <- trva[26:50,] # Validation

# Random initializaiton of the weights in the interval [-1, 1]
winit <- runif(31, -1, 1)

error = c()
for(i in 1:10){
  # number of hidden neurons (vertices) in each layer: 10
  # stop the threshold automatically
  
  nn <- neuralnet(Sin ~ Var, data = tr, hidden = 10, threshold = i/1000, 
                  startweights = winit)
  
  # The validation set for early stop of the gradient descent.
  # fitted values yp by using validation dataset
  y_v = compute(nn, va$Var)
  yp = y_v$net.result
  # compute error function
  error[i] = mean((va$Sin - yp)^2)
  
}
error
plot(error, type="l", main = "MSE")


# chosen value for the threshold
index = which.min(error)
index
#[1] 4

# final neural network
plot(nn <- neuralnet(Sin ~ Var, data = tr, hidden = 10, threshold = 4/1000, 
                     startweights = winit))

# Plot of the predictions (blue dots) and the data (red dots)
plot(prediction(nn)$rep1, col="blue")
points(trva, col = "red")