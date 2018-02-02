##############temperature forecast for a date and place in Sweden
# predicted temperatures from 4 am to 24 pm in an interval of 2 hours.
# Use a kernel that is the sum of three Gaussian kernels

set.seed(1234567890)
library(geosphere)

stations <- read.csv("C:/Users/Sam/Desktop/machine learning/lab5/stations.csv")
temps <- read.csv("C:/Users/Sam/Desktop/machine learning/lab5/temps50k.csv")

st1 <- merge(stations,temps,by="station_number")
ind <- sample(1:50000, 5000)
st <- st1[ind,]

# define three values
h_distance <- 5
h_date <- 5
h_time <- 5

# The point to predict (up to the students)
a <- 58.4274 
b <- 14.826
# The date to predict (up to the students)
#date_p <- "2013-11-04" 
#times_p <- c("04:00:00", "06:00:00", ..., "24:00:00")  # 11 elements

# number of prediction
n = (24 - 4)/2 + 1  #11

# fliter the data posterior to the day and hour of forecast.
# row 16462 is the position of "2013-11-04" 
datep = as.numeric(st1$date[16462])

stnew = st[as.numeric(st$date) < datep, ]


################ distance 
lat = as.numeric(stnew$latitude) 
lon = as.vector(stnew$longitude) ### contain NA
lon = as.numeric(lon)

# remove NA from lon
#ind = c()
#for(i in 1:length(lon)){
#  if(is.na(lon[i])) ind = c(ind, i)
#}
#ind 
#lon = lon[-ind]
#lat = lat[-ind]

# The point to predict (up to the students)
a <- 58.4274 
b <- 14.826

# Gaussian kernels for distance
h_distance = 5
dis = numeric(length(lat))
dist_kl = c()
for(k in 1:length(lat)){
  #compute and scale the distance /10000000
  dis[k] = distHaversine(c(lat[k], lon[k]), c(a,b))/1000000 
  dist_kl[k] = exp(-(dis[k]/h_distance)^2)  
}
dist_kl      


#################### date
date = as.numeric(stnew$date)
# 1 day, no loop in 1:11
datetest = as.numeric(st1$date[16462])

# Gaussian kernels for date
h_date = 5
#compute and scale the distance /10000 
dis = abs(date - datetest)/10000   
date_kl = numeric(length(dis))
for (j in 1:length(dis)){
  date_kl[j] = exp(-(dis[j]/h_date)^2)  
}
date_kl 


################## time
time = as.vector(stnew$time)
# convert time to numeric
time = sapply(strsplit(time,":"),
              function(x) {
                x <- as.numeric(x)
                x[1]
              }
)

# 11 elements
timetest = seq(from=4, to=24, by=2)


# sum of three Gaussian kernels
gkernel = function(X, Y, Xtest, h_time){
  n = length(Xtest)
  pred = numeric(n)
  dis = numeric(n)
  for (i in 1:n){
    dis = abs(X-Xtest[i])
    kl = numeric(length(dis))
    time_kl = numeric(length(dis))
    for (j in 1: length(dis)){
      time_kl[j] = exp(-(dis[j]/h_time)^2)
    }
    #sum of three Gaussian kernels
    kl = time_kl + dist_kl[i] + date_kl[i]  
    #print(kl)
    pred[i] = sum(kl*Y)/sum(kl)
  }
  return(pred)
}

temperature = as.numeric(stnew$air_temperature) 

# call prediction function
temp = gkernel(X=time, Y=temperature, Xtest = timetest, h_time = 5) #len:11
temp

plot(temp, type="o")

