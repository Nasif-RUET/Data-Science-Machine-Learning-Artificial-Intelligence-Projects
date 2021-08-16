# preparing the dataset
data <- read.csv("C:/Users/atiks/OneDrive/Desktop/binary.csv",sep=",",header = TRUE )
str(data)
###Create training and test data set
set.seed(123)
ind <- sample(2, nrow(data), replace = TRUE, prob = c(0.7, 0.3))
training_data <- data[ind==1, ]
test_data <- data[ind==2, ]

#### Building neiral Network model
#install.packages("neuralnet")
library(neuralnet)
set.seed(333)
n <- neuralnet(admit~gre + gpa + rank,
               data = training_data,
               hidden = 1,
               err.fct = "ce",
               linear.output = FALSE)
               
#lifesign = 'full',
#rep = 5,
#algorithm = "rprop+",
# stepmax = 100000)

# plot our neural network 
plot(n)

# error
n$result.matrix

# Prediction
output <- compute(n,training_data[, -1])
head(output$net.result)
head(training_data[1, ])

##output calculations

in4=-0.08281+(1.93468*660)+(-2.05129*3.67)+(0.27774*3)
out4=1/(1+exp(-in4))
in5=-0.98842+(out4*0.26838)
output5=1/(1+exp(-in5))
output5

# confusion Matrix $classification accuracy -Training data
output <- compute(n, training_data[, -1])
p1 <- output$net.result
pred1 <- ifelse(p1 > 0.5, 1, 0)
tab1 <- table(pred1, actual=training_data$admit)
tab1

####Accuracy
sum(diag(tab1)) / sum(tab1)
# confusion Matrix $classification accuracy -Testing data
output2 <- compute(n, test_data[, -1])
p2 <- output2$net.result
pred2 <- ifelse(p2 > 0.5, 1, 0)
tab2 <- table(pred2, actual=test_data$admit)
tab2
####Accuracy
sum(diag(tab2)) / sum(tab2)

