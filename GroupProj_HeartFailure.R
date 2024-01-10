#Heart Failure Prediction and Analysis Using Classification Techniques - R script
#Importing the dataset
library(readr)
heart <- read_csv("heart.csv")
View(heart) #HeartDisease is the required target variable

#INSPECTING THE DATASET
dim(heart) #dimension of the dataset is 918 rows and 12 columns
str(heart) #attributes are num and chr

#DATA CLEANING & PREPARATION
#Converting categorical attributes to factor datatype
#attributes that are categorical are represented as numerical or character in the original dataset.
#these attributes need to be converted to factor datatype
heart$HeartDisease <- as.factor(heart$HeartDisease) #target variable
heart$Sex <- as.factor(heart$Sex)
heart$ChestPainType <- as.factor(heart$ChestPainType)
heart$FastingBS <-  as.factor(heart$FastingBS)
heart$RestingECG <-  as.factor(heart$RestingECG)
heart$ExerciseAngina <-  as.factor(heart$ExerciseAngina)
heart$ST_Slope <-  as.factor(heart$ST_Slope)
#there are 5 numerical variables: Age, RestingBP, Cholesterol, MaxHR, OldPeak
#there are 7 categorical variables (including the target variable, HeartDisease):
#Sex, ChestPainType, FastingBS, RestingECG, ExerciseAngina, ST_Slope
which(is.na(heart)) #no null values returned
unique(heart) #no duplicate values
#the dataset is clean with no missing values or duplicate values. 
dim(heart)
#dimemsions of the dataset remain as 918 rows and 12 columns
#NUMERICAL VARIABLES:
#Constructing a side-by-side box plot for the relevant numerical variables to look for outliers
boxplot(heart$Cholesterol, heart$RestingBP, heart$MaxHR, heart$Oldpeak,
        main = "Boxplot for Numerical Variables",
        names = c("Cholesterol", "Resting BP", "Max HR", "Old Peak"))
#from initial analysis, it is seen that Cholesterol has a very high number of outliers
heart$Cholesterol
#looking closely at this column, it is seen that there are a very high number of values that are 0. This is unusual and is more than likely a data collection error.
#we infer that the data collection was done by mistakenly putting a value of 0 instead of NA where data was not properly collected
#to fix this, imputation strategy is used and the median value is imputed into the values that are 0
median(heart$Cholesterol) #median = 223
heart$Cholesterol <- ifelse(heart$Cholesterol == 0, median(heart$Cholesterol), heart$Cholesterol)
#usually, data with any outliers would be treated using imputation or omission strategy
#however, since it is medical data it is important to take the nature and context of the data into account
#in medical data, the outliers can represent valid and abnormal health conditions that need attention
#treating it as an outlier does not make sense in this case as valuable data would be lost
#for cholesterol, the normal value is less than 120 mm/dl. Anything greater than this is a sign of high cholesterol.
#resting blood pressure is in the normal range when it is 120/20 (systolic = 120, diastolic = 80), so variations are a sign of high/low BP.
#similarly, normal range for MaxHR varies with age, so it is not correct to treat the range of MaxHR in this dataset as outliers, since age varies from 28 to 77 in it
#Old Peak is actually a term for ST Depression. The baseline is 0 which indicates a healthy heart, but anything greater than 0 
#i.e. a deflection from the baseline shows ST depression, which is a sign of the muscle not receiving enough oxygen, amongst other factors.
boxplot(heart$Cholesterol,
        main = "Boxplot for Cholesterol"
)
#the 0's are removed, and only the valid values remain
#CATEGORICAL VARIABLES:
#the categorical variables are all clean, with no transformation required.
#they have already been converted into the 'factor' datatype for ease of use

#CLASSIFICATION MODELS
#The models chosen are classification models - logistic regression and classification tree - since the target variable 'HeartDisease' is binary and categorical.
options(scipen=999) #turn off scientific notation
#Loading the necessary packages to run the model 
#install them first if not installed already
install.packages("caret")
install.packages("gains")
install.packages("rpart")
install.packages("rpart.plot")
install.packages("pROC")
library(caret)
library(gains)
library(rpart)
library(rpart.plot)
library(pROC)

#MODEL-1: LOGISTIC REGRESSION
#Data partitioning (ratio of observations in the training set to validation set = 70:30)
set.seed(1) #random seed to randomly place observations between the training and validation set without any bias
myIndex <- createDataPartition(heart$HeartDisease, p=0.7, list=FALSE)
trainingSet <- heart[myIndex,]
dim(trainingSet) #dimensions of training set is 643 rows and 12 columns
validationSet <- heart[-myIndex,]
dim(validationSet) #dimensions of validation set is 275 rows and 12 columns

#Training Set
#Initially running a logistic model on the training set with all attributes to identify which attributes are significant
logisticRegModel <- glm(HeartDisease ~ Age + Sex + ChestPainType + RestingBP + Cholesterol + FastingBS + RestingECG +
                          MaxHR + ExerciseAngina + Oldpeak + ST_Slope, 
                        family = binomial, 
                        data = trainingSet)
summary(logisticRegModel)
#looking at the p-values, the attributes that are significant (at p<0.05) are: Sex, ChestPainType, FastingBS, ExerciseAngina, Oldpeak, ST_Slope, Max HR
#running the logistic regression model using only the significant attributes
logisticRegModel2 <- glm(HeartDisease ~Sex + ChestPainType + FastingBS +
                           ExerciseAngina + MaxHR + Oldpeak + ST_Slope, 
                         family = binomial, 
                         data = trainingSet)
summary(logisticRegModel2) #note that all attributes in this model are significant at p<0.05
#coefficients of the logistic model
cf <- coef(logisticRegModel2)
#partial effect of each predictor variable on the odds
(exp(cf[-1])-1)*100 #except for intercept

#SexM ChestPainTypeATA ChestPainTypeNAP  ChestPainTypeTA       FastingBS1  ExerciseAnginaY 
#406.397976       -87.875662       -81.770468       -75.961989       261.725657        72.582232 
#MaxHR          Oldpeak     ST_SlopeFlat       ST_SlopeUp 
#-1.045282        51.352274       295.697592       -60.058704 

#interpretation of results: 
#Males are more than 406% more susceptible to heart disease than females
#Having high fasting blood sugar (i.e greater than 120mg/dl) gives a 261% more chance to have heart disease than those with normal fasting BS
#Those with Exercise Angina, which is chest pain due to exercise, have a 146% more chance to have heart disease than those who do not
#Those with OldPeak, or ST Depression > 0, which is abnormal, have 51% more likelihood to be at risk to heart disease
#Those with a flat ST_Slope are at 295% more risk to heart disease, while those with a normal uploping ST slope are 66% less likely to have it

#Accuracy, Sensitivity and Specificity
pHatLog <- predict(logisticRegModel2, trainingSet,type = "response")
yHatLog <- ifelse(pHatLog >= 0.5, 1,0)
sprintf("Accuracy for Training Set of Model-1 = %f",100 * mean(trainingSet$HeartDisease == yHatLog))
#Accuracy is 86.625194%
yTP2 <- ifelse(yHatLog == 1 & trainingSet$HeartDisease == 1, 1, 0)
yTN2 <- ifelse(yHatLog == 0 & trainingSet$HeartDisease == 0, 1, 0)
sprintf("Sensitivity for Training Set of Model-1 = %f",100*(sum(yTP2)/sum(trainingSet$HeartDisease==1)))
#Sensitivty is 89.606742%
sprintf("Specificity for Training Set of Model-1 = %f",100*(sum(yTN2)/sum(trainingSet$HeartDisease==0)))
#Specificity  is 82.926829%

#Validation Set
#Performance Measures for Logistic Regression Model
#Accuracy, Sensitivity and Specificity
pHatLog <- predict(logisticRegModel2, validationSet,type = "response")
yHatLog <- ifelse(pHatLog >= 0.5, 1,0)
sprintf("Accuracy for Validation Set of Model-1 = %f",100 * mean(validationSet$HeartDisease == yHatLog))
#Accuracy is 88.00%: this means approx 88.00% of observations are classified correctly
yTP2 <- ifelse(yHatLog == 1 & validationSet$HeartDisease == 1, 1, 0)
yTN2 <- ifelse(yHatLog == 0 & validationSet$HeartDisease == 0, 1, 0)
sprintf("Sensitivity for Validation Set of Model-1 = %f",100*(sum(yTP2)/sum(validationSet$HeartDisease==1))) 
#Sensitivity is 91.447368%: this means that approx 91.45% of target class cases are classified correctly
sprintf("Specificity for Validation Set of Model-1 = %f",100*(sum(yTN2)/sum(validationSet$HeartDisease==0))) 
#Specificity  is 83.739837%: this means that approx 83.74% of non-target class cases are classified correctly

#Confusion Matrix 
validationSet$HeartDisease <- as.factor(validationSet$HeartDisease)
yHatLog <- as.factor(yHatLog)
conf_matrix <-confusionMatrix(yHatLog, validationSet$HeartDisease) #data is pred, ref is actual
conf_matrix
#Positive Pred Value, also called Precision is 0.8879: this means that approx 88.79% of predicted target case classes belong to the target class
conf_matrix_df <- as.data.frame(as.table(conf_matrix))
# Creating the confusion matrix plot using ggplot
#Install the package if not done already
install.packages("ggplot2")
#Load the ggplot2 package
library(ggplot2)
ggplot(conf_matrix_df, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile(color = "black") +
  geom_text(aes(label = sprintf("%d", Freq)), vjust = 1) +
  scale_fill_gradient(low = "white", high = "darkgreen") +
  labs(x = "Predicted", y = "Actual", fill = "Frequency") +
  theme_minimal()

#from the confusion matrix: TP = 139, TN = 103, FP = 20, FN = 13
totObs <-139+103+20+13
misclassification_rate <- (20 + 13) / totObs 
misclassification_rate
#misclassification rate is 0.12: this means that 12% of observations are incorrectly classified

#MODEL-2: CLASSIFICATION TREE
#Data partitioning (ratio of observations in the training set to validation set = 70:30)
set.seed(1) #random seed to randomly place observations between the training and validation set without any bias
myIndex <- createDataPartition(heart$HeartDisease, p=0.7, list=FALSE)
trainingSet <- heart[myIndex,]
dim(trainingSet) #dimensions of training set is 643 rows and 12 columns
validationSet <- heart[-myIndex,]
dim(validationSet) #dimensions of validation set is 275 rows and 12 columns

#Full Classification Tree on the Training Set
set.seed(1)
full_tree <- rpart(HeartDisease ~ ., data = trainingSet, method = "class", cp = 0, minsplit = 2, minbucket = 1)
#plotting the full classification tree
prp(full_tree, type = 1, extra = 1, under = TRUE)
#printing the complexity parameters for the full tree
printcp(full_tree)
#minimum error tree has CP = 0.0243902, xerror = 0.37631 with 7 splits
#The pruned tree has the same xerror but only 5 splits, so it is preferred, with CP = 0.0243902

#Pruned Tree
pruned_tree <- prune(full_tree, cp = 0.0243903) #CP slightly bigger than given to account for rounding errors
prp(pruned_tree, type = 1, extra = 1, under = TRUE)
#one root node: ST_Slope
#three interior nodes: Chest Pain Type, OldPeak
#four leaf nodes: HeartDisease (1,0) - note that they are not pure subsets

#Performance Measures in Classification tree on validation set
#Using the pruned tree to predict in the validation set
validationSet$HeartDisease <- as.factor(validationSet$HeartDisease)
predicted_class <- predict(pruned_tree, validationSet, type = "class")
conf_mat2<- confusionMatrix(predicted_class, validationSet$HeartDisease, positive = "1")
conf_mat2
#Accuracy is 0.8509: this means that approx 85.09% of observations are classified correctly
#Sensitivity is 0.9342: this means that approx 93.42% of target class cases are classified correctly
#Specificity is 0.7480: this means that approx 74.80% of non-target class cases are classified correctly
#Positive Pred Value, also called Precision is 0.8208: this means that approx 82.08% of predicted target case classes belong to the target class

#Plotting the confusion matrix
conf_matrix_df2 <- as.data.frame(as.table(conf_mat2))
ggplot(conf_matrix_df2, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile(color = "black") +
  geom_text(aes(label = sprintf("%d", Freq)), vjust = 1) +
  scale_fill_gradient(low = "white", high = "darkgreen") +
  labs(x = "Predicted", y = "Actual", fill = "Frequency") +
  theme_minimal()
#from the confusion matrix: TP = 142, TN = 92, FP = 31, FN = 10
totObs <- 142 + 92 + 31 + 10
misclassification_rate <- (31 + 10) / totObs 
misclassification_rate
#misclassification rate is 0.1490909: this means that approx 14.91% of observations are incorrectly classified
#Finding probability of predicted class
#the sensitivity is quite high, so the default cutoff of 0.5 for finding probability is sufficient
predicted_prob <- predict(pruned_tree, validationSet, type= 'prob')
head(predicted_prob)

#To evaluate model performance independent of the cutoff value, the cumulative lift chart, decile chart and ROC Curve are used
#Cumulative Lift Chart
validationSet$HeartDisease <- as.numeric(as.character(validationSet$HeartDisease))
gains_table <- gains(validationSet$HeartDisease, predicted_prob[,2])
gains_table
#plotting the lift chart
plot(c(0, gains_table$cume.pct.of.total*sum(validationSet$HeartDisease)) ~ c(0, gains_table$cume.obs), xlab = 'Number of cases', ylab = "Cumulative", type = "l")
lines(c(0, sum(validationSet$HeartDisease))~c(0, dim(validationSet)[1]), col="red", lty=2)
#the classification model shows superior predictive power when compared to the baseline model
#Decile Chart
barplot(gains_table$mean.resp/mean(validationSet$HeartDisease), names.arg=gains_table$depth, xlab="Percentile", ylab="Lift", ylim=c(0, 3.0), main="Decile-Wise Lift Chart")
#interpretation of graph: 
#The top 6% of individuals with the highest predicted probability of having heart disease can be correctly captured as having heart disease by over 1.5 times as compared to if 6% of individuals are randomly selected
#ROC curve
roc_object <- roc(validationSet$HeartDisease, predicted_prob[,2])
plot.roc(roc_object)
auc(roc_object)
#interpretation of graph:
#again this curve shows that the classification model outperforms the baseline model in terms of its sensitivity and specificity  across all cutoff values
#Area under the curve = 0.8349 also reinforces this finding

#MODEL-3: LOGISTIC REGRESSION
#Using only the attributes from the pruned tree in Model-2 as the predictor variables for the logistic regression model
#Data Partitioning (ratio of observations in the training set to validation set = 70:30)
set.seed(1) #random seed to randomly place observations between the training and validation set without any bias
myIndex <- createDataPartition(heart$HeartDisease, p=0.7, list=FALSE)
trainingSet <- heart[myIndex,]
dim(trainingSet) #dimensions of training set is 643 rows and 12 columns
validationSet <- heart[-myIndex,]
dim(validationSet) #dimensions of validation set is 275 rows and 12 columns

#Training Set
#the attributes chosen are: ST_Slope, Cholestrol
#running the logistic regression model with these as the predictor variables
logisticRegModel4 <- glm(HeartDisease ~ST_Slope + Oldpeak + ChestPainType, 
                         family = binomial, 
                         data = trainingSet)
summary(logisticRegModel4)
#note that all parameters except and Resting BP are significant at p<0.05
#coefficients of the logistic model
cf <- coef(logisticRegModel4)
#partial effect of each predictor variable on the odds
(exp(cf[-1])-1)*100 #except for intercept
#ST_SlopeFlat       ST_SlopeUp          Oldpeak ChestPainTypeATA 
#168.41495        -75.35189         52.71701        -92.22435 
#ChestPainTypeNAP  ChestPainTypeTA 
#-85.57153        -82.76344 

#interpretation of results: 
#Those with OldPeak, or ST Depression > 0, which is abnormal, have 52% more liklihood to be at risk to heart disease
#Those with a flat ST_Slope are at 168% more risk to heart disease, however this was seen as not significant in the p-values, so it will not be given full consideration in this model

#Accuracy, Sensitivity and Specificity
pHatLog <- predict(logisticRegModel4, trainingSet,type = "response")
yHatLog <- ifelse(pHatLog >= 0.5, 1,0)
sprintf("Accuracy for Training Set of Model-3 = %f",100 * mean(trainingSet$HeartDisease == yHatLog)) 
#Accuracy is 83.203733%
yTP2 <- ifelse(yHatLog == 1 & trainingSet$HeartDisease == 1, 1, 0)
yTN2 <- ifelse(yHatLog == 0 & trainingSet$HeartDisease == 0, 1, 0)
sprintf("Sensitivity for Training Set of Model-3 = %f",100*(sum(yTP2)/sum(trainingSet$HeartDisease==1))) 
#Sensitivity is 86.797753%
sprintf("Specificity for Training Set of Model-3 = %f",100*(sum(yTN2)/sum(trainingSet$HeartDisease==0))) 
#Specificity is 78.745645%

#Validation Set
#Performance Measures for Logistic Regression Model
#Accuracy, Sensitivity and Specificity
pHatLog <- predict(logisticRegModel4, validationSet,type = "response")
yHatLog <- ifelse(pHatLog >= 0.5, 1,0)
sprintf("Accuracy for Validation Set of Model-3 = %f",100 * mean(validationSet$HeartDisease == yHatLog))
#Accuracy is 82.909091%: this means that approx 82.91% of observations are correctly classified
yTP2 <- ifelse(yHatLog == 1 & validationSet$HeartDisease == 1, 1, 0)
yTN2 <- ifelse(yHatLog == 0 & validationSet$HeartDisease == 0, 1, 0)
sprintf("Sensitivity for Validation Set of Model-3 = %f",100*(sum(yTP2)/sum(validationSet$HeartDisease==1))) 
#Sensitivity is 86.842105%: this means that approx 86.84% of target class cases are correctly classified 
sprintf("Specificity for Validation Set of Model-3 = %f",100*(sum(yTN2)/sum(validationSet$HeartDisease==0))) 
#Specificity is 78.048780%: this means that approx 78.05% of non-target class cases are correctly classified 

#Confusion Matrix
validationSet$HeartDisease <- as.factor(validationSet$HeartDisease)
yHatLog <- as.factor(yHatLog)
conf_matrix <-confusionMatrix(yHatLog, validationSet$HeartDisease) #data is pred, ref is actual
conf_matrix
#Positive Pred Value, also called Precision is 0.8276: this means that approx 82.76% of predicted target case classes belong to the target class
conf_matrix_df <- as.data.frame(as.table(conf_matrix))
# Create the ggplot
ggplot(conf_matrix_df, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile(color = "black") +
  geom_text(aes(label = sprintf("%d", Freq)), vjust = 1) +
  scale_fill_gradient(low = "white", high = "darkgreen") +
  labs(x = "Predicted", y = "Actual", fill = "Frequency") +
  theme_minimal()
#From the confusion matrix: TP = 132, TN = 96, FP = 27, FN = 20
totObs <- 132 + 96 + 27 + 20
misclassification_rate <- (27 + 20) / totObs 
misclassification_rate
#misclassification rate is 0.1709091: this means that approx 15.64% of observations are incorrectly classified

#MODEL-4: CLASSIFICATION TREE
#Decreasing the training set to by 10% to see if there are any improvements in accuracy
#Data Partitioning (ratio of observations in the training set to validation set = 60:40)
set.seed(1)
myIndex <- createDataPartition(heart$HeartDisease , p=0.6, list=FALSE)
trainingSet <- heart[myIndex,]
dim(trainingSet) #dimensions of training set is 551 rows and 12 columns
validationSet <- heart[-myIndex,]
dim(validationSet) #dimensions of validation set is 367 rows and 12 columns

#Full Classification Tree on the Training Set
set.seed(1)
full_tree <- rpart(HeartDisease ~ ., data = trainingSet, method = "class", cp = 0, minsplit = 2, minbucket = 1)
#plotting the full classification tree
prp(full_tree, type = 1, extra = 1, under = TRUE)
#printing the complexity parameters for the full tree
printcp(full_tree)
#minimum error tree has CP = 0.0081301 with xerror = 0.39024 and 6 splits
#In this case, the minimum error tree is also the pruned tree

#Pruned Tree
pruned_tree <- prune(full_tree, cp = 0.0081302) #CP slightly bigger than given to account for rounding errors
prp(pruned_tree, type = 1, extra = 1, under = TRUE)
#one root node: ST_Slope 
#five interior node: ChestPainType, OldPeak, Sex
#seven leaf nodes
#Using the pruned tree to predict in the validation set
validationSet$HeartDisease <- as.factor(validationSet$HeartDisease)
predicted_class <- predict(pruned_tree, validationSet, type = "class")
conf_mat2<- confusionMatrix(predicted_class, validationSet$HeartDisease, positive = "1")
conf_mat2
#Accuracy is 0.8529: this means that approx 85.29% of observations are classified correctly
#Sensitivity is 0.9064: this means that approx 90.64% of target class cases are classified correctly
#Specificity is 0.7866: this means that approx 78.66% of non-target class cases are classified correctly
#Positive Pred Value, also called Precision is 0.8402: this means that approx 84.02% of predicted target case classes belong to the target class

#Plotting the confusion matrix
conf_matrix_df2 <- as.data.frame(as.table(conf_mat2))
ggplot(conf_matrix_df2, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile(color = "black") +
  geom_text(aes(label = sprintf("%d", Freq)), vjust = 1) +
  scale_fill_gradient(low = "white", high = "darkgreen") +
  labs(x = "Predicted", y = "Actual", fill = "Frequency") +
  theme_minimal()
#from the confusion matrix: TP = 184, TN = 129, FP = 35, FN = 19
totObs <- 184 + 129 + 35 + 19
misclassification_rate <- (35 + 19) / totObs 
misclassification_rate
#misclassification rate is 0.147139: this means that approx 14.71% of observations are incorrectly classified

#Finding probablity of predicted class
predicted_prob <- predict(pruned_tree, validationSet, type= 'prob')
head(predicted_prob)

#To evaluate model performance independent of the cutoff value, the cumulative lift chart, decile chart and ROC Curve are used
#Cumulative Lift Chart
validationSet$HeartDisease <- as.numeric(as.character(validationSet$HeartDisease))
gains_table <- gains(validationSet$HeartDisease, predicted_prob[,2])
gains_table
#plotting the lift chart
plot(c(0, gains_table$cume.pct.of.total*sum(validationSet$HeartDisease)) ~ c(0, gains_table$cume.obs), xlab = 'Number of cases', ylab = "Cumulative", type = "l")
lines(c(0, sum(validationSet$HeartDisease))~c(0, dim(validationSet)[1]), col="red", lty=2)
#the classification model shows superior predictive power when compared to the baseline model
#Decile chart
barplot(gains_table$mean.resp/mean(validationSet$HeartDisease), names.arg=gains_table$depth, xlab="Percentile", ylab="Lift", ylim=c(0, 3.0), main="Decile-Wise Lift Chart")
#interpretation of graph: 
#The top 50% of individuals with the highest predicted probability of having heart disease can be correctly captured as having heart disease by about 1.5 times as compared to if 50% of individuals are randomly selected.
#ROC Curve
roc_object <- roc(validationSet$HeartDisease, predicted_prob[,2])
plot.roc(roc_object)
auc(roc_object)
#interpretation of graph:
#again this curve shows that the classification model outperforms the baseline model in terms of its sensitivity and specificity  across all cutoff values
#Area under the curve = 0.8649 also reinforces this finding

#MODEL EVALUATION
#Comparison of models using their performance measures
Accuracy <- c(88, 85.09, 82.91, 85.29)
Sensitivity <- c(91.45, 93.42, 86.84, 90.64)
Specificity <- c(83.74, 74.8, 78.04, 78.66)
df <- data.frame(Accuracy, Sensitivity, Specificity)
leg_colors <- c("salmon", "lightblue", "lightgreen", "orange")
#barplot to visualize the performance across models
barplot(as.matrix(df),
        main = "Comparing the Performance Measures Across Models",
        beside = TRUE, #plots vertical bars
        ylab = "Value (in %)",
        ylim = c(0,100), #range of y-axis scale,
        col = leg_colors
)
legend(x = "bottomright",
       legend = c("Model-1", "Model-2", "Model-3", "Model-4"),
       fill = leg_colors,
       title = "Legend")
#interpreting the results: 
#Based on the models built, I would recommend Model-1 as the recommended model. 
#Accuracy of predicting in the validation set is the most important measure of a robust model, and Model-1 has the highest accuracy at 88%. 
#The sensitivity and specificity of Model-1, sre also relatively high at 91.45% and 83.74% respectively.
#Keeping in mind that the business problem is to identify those with an elevated risk of heart disease, sensitivity is a very important measure, as it measures the proportion of correctly classified target cases i.e. those who are at risk for heart disease. 
#Thus, Model-1 is recommended.

#Appendix:
#Finding relationships between the statistically significant variables by plotting and visualizing them
#stacked column chart with sex v st slope
contingency_table <- table(heart$Sex, heart$ST_Slope) #freq dist
View(contingency_table)
prop.cont.table <- prop.table(contingency_table) #proportion
View(prop.cont.table)
barplot(table(heart$Sex, heart$ST_Slope), 
        col= c("pink", "blue"), 
        xlab = "ST_Slope Type",
        ylab = "Frequency of Cases",
        main = "Plot of ST Slope vs Sex",
        legend = rownames(contingency_table))
#Of the 57.1% of individuals with abnormal ST_Slopes i.e. down sloping & flat, 48.03% of them are male

#stacked column with heart disease v sex
contingency_table <- table(heart$HeartDisease, heart$Sex) #freq dist
View(contingency_table)
prop.cont.table <- prop.table(contingency_table) #proportion
View(prop.cont.table)
barplot(table(heart$HeartDisease, heart$Sex), 
        col= c("pink", "blue"), 
        xlab = "Sex",
        ylab = "Frequency of Cases",
        main = "Plot of Sex vs Heart Disease",
        legend = rownames(contingency_table))
#49.8% of people having heart disease are males, in comparison to 5.44% females - almost 10 times as likely!
#MALES ARE MORE SUSPECTIBLE TO HEART DISEASE

#box plot of age v heart disease
boxplot(Age ~ HeartDisease, data = heart, 
        xlab = "Heart Disease", 
        ylab = "Age (in years)",
        main = "Box Plot of Age by Heart Disease")
#the mean age of those with heart disease is higher than those without any heart disease. 
#Those who have heart disease but are younger (less than 35) are shown as outliers, which agrees with the historical trend of older people usually being at risk of heart attacks
