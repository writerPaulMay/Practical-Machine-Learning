# Practical Machine Learning Assignment

## Set up environment and load source data  
First we set up the environment, load the libraries we will need and read in the source data.  

```r
if(getwd()!="/home/paul/coursework") {setwd("/home/paul/coursework")}  
library(tidyverse)
```

```
## Loading tidyverse: ggplot2
## Loading tidyverse: tibble
## Loading tidyverse: tidyr
## Loading tidyverse: readr
## Loading tidyverse: purrr
## Loading tidyverse: dplyr
```

```
## Conflicts with tidy packages ----------------------------------------------
```

```
## filter(): dplyr, stats
## lag():    dplyr, stats
```

```r
library(caret)
```

```
## Loading required package: lattice
```

```
## 
## Attaching package: 'caret'
```

```
## The following object is masked from 'package:purrr':
## 
##     lift
```

```r
library(rpart)
library(corrplot)
library(rattle)
```

```
## Rattle: A free graphical interface for data mining with R.
## Version 4.1.0 Copyright (c) 2006-2015 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
library(parallel)
library(doParallel)
```

```
## Loading required package: foreach
```

```
## 
## Attaching package: 'foreach'
```

```
## The following objects are masked from 'package:purrr':
## 
##     accumulate, when
```

```
## Loading required package: iterators
```

```r
library(gridExtra)  
```

```
## 
## Attaching package: 'gridExtra'
```

```
## The following object is masked from 'package:dplyr':
## 
##     combine
```

```r
rawData <- read.csv("MachineLearningAssessment/pml-training.csv", na.strings=c("NA", "", "#DIV/0!"))  
AssignmentData <- read.csv("MachineLearningAssessment/pml-testing.csv", na.strings=c("NA", "", "#DIV/0!")) 
```
## Tidy the source data  
After exploratory data analysis, we decided to remove derived data from the data set, along with irrelevant identifiers and columns with over half missing values.    

```r
rawData <- select(rawData, -starts_with("avg")) ## remove derived columns
rawData <- rawData[, -(1:7)] ## remove irrelevant identifiers
nearZeros <- nearZeroVar(rawData, saveMetrics=TRUE)  ## find columns with near-zero values  
rawData <- rawData[, !nearZeros$nzv] ## remove columns with near-zero values
naMeans <- colMeans(sapply(rawData, is.na)) ## find the mean number of NAs in eachcolumn
rawData <- rawData[, naMeans<=0.5] ## remove columns with more than 50% NAs
```
## Create new training and testing data sets
We now split the tidied source data into a training and testing set. We will reserve the testing set for evaluating the models.  

```r
inTrain <- createDataPartition(y=rawData$classe, p=0.75, list=FALSE)
training <- rawData[inTrain,]
testing <- rawData[-inTrain,]  
```
## Look for correlations
Here we look to see if any of the predictors are correlated.  

```r
trCor <- cor(training[, -53]) 
topCor <- findCorrelation(trCor, cutoff = 0.75)
names(training)[topCor]  
```

```
##  [1] "accel_belt_z"      "roll_belt"         "accel_arm_y"      
##  [4] "accel_belt_y"      "total_accel_belt"  "accel_dumbbell_z" 
##  [7] "accel_belt_x"      "pitch_belt"        "magnet_dumbbell_x"
## [10] "accel_dumbbell_y"  "magnet_dumbbell_y" "accel_dumbbell_x" 
## [13] "accel_arm_x"       "accel_arm_z"       "magnet_arm_y"     
## [16] "magnet_belt_z"     "accel_forearm_y"   "gyros_arm_x"
```
This tells us there are clusters of correlated predictors. A plot will help show this.  

```r
corrplot(trCor, method="square", tl.cex=0.6, tl.col="black")  
```

![](Practical_Machine_Learning_Assignment_writerPaulMay_files/figure-html/plot_correlations-1.png)<!-- -->
   
## Decision Tree model  

```r
dtTraining <- train(classe ~ ., data=training, method="rpart")  
fancyRpartPlot(dtTraining$finalModel)
```

![](Practical_Machine_Learning_Assignment_writerPaulMay_files/figure-html/decision_tree_model-1.png)<!-- -->
  
The decision tree runs quickly and has an elegant structure. But how accurate is is?  

```r
confusionMatrix(dtTraining)  
```

```
## Bootstrapped (25 reps) Confusion Matrix 
## 
## (entries are percentual average cell counts across resamples)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 25.3  8.5  8.2  7.4  3.1
##          B  0.6  6.1  0.7  2.6  2.6
##          C  2.0  3.7  7.7  3.7  3.6
##          D  0.4  0.8  0.8  2.8  0.8
##          E  0.1  0.0  0.0  0.0  8.5
##                             
##  Accuracy (average) : 0.5041
```
The accuracy on the training set about fifty-fifty - as good as tossing a coin. We can expect it to be worse on new data.  
  
## Linear Discriminant Analysis model  

```r
ldaTraining <- train(classe ~ ., data=training, method="lda")  
```

```
## Loading required package: MASS
```

```
## 
## Attaching package: 'MASS'
```

```
## The following object is masked from 'package:dplyr':
## 
##     select
```

```r
ldaTraining  
```

```
## Linear Discriminant Analysis 
## 
## 14718 samples
##    52 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 14718, 14718, 14718, 14718, 14718, 14718, ... 
## Resampling results:
## 
##   Accuracy   Kappa    
##   0.6977584  0.6175572
```
This model is produced fairly fast, but its accuracy is still low. 
  
## Random Forest model  
We use parallel processing to reduce the run time of this model.

```r
cluster <- makeCluster(detectCores() - 1)  
registerDoParallel(cluster)  
fitControl <- trainControl(method = "cv", number=10, preProcOptions="pca", allowParallel = TRUE)  
rfTraining <- train(classe ~ ., data=training, method="rf", trControl=fitControl)  
```

```
## Loading required package: randomForest
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:gridExtra':
## 
##     combine
```

```
## The following object is masked from 'package:dplyr':
## 
##     combine
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
stopCluster(cluster)  
registerDoSEQ()  
rfTraining  
```

```
## Random Forest 
## 
## 14718 samples
##    52 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 13245, 13246, 13248, 13244, 13246, 13247, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9932737  0.9914911
##   27    0.9929335  0.9910606
##   52    0.9860029  0.9822922
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```
Random Forest is the best of these three models.

## Comparison of model performance
Let's look at the accuracy scores of the three models when using the testing set.

```r
dtPred <- predict(dtTraining, newdata=testing)  
ldaPred <- predict(ldaTraining, newdata=testing)  
rfPred <- predict(rfTraining, newdata=testing)  
confusionMatrix(dtPred, testing$classe)  
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1271  376  407  346  139
##          B   26  327   23  138  120
##          C   96  246  425  320  252
##          D    0    0    0    0    0
##          E    2    0    0    0  390
## 
## Overall Statistics
##                                          
##                Accuracy : 0.492          
##                  95% CI : (0.478, 0.5061)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.3364         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9111  0.34457  0.49708   0.0000  0.43285
## Specificity            0.6386  0.92238  0.77427   1.0000  0.99950
## Pos Pred Value         0.5006  0.51577  0.31740      NaN  0.99490
## Neg Pred Value         0.9476  0.85433  0.87938   0.8361  0.88675
## Prevalence             0.2845  0.19352  0.17435   0.1639  0.18373
## Detection Rate         0.2592  0.06668  0.08666   0.0000  0.07953
## Detection Prevalence   0.5177  0.12928  0.27304   0.0000  0.07993
## Balanced Accuracy      0.7749  0.63347  0.63567   0.5000  0.71618
```

```r
confusionMatrix(ldaPred, testing$classe)  
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1146  150  108   45   40
##          B   42  626   72   38  147
##          C  104  106  567  105   91
##          D  101   23   89  588   88
##          E    2   44   19   28  535
## 
## Overall Statistics
##                                          
##                Accuracy : 0.706          
##                  95% CI : (0.693, 0.7187)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.6275         
##  Mcnemar's Test P-Value : < 2.2e-16      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8215   0.6596   0.6632   0.7313   0.5938
## Specificity            0.9023   0.9244   0.8997   0.9266   0.9768
## Pos Pred Value         0.7696   0.6768   0.5827   0.6614   0.8519
## Neg Pred Value         0.9271   0.9188   0.9267   0.9462   0.9144
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2337   0.1277   0.1156   0.1199   0.1091
## Detection Prevalence   0.3036   0.1886   0.1984   0.1813   0.1281
## Balanced Accuracy      0.8619   0.7920   0.7814   0.8290   0.7853
```

```r
confusionMatrix(rfPred, testing$classe)  
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1390    3    0    0    0
##          B    4  944    2    0    0
##          C    0    2  850   14    0
##          D    0    0    3  790    0
##          E    1    0    0    0  901
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9941         
##                  95% CI : (0.9915, 0.996)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9925         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9964   0.9947   0.9942   0.9826   1.0000
## Specificity            0.9991   0.9985   0.9960   0.9993   0.9998
## Pos Pred Value         0.9978   0.9937   0.9815   0.9962   0.9989
## Neg Pred Value         0.9986   0.9987   0.9988   0.9966   1.0000
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2834   0.1925   0.1733   0.1611   0.1837
## Detection Prevalence   0.2841   0.1937   0.1766   0.1617   0.1839
## Balanced Accuracy      0.9978   0.9966   0.9951   0.9909   0.9999
```
To get a sense of how the models performed, we can plot the actual values of `classe` against the predicted values in each model.  

```r
dtPlot <- qplot(dtPred, colour=classe, xlab="Decision Tree", data=testing)  
ldaPlot <- qplot(ldaPred, colour=classe, xlab="Linear Discriminant Analysis", data=testing)  
rfPlot <- qplot(rfPred, colour=classe, xlab="Random Forest", data=testing)  
grid.arrange(dtPlot, ldaPlot, rfPlot, ncol=3, bottom="Comparison of Models")  
```

![](Practical_Machine_Learning_Assignment_writerPaulMay_files/figure-html/plot_model_comparison-1.png)<!-- -->
The plot shows that the Decision Tree fails to assign any observations to classe D - so its performance could even be said to be worse than random. 
  
The Linear Discriminant Analysis model appears more uniform in its errors.  
  
The fit of the Random Forest model is extremely close. The Random Forest model cannot be interpreted and it is computationally intensive, but it produces the best results.  
  
## Addendum: Process the raw data for the Assignment test  
The 20 rows in the Assignment testing set are not applied to the model here, so as not to include the quiz answers in this output. However we include the tidying code for this data set, since we have to ensure it is in the same format as our model.  

```r
AssignmentData <- dplyr::select_(AssignmentData, .dots=colnames(rawData)[-53])    
```
  
  
