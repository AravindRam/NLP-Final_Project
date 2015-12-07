library(h2o)
library(data.table)
library(Metrics)
library(fastmatch)
library(zoo)
library(caret)
h2o.init(nthreads=-1)

train_data <- fread('csvDump10.csv')
#train_data$id <- NULL
#train_cols <- train_data[, !duplicated(colnames(train_data))]
#Make score1 a factor variable for classification
#new_train_data <- train_data[, train_cols]
train_data <- train_data[, unique(names(train_data)), with=FALSE]


#Rename all the columns because it may contain duplicates
#k <- as.character(1:length(names(train_data)))
#names(train_data) <- k

#inbuild <- createDataPartition(y=train_data$`1`, p=0.8, list=FALSE)

#train_frame <- train_data[train_data$'1' != 4, ]
#test_frame <- train_data[train_data$'1' == 4, ]
#train_frame <- factor(train_frame$'1')
#test_frame <- factor(test_frame$'1')

train_frame <- train_data[train_data$score1 != 4, ]
test_frame <- train_data[train_data$score1 == 4, ]
train_frame$score1 <- factor(train_frame$score1)
test_frame <- subset( test_frame, select = -score1 )
#test_frame$score1 <- NULL

train_hex_gbm<-as.h2o(train_frame,destination_frame="train_gbm.hex")
train_hex_rf<-as.h2o(train_frame,destination_frame="train_rf.hex")

feature_cols <- names(train_data)[-1] # Remove the score1 from feature cols


GbmHex<-h2o.gbm(x = feature_cols,
                y="score1",training_frame=train_hex_gbm,model_id="gbmStarter.hex", distribution="AUTO",
                nfolds = 0,
                seed = 23887,
                ntrees = 200,
                max_depth = 30,
                min_rows = 30,
                learn_rate = 0.015)

rfHex<-h2o.randomForest(x=feature_cols,
                        y="score1",training_frame=train_hex_rf,model_id="rf.hex", ntrees=50, sample_rate = 0.7)

test_hex<-as.h2o(test_frame,destination_frame="test.hex")

predictions_gbm <- as.data.frame(h2o.predict(GbmHex,test_hex))
predictions_rf <- as.data.frame(h2o.predict(rfHex,test_hex))

soln <- fread('public_leaderboard_solution.csv')
one <- soln[soln$essay_set==10, ]
human_score <- one$essay_score

score_gbm <- ScoreQuadraticWeightedKappa(predictions_gbm$predict, human_score, 0, 3)
score_rf <- ScoreQuadraticWeightedKappa(predictions_rf$predict, human_score, 0, 3)


# Deep learning model
model <- h2o.deeplearning(x = feature_cols,  y = "score1", training_frame =train_hex_gbm, activation = "TanhWithDropout", input_dropout_ratio = 0.2, hidden_dropout_ratios = c(0.5,0.5,0.5), balance_classes = TRUE, hidden = c(50,50,50), epochs = 100) 
result_deeplearning <- h2o.predict(model, test_hex)

result_deeplearning <- as.data.frame(result_deeplearning)

score_deeplearning <- ScoreQuadraticWeightedKappa(result_deeplearning$predict, human_score, 0, 3)
