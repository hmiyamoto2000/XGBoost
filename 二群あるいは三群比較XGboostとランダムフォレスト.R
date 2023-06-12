

library(xgboost)
FCPC.train <- read.csv("A.csv") # tibble::glimpse() 
FCPC.test <- read.csv("B.csv")
dim(FCPC.train) 

train.x <- FCPC.train[, 2:102] #dim(FCPC.train) [1]  8 102の場合
x <- rbind(train.x,FCPC.test[,-1]) # x <- rbind(train.x,FCPC.train[,-1])?

y_0 <- c(FCPC.train$Species)
y_1 <- as.factor(y_0)
y <- as.integer(y_1)-1


#FCPC.train <- rbind(y_0,train.x)
#FCPC.test <- rbind(y_0,FCPC.test[,1])


x <- as.matrix(x)

#p <- FCPC.test[,2:22]
#FCPC.test <- rbind(x_1_f,p)

trind <- 1:length(y) # 先程定義したx の中の訓練データを指すのに使う
teind <- (nrow(train.x)+1):nrow(x) # 先程定義したx の中の検証用データを指すのに使う

set.seed(131) # 固定シードで試す
param <- list("objective" = "multi:softprob", # 多クラスの分類で各クラスに所属する確率を求める
              "eval_metric" = "mlogloss", # 損失関数の設定
              "num_class" = 2 # class がいくつ存在するのか  #8にしてみる
)

k<-round(1+log2(nrow(train.x)))
cv.nround <- 100 #search
bst.cv <- xgb.cv(param=param, data = x[trind,], label = y,  nfold = k, nrounds=cv.nround)

set.seed(131)
nround <- 27


# モデルの構築
bst <- xgboost(param=param, data = x[trind,], label = y, nrounds=nround)
pred <- predict(bst,x[teind,]) # モデルを使って予測値を算出
pred <- matrix(pred,2,length(pred)/2)　　　#2群比較であれば2、3群比較であれば3にしてみる
pred <- t(pred)
colnames(pred)<-c("H_Control","H_Test") #Speciesの群分けControl H_Control

head(pred,2) #2群の場合

param <- list("objective" = "multi:softmax", # multi:softmax に変更！
              "eval_metric" = "mlogloss", 
              "num_class" = 2 #2群の場合
)


set.seed(131)
nround <- 27
bst <- xgboost(param=param, data = x[trind,], label = y, nrounds=nround)
pred <- predict(bst,x[teind,])


#二群のとき
x_1_f <- FCPC.test[,1]
for(i in 1:length(pred)){
  if(pred[i]==0) {pred[i]="H_Control"}　#Speciesの群分けControl H_Control
  else if(pred[i]==1) {pred[i]="H_Test"}
}

table(x_1_f,pred)

#三群のとき
x_1_f <- FCPC.test[,1]
for(i in 1:length(pred)){
  if(pred[i]==0) {pred[i]="Control"}
  else if(pred[i]==1) {pred[i]="H_Control"}
  else {pred[i]="H_Test"}
}

table(x_1_f,pred)




set.seed(131)
nround <- 27
bst <- xgboost(param=param, data = x[trind,], label = y, nrounds=nround)
pred <- predict(bst,x[teind,])

for(i in 1:length(pred)){
  if(pred[i]==0) {pred[i]="Control"}
  else if(pred[i]==1) {pred[i]="H_Control"}
  else {pred[i]="H_Test"}
}
table(x_1_f,pred)


sink('XGboost_pre_x_1_f.txt', append = TRUE)
print (table(x_1_f,pred))
sink()

write.csv(table(x_1_f,pred),"XGboost_pre_x_1_f.csv")


# 変数重要度を求める
imp<-xgb.importance(names(y_1),model=bst)
print(imp)
xgb.plot.importance(imp) 

pdf ("XGBoostgraph.pdf") 
xgb.plot.importance(imp) 
dev.off()

write.csv(print(imp),"XGBoostgraph_raw.csv")

# 変数重要度を求める
imp<-xgb.importance(names(y_1),model=bst)　#par(mar=c(100, 20, 30, 600))
print(imp)
xgb.plot.importance(imp) 

pdf ("XGBoostgraph.pdf") 
xgb.plot.importance(imp) 
dev.off()

write.csv(print(imp),"XGBoostgraph_raw.csv")

xgb.plot.tree(feature_names=names(FCPC.test[,-1]),model=bst, n_first_tree=2)

pdf ("XGBoostgraph_Tree.pdf") 
xgb.plot.tree(feature_names=names(FCPC.test[,-1]),model=bst, trees=2)
dev.off()





#ランダムフォレスト

#setwd("~/Downloads")
#install.packages("HOGE.tar.gz", repos = NULL, type = "source")

library(randomForest)
set.seed(131)
train.x<- FCPC.train[,2:102] #dim(FCPC.train) [1]  8 102
train.y<-as.factor(FCPC.train[,1])
model.rf<-tuneRF(train.x,train.y,doBest=T)
pred<-predict(model.rf,FCPC.test[,2:102]) #dim(FCPC.train) [1]  8 102
table(FCPC.test[,1],pred)
print(model.rf$importance /sum(model.rf$importance))

write.csv(print(model.rf$importance /sum(model.rf$importance)),"randomForest_raw.csv")

rf_pred <- table(FCPC.test[,1],pred)
write.csv(rf_pred,"randomForest_pred.csv")


#factor型に名前をする必要あり
FCPC.test_n <- cbind(y_1, train.x)
library(randomForest)
set.seed(22)
model = randomForest(y_1 ~ ., data = FCPC.test_n, importance = TRUE, proximity = TRUE)
print(model)
print(importance(model))
write.csv(print(importance(model)),"randomForest_pred_importance.csv")

print(varImpPlot(model))

write.csv(print(varImpPlot(model)),"randomForest_pred_importance_Gini.csv")

par(mar=c(100, 20, 30, 40)) #par(oma = c(3, 3, 3, 2))
rpp2 <- varImpPlot(model)
varFileName <- paste("randomforest_tree_var.png",sep="") #フォルダの位置確認
par(mar=c(100, 20, 30, 600)) #par(oma = c(3, 3, 3, 2))
png(file=varFileName, res=125, w=750, h=750)
rpp2 <- varImpPlot(model)
dev.off()









