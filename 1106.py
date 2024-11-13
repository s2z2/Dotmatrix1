import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

df=pd.read_excel("Web顧客別データ.xlsx")

num_att=["年齢","前回購入からの日数"]
cat_att=["性別"]
cat_onehot=pd.get_dummies(df[cat_att],dtype=int)
X=pd.concat([df[num_att],cat_onehot],axis=1)
Y=df["休眠顧客化"]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)

#決定木
model=DecisionTreeClassifier(max_depth=8,random_state=0)
# ランダムフォレスト
#model=RandomForestClassifier(max_depth=8,random_state=0)
# ニューラルネットワーク
#model=MLPClassifier(max_iter=500,hidden_layer_sizes=(50,),random_state=0)
# AdaBoost
#model=AdaBoostClassifier(random_state=0)

# 学習
model.fit(X_train,Y_train)
# テストデータの予測
Y_pred=model.predict(X_test)

# 混同行列の計算
tn,fp,fn,tp=confusion_matrix(Y_test,Y_pred,labels=["FALSE","TRUE"]).ravel()
# 評価指標の計算
# 正解率
accuracy=(tp+tn)/(tn+fp+fn+tp)
# 適合率
precision=tp/(tp+fp)
# 再現率
recall=tp/(tp+fn)
# F1スコア
f1score=2*precision*recall/(precision+recall)

# 混同行列の表示
print("TN",tn,"FP",fp,"FN",fn,"TP",tp)
# 各評価指標の表示
print("Accuracy",accuracy)
print("Precision",precision)
print("Recall",recall)
print("F1 score",f1score)

