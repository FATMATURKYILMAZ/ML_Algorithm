#sklearn:ML Library
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

#1)Veri seti incelenemesi
cancer=load_breast_cancer()
df=pd.DataFrame(data=cancer.data,columns=cancer.feature_names)
df["target"]=cancer.target

#2)Makine Öğrenmesi Modelinin Seçilmesi-KNN Sınıflandırıcısı
#3)Modelin train edilmesi

X=cancer.data #features
y=cancer.target #target

#train test split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

#standartazsyon(ölçeklendirme)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#knn modeli oluştur ve train et
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)  #fit fonksiyonu verimizi(samples+target) kullanarak knn algoritmasını eğitir.

#3)Sonuçların değerlendirilmesi:test
y_pred=knn.predict(X_test)
accuray=accuracy_score(y_test,y_pred)
print("Doğruluk:",accuray)

conf_matrix=confusion_matrix(y_test,y_pred)
print("confusion matrix:")
print(conf_matrix)