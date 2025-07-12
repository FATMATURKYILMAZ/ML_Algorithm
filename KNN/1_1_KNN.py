#sklearn:ML Library
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

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

#4)Sonuçların değerlendirilmesi:test
y_pred=knn.predict(X_test)
accuray=accuracy_score(y_test,y_pred)
print("Doğruluk:",accuray)

conf_matrix=confusion_matrix(y_test,y_pred)
print("confusion matrix:")
print(conf_matrix)

#5)Hiperparametre ayarlaması:Amaç başarı skorunu arttırmak
#KNN:Hyperparameter:K K:1,2,3...N Accuracy:%A.%B...

accuracy_values=[]
k_values=[]
for k in range(1,21):
     knn=KNeighborsClassifier(n_neighbors=k)
     knn.fit(X_train,y_train)
     y_pred=knn.predict(X_test)
     accuracy=accuracy_score(y_test, y_pred)
     accuracy_values.append(accuracy)
     k_values.append(k)
"""    
plt.figure()
plt.plot(k_values, accuracy_values,marker="o",linestyle="-")  
plt.title("K değerine göre doğruluk")   
plt.xlabel("K değeri")
plt.ylabel("Doğruluk")
plt.xticks(k_values)
plt.grid(True)
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

# Veri oluştur
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()

# Gürültü ekle
y[::5] += 1 * (0.5 - np.random.rand(8))

# Tahmin için X değerleri
T = np.linspace(0, 5, 500)[:, np.newaxis]

# "uniform" ve "distance" ağırlık seçeneklerini dener
for weight in ["uniform", "distance"]:
    knn = KNeighborsRegressor(n_neighbors=5, weights=weight)  # HATA BURADA DÜZELTİLDİ
    y_pred = knn.fit(X, y).predict(T)
    
    # Grafik çizimi
    plt.figure()
    plt.scatter(X, y, color="green", label="data")
    plt.plot(T, y_pred, color="blue", label="prediction")
    plt.axis("tight")
    plt.legend()
    plt.title("KNN Regressor weights = {}".format(weight))

plt.tight_layout()
plt.show()














