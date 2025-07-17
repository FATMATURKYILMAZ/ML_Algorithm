from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,confusion_matrix

import matplotlib.pyplot as plt
#veri seti inceleme ve analiz
iris=load_iris()

X=iris.data #features
y=iris.target #target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42) #veri sayısı arttığında test_size küçültülür.

#decision tree(DT) MODELİ OLUŞTUR VE TRRAİN ET.
tree_clf=DecisionTreeClassifier(criterion="entropy",max_depth=5,random_state=42)#criterion="entropy"
#Scikit-learn'den gelen DecisionTreeClassifier sınıfından oluşturulmuş ve eğitilmiş bir model nesnesidir.
tree_clf.fit(X_train,y_train)

#decision tree evalution test
y_pred=tree_clf.predict(X_test)#Test verileri için tahmin et, Sonuçları y_pred içinde sakla
accurcy=accuracy_score(y_test, y_pred)
print("İris veri seti ile eğitilen bir modelin doğruluğu:",accurcy)

conf_matrix=confusion_matrix(y_test, y_pred)
print("conf-matrix:")
print(conf_matrix)

plt.figure(figsize=(15,10))
plot_tree(tree_clf, filled=True, feature_names=iris.feature_names,class_names=list(iris.target_names))
plt.show()
feature_importance=tree_clf.feature_importances_
feature_names=iris.feature_names
feature_importance_sorted=sorted(zip(feature_importance,feature_names),reverse=True)
for importance,feature_name in feature_importance_sorted:
    print(f"{feature_name}: {importance}")
    
#Tree'ler random forest ve decision tree tabanlı algorimalar feature selection açısından sık kullanılan algoritmalardır.  

#%%
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.inspection import DecisionBoundaryDisplay
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
#veri seti inceleme ve analiz

# Veri setini yükle
iris=load_iris()

n_classes=len(iris.target_names)
plot_colors="ryb"


# Özellik çiftleri üzerinden dön
# Özellik çiftleri üzerinden dön
for pairidix, pair in enumerate([[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]):
    X = iris.data[:, pair]
    y = iris.target

    # Karar ağacı modelini eğit
    clf = DecisionTreeClassifier().fit(X, y)

    # Alt grafik oluştur
    ax = plt.subplot(2, 3, pairidix + 1)

    # Grafik boşluklarını ayarla
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    # Karar sınırlarını çiz
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        cmap=plt.cm.RdYlBu,
        response_method="predict",
        ax=ax,
        xlabel=iris.feature_names[pair[0]],
        ylabel=iris.feature_names[pair[1]]
    )

    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                    cmap=plt.cm.RdYlBu, edgecolors="black")

plt.legend()
plt.show()
#%%
#scikit-learn içindeki diyabet veri setini yükler.
from sklearn.datasets import load_diabetes
#veriyi eğitim ve test olarak böler.
from sklearn.model_selection import train_test_split
#karar ağacı tabanlı regresyon algoritması.
from sklearn.tree import DecisionTreeRegressor
#tahmin hatasını ölçmek için MSE (Mean Squared Error) hesaplar.
from sklearn.metrics import mean_squared_error
#bilimsel hesaplamalar için kullanılır (burada karekök almak için).
import numpy as np


#Veri Setinin Yüklenmesi
diabetes=load_diabetes()

X=diabetes.data #features Bağımsız değişkenler (10 özellik)
y=diabetes.target #target Bağımlı değişken (hedef)

#Verinin Eğitim ve Test Olarak Bölünmesi
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42) #veri sayısı arttığında test_size küçültülür.

# Karar Ağacı Regresyon Modeli Oluşturulması ve Eğitilmesi
tree_reg=DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train,y_train)

#Tahmin Yapılması
y_pred=tree_reg.predict(X_test)

#Model Performansının Değerlendirilmesi
mse=mean_squared_error(y_test, y_pred)
print("mse:",mse)

rmse=np.sqrt(mse)
print("rmse:",rmse)