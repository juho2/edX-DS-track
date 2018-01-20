import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer,MaxAbsScaler,MinMaxScaler,KernelCenterer,StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
path = r'F:\Juhon\python\EdX Azure\Python for DS\Module6\Datasets\parkinsons.data'
X = pd.read_csv(path)
y = X.status.copy()
X.drop(['name', 'status'], axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

#preps = [Normalizer(), MaxAbsScaler(), MinMaxScaler(), KernelCenterer(), StandardScaler()]
preps = [StandardScaler()]

#svc = SVC()
#svc.fit(X_train, y_train)
#print(svc.score(X_test, y_test))

best_global = 0
for pp in preps:
    X_train = pp.fit_transform(X_train)
    X_test = pp.transform(X_test)

    best_dim_score = 0    
    X_tr = X_train.copy()
    X_te = X_test.copy()
#    for n in np.arange(21,23):
#        pca = PCA(n_components=n)
#        X_train = pca.fit_transform(X_tr)
#        X_test = pca.transform(X_te)
    for n_neigh in np.arange(2,6):
        for n_comp in np.arange(4,7):
            iso = Isomap(n_neighbors=n_neigh, n_components=n_comp)
            X_train = iso.fit_transform(X_tr)
            X_test = iso.transform(X_te)
            best_score = 0
            for C in np.arange(0.05,2.05,0.05):
                for gamma in np.arange(0.001,0.101,0.001):
                    svc = SVC(C=C, gamma=gamma)
                    svc.fit(X_train, y_train)
                    score = svc.score(X_test, y_test)
                    if score > best_score:
#                        print(score, C, gamma)
                        best_score = score
            if best_score > best_dim_score:
#                print(pca,score)
                print(iso, score)
                best_dim_score = best_score
    if best_dim_score > best_global:
       print(pp, iso, best_dim_score)
       best_global = best_dim_score
print(best_global)