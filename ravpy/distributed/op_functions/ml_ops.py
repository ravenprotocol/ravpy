def linear_regression(x,y, params=None):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression().fit(x, y) 
    return model

def knn_classifier(x,y,params=None):#k=None):
    k = params.get('k', None)
    if k is None:
        raise Exception("k param is missing")
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=k).fit(x, y) 
    return model

def knn_regressor(x,y,params=None):#k=None):
    k = params.get('k', None)
    if k is None:
        raise Exception("k param is missing")
    from sklearn.neighbors import KNeighborsRegressor
    model = KNeighborsRegressor(n_neighbors=k).fit(x, y) 
    return model

def logistic_regression(x, y, params=None):#random_state=0):
    random_state = params.get('random_state', 0)
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(random_state=random_state).fit(x, y) 
    return model

def naive_bayes(x, y, params=None):
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB().fit(x, y) 
    return model

def kmeans(x, params=None):#n_clusters=None):
    n_clusters = params.get('n_clusters', None)
    if n_clusters is None:
        raise Exception("n_clusters param is missing")
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=n_clusters).fit(x) 
    return model

def svm_svc(x, y, params=None):#kernel='linear'):
    kernel = params.get('kernel', 'linear')
    from sklearn.svm import SVC
    model = SVC(kernel=kernel).fit(x, y) 
    return model

def svm_svr(x, y, params=None):#kernel='linear'):
    kernel = params.get('kernel', 'linear')
    from sklearn.svm import SVR
    model = SVR(kernel=kernel).fit(x, y) 
    return model

def decision_tree_classifier(x, y, params=None):
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier().fit(x, y) 
    return model

def decision_tree_regressor(x, y, params=None):
    from sklearn.tree import DecisionTreeRegressor
    model = DecisionTreeRegressor().fit(x, y) 
    return model

def random_forest_classifier(x, y, params=None):#n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2):
    n_estimators = params.get('n_estimators', 100)
    criterion = params.get('criterion', 'gini')
    max_depth = params.get('max_depth', None)
    min_samples_split = params.get('min_samples_split', 2)

    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split).fit(x, y) 
    return model

def random_forest_regressor(x, y, params=None):#n_estimators=100, criterion='squared_error', max_depth=None, min_samples_split=2):
    n_estimators = params.get('n_estimators', 100)
    criterion = params.get('criterion', 'squared_error')
    max_depth = params.get('max_depth', None)
    min_samples_split = params.get('min_samples_split', 2)

    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split).fit(x, y) 
    return model