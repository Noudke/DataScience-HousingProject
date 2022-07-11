from pandas import read_excel, DataFrame, get_dummies, concat
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, precision_score, recall_score, mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.svm import SVR, SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import openpyxl


def load_data(predicted_column, normalize=False, path='Housing.xlsx', columns=None):
    if columns is None:
        columns = ["area", "bathrooms", "stories"]

    dataset = read_excel(path, sheet_name="Housing (1)", nrows=500)

    # Converting the categorical variable into numerical
    varlist = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

    # Defining the map function
    def binary_map(x):
        return x.map({'yes': 1, "no": 0})

    # Applying the function to the housing list
    dataset[varlist] = dataset[varlist].apply(binary_map)

    # Creating dummy variable
    status = get_dummies(dataset['furnishingstatus'])

    # Dropping the first column from status dataset
    status = get_dummies(dataset['furnishingstatus'], drop_first=True)

    # Adding the status to the original housing dataframe
    dataset = concat([dataset, status], axis=1)

    # Dropping 'furnishingstatus' as we have created the dummies for it
    dataset.drop(['furnishingstatus'], axis=1, inplace=True)

    x = dataset.loc[:, columns]

    if normalize:
        x = preprocessing.normalize(x)

    # De variabelekolom waarop voorspellingen gedaan worden.
    y = dataset.loc[:, predicted_column]

    return x, y


def load_data_random_forest_regression(predicted_column, normalize=False, path='Housing.xlsx', columns=None):
    if columns is None:
        columns = ["area", "bathrooms", "stories", "basement", 'bedrooms']

    dataset = read_excel(path, sheet_name="Housing (1)", nrows=500)

    # Converting the categorical variable into numerical
    varlist = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

    # Defining the map function
    def binary_map(x):
        return x.map({'yes': 1, "no": 0})

    # Applying the function to the housing list
    dataset[varlist] = dataset[varlist].apply(binary_map)

    # Creating dummy variable
    status = get_dummies(dataset['furnishingstatus'])

    # Dropping the first column from status dataset
    status = get_dummies(dataset['furnishingstatus'], drop_first=True)

    # Adding the status to the original housing dataframe
    dataset = concat([dataset, status], axis=1)

    # Dropping 'furnishingstatus' as we have created the dummies for it
    dataset.drop(['furnishingstatus'], axis=1, inplace=True)

    x = dataset.loc[:, columns]

    if normalize:
        x = preprocessing.normalize(x)

    # De variabelekolom waarop voorspellingen gedaan worden.
    y = dataset.loc[:, predicted_column]

    return x, y


def load_data_all_variables(predicted_column, path='Housing.xlsx'):
    dataset = read_excel(path, sheet_name="Housing (1)", nrows=500)

    # Converting the categorical variable into numerical
    varlist = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

    # Defining the map function
    def binary_map(x):
        return x.map({'yes': 1, "no": 0})

    # Applying the function to the housing list
    dataset[varlist] = dataset[varlist].apply(binary_map)

    # Creating dummy variable
    status = get_dummies(dataset['furnishingstatus'])

    # Dropping the first column from status dataset
    status = get_dummies(dataset['furnishingstatus'], drop_first=True)

    # Adding the status to the original housing dataframe
    dataset = concat([dataset, status], axis=1)

    # Dropping 'furnishingstatus' as we have created the dummies for it
    dataset.drop(['furnishingstatus'], axis=1, inplace=True)

    # Normaliseren van de kolommen
    df_scaled = dataset.copy()

    scaler = MinMaxScaler()
    for x in dataset.columns:
        df_scaled[[x]] = scaler.fit_transform(dataset[[x]])

    # Splitting the data into training and testing
    x = df_scaled.drop([predicted_column], axis=1)
    y = df_scaled[predicted_column]

    return x, y


# Deel 1

def mlr(x, y):
    # De trainings- en validatiepartities. 20% van de data wordt gebruikt voor validatie.
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.20, random_state=1)

    # Linear Regression
    linear_regression = LinearRegression(normalize=True)
    linear_regression.fit(x_train, y_train)
    predictions_linear_regression = linear_regression.predict(x_validation)

    print("Multi linear regression: ")
    print(r2_score(y_validation, predictions_linear_regression))
    print("Mean absolute error: " + str(mean_absolute_error(y_validation, predictions_linear_regression)))
    #print("Voor een huis met de volgende gegevens verwachten we een prijs van: "
    #      + str(linear_regression.predict([[8000, 3, 3]])[0]))


def logistic(x, y):

    # De trainings- en validatie partities. 20% van de data wordt gebruikt voor validatie.
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.20, random_state=1)

    logistic_regression = LogisticRegression(random_state=1).fit(x_train, y_train)
    predictions_logistic_regression = logistic_regression.predict(x_validation)

    print("Logistic: ")
    print(accuracy_score(predictions_logistic_regression, np.array(y_validation)))

   # print("Voor een huis met de volgende gegevens verwachten we wel (1) of geen (0) airconditioning: "
   #       + str(logistic_regression.predict([[6000, 3, 3]])[0]))


# Deel 2

def tree_regression(x, y):

    # De trainings- en validatie partities. 20% van de data wordt gebruikt voor validatie.
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.20, random_state=1)

    regressor = DecisionTreeRegressor(random_state=0).fit(x_train, y_train)
    predictions_regression_tree = regressor.predict(x_validation)

    print("Tree regression: ")
    print("r2-score: " + str(r2_score(y_validation, predictions_regression_tree)))
    print("Mean absolute error: " + str(mean_absolute_error(y_validation, predictions_regression_tree)))
#    print("Voor een huis met de volgende gegevens verwachten we een prijs van: " + str(regressor.predict([[8000, 3, 3]])[0]))


def classification_tree(x, y):

    # De trainings- en validatiepartities. 15% van de data wordt gebruikt voor validatie.
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.15, random_state=1)

    clf = DecisionTreeClassifier(random_state=1).fit(x_train, y_train)
    predictions_clf = clf.predict(x_validation)

    print("Classification tree: ")
    print(accuracy_score(predictions_clf, np.array(y_validation)))

    #print("Voor een huis met de volgende gegevens verwachten we wel (1) of geen (0) airconditioning: " +
    #      str(clf.predict([[8000, 3, 3]])[0]))


def random_forest_regressor(x, y):
    # De trainings- en validatiepartities. 20% van de data wordt gebruikt voor validatie.
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.20, random_state=1)

    clf = RandomForestRegressor(n_estimators=100, random_state=0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_validation)

    print("Random Forest Regression: ")
    print("r2-score: " + str(r2_score(y_validation, y_pred)))
    print("Mean absolute error: " + str(mean_absolute_error(y_validation, y_pred)))
    #print("Voor een huis met de volgende gegevens verwachten we een prijs van: " +
    #      str(clf.predict([[8000, 3, 3,1,4]])[0]))


def random_forest_classification(x, y):

    # De trainings- en validatiepartities. 15% van de data wordt gebruikt voor validatie.
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.15, random_state=1)

    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_validation)

    print("Random Forest Classification: ")
    print(accuracy_score(y_validation, y_pred))
    #print("Voor een huis met de volgende gegevens verwachten we wel (1) of geen (0) airconditioning: " +
    #      str(clf.predict([[8000, 3, 3]])[0]))


# Deel 3

def neural_net_regression(x, y):
    # De trainings- en validatie partities. 20% van de data wordt gebruikt voor validatie.
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.25, random_state=1)

    regressor = MLPRegressor(activation='identity', solver='lbfgs', verbose=5)
    regressor.fit(x_train, y_train)
    predictions_regression_tree = regressor.predict(x_validation)

    print("Neural Net-Regression: ")
    print("r2-score: " + str(r2_score(y_validation, predictions_regression_tree)))
    print("Mean absolute error: " + str(mean_absolute_error(y_validation, predictions_regression_tree)))
    #print("Voor een huis met de volgende gegevens verwachten we een prijs van: " + str(
    #    regressor.predict([[8000, 3, 3]])[0]))


def neural_net_classification(x, y):
    # De trainings- en validatiepartities. 20% van de data wordt gebruikt voor validatie.
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.20, random_state=1)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_validation)

    print("Neural Net-Classification: ")
    print(accuracy_score(y_validation, y_pred))
    #print("Voor een huis met de volgende gegevens verwachten we wel (1) of geen (0) airconditioning: " +
    #      str(clf.predict([[8000, 3, 3]])[0]))


def SVM_regression(x, y):
    # De trainings- en validatiepartities. 25% van de data wordt gebruikt voor validatie.
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.25, random_state=1)

    regressor = SVR(kernel='linear', C=0.1)
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_validation)

    print("SVM-regression: ")
    print("r2-score: " + str(r2_score(y_validation, y_pred)))
    print("Mean absolute error: " + str(mean_absolute_error(y_validation, y_pred)))
    #print("Voor een huis met de volgende gegevens verwachten we een prijs van: " +
    #      str(regressor.predict([[8000, 3, 3]])[0]))


def SVM_classification(x, y):
    # De trainings- en validatiepartities. 25% van de data wordt gebruikt voor validatie.
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.25, random_state=1)

    clf = SVC(kernel='linear', random_state=1, C=0.1)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_validation)

    print("SVM-classification: ")
    print(accuracy_score(y_validation, y_pred))
    #print("Voor een huis met de volgende gegevens verwachten we wel (1) of geen (0) airconditioning: " +
    #      str(clf.predict([[8000, 3, 3]])[0]))
    

if __name__ == '__main__':
    # Het oude laadproces
    X, Y = load_data("price")
    mlr(X, Y)
    tree_regression(X, Y)
    neural_net_regression(X, Y)
    SVM_regression(X, Y)
    X, Y = load_data("airconditioning")
    logistic(X, Y)
    classification_tree(X, Y)
    random_forest_classification(X, Y)
    neural_net_classification(X, Y)
    SVM_classification(X, Y)
    X, Y = load_data_random_forest_regression("price")
    random_forest_regressor(X, Y)

    # Het nieuwe laadproces
    X, Y = load_data_all_variables("price")
    mlr(X, Y)
    tree_regression(X, Y)
    neural_net_regression(X, Y)
    SVM_regression(X, Y)
    random_forest_regressor(X, Y)
    X, Y = load_data_all_variables("airconditioning")
    logistic(X, Y)
    classification_tree(X, Y)
    random_forest_classification(X, Y)
    neural_net_classification(X, Y)
    SVM_classification(X, Y)
