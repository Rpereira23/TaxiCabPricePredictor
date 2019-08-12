

from math import sqrt
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor

from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

from sklearn.preprocessing import StandardScaler


class Regression(object):

    def __init__(self, X, y, model_type="linear_regression", parameters=None, test_size=0.3, normalize=False):

        self.X_train, self.X_test, self.y_train, \
        self.y_test = train_test_split(X, y, test_size=test_size, random_state=123)

        if normalize:
            scaler = StandardScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)

        if model_type == "linear_regression":
            self.model = LinearRegression()

        elif model_type == "stochastic_gradient_descent":
            self.model = SGDRegressor()

        elif model_type == "svr":
            self.model = SVR()

        elif model_type == "decision_tree":
            self.model = DecisionTreeRegressor()

        else:
            raise ValueError("Given Unknown Model Type: {}".format(model_type))

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def validate_prediction(self, y_pred, y):
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)

        return "mse = {mse} & mae = {mae} & rmse = {rmse}".format(mse=mse, mae=mae, rmse=sqrt(mse))



