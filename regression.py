from math import sqrt
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor

from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

class Regression(object):
    
    def __init__(self, X, y, model_type="linear_regression", parameters=None, test_size=0.3, normalize=False):
        self.model_type = model_type

        self.X_train, self.X_test, self.y_train, \
        self.y_test = train_test_split(X, y, test_size=test_size, random_state=123)

        if normalize:
            scaler = StandardScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)

        self.linear_parameters= {"fit_intercept" : True, "normalize" : False, "copy_X"  : True, "n_jobs" : 1}
        self.sgd_parameters = {"loss" : "squared_loss", 
                            "penalty" : "l2", 
                            "alpha" :  0.0001, 
                            "l1_ratio" : 0.15, 
                            "fit_intercept" : True, 
                            "random_state" : None, 
                            "learning_rate" : "invscaling"}
                            
        self.svr_parameters = {
            "kernel" : "rbf",
            "degree" : 3,
            "gamma" : "auto",
            "coef0" : 0.0,
            "tol" : 1e-3, 
            "C" : 1.0, 
            "epsilon" : 0.1,
            "shrinking" : True,
            "max_iter" : -1,
        }
        self.decision_tree_parameters = {
            "criterion" : "mse",
            "splitter" : "best",
            "max_depth" : None,
            "min_samples_split" : 2,
            "min_samples_leaf" : 1, 
            "min_weight_fraction_leaf" : 0, 
            "random_state" : None,
            "max_features" : None,
            }

        self.lasso_parameters = {
            "normalize" : False, 
            "selection" : "cyclic", 
            "alpha" :  1.0, 
            "copy_X"  : True,
            "max_iter" : -1, 
            "fit_intercept" : True, 
            "warm_start" : False, 
            "random_state" : None,
            "positive" : False
        }

        self.ridge_parameters = {
            "normalize" : False, 
            "solver" : "auto", 
            "alpha" :  1.0, 
            "copy_X"  : True,
            "max_iter" : -1, 
            "fit_intercept" : True, 
            "random_state" : None,
        }

        self.model, self.parameters = self.__create_model(model_type, parameters) 


    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def validate_prediction(self, y, y_pred):
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)

        return "mse = {mse} & mae = {mae} & rmse = {rmse}".format(mse=mse, mae=mae, rmse=sqrt(mse))

    def get_rmse(self, y, y_pred):
        mse = mean_squared_error(y, y_pred)

        return sqrt(mse)


    def perform_cross_validation(self, X_train, y_train, cv=5):
        return cross_val_score(estimator=self.model, X=X_train, y=y_train, cv=cv)

    def perform_grid_search_cross_validation(self, X_train, y_train, 
                                            grid_param, scoring='accuracy', cv=5, n_jobs=1):
        gd_sr = GridSearchCV(estimator=self.model, param_grid=grid_param, 
                             scoring=scoring, cv=cv, n_jobs=n_jobs)

        gd_sr.fit(X_train, y_train)
        return gd_sr.best_params_


    def update_with_parameters(self, new_parameters):
        parameters = self.__update_parameters(new_parameters, self.parameters)
        self.model, self.parameters = self.__create_model(self.model_type, parameters)
        self.fit(self.X_train, self.y_train)

    def __create_model(self, model_type, parameters):
        if model_type == "linear_regression":
            updated_parameters = self.__update_parameters(parameters, self.linear_parameters)
            model = LinearRegression(
                fit_intercept=updated_parameters["fit_intercept"], 
                normalize=updated_parameters["normalize"],
                copy_X=updated_parameters["copy_X"],
                n_jobs=updated_parameters["n_jobs"] )

        elif model_type == "stochastic_gradient_descent":
            updated_parameters = self.__update_parameters(parameters, self.sgd_parameters)
            model = SGDRegressor(
                loss=updated_parameters["loss"], 
                penalty=updated_parameters["penalty"],
                alpha=updated_parameters["alpha"],
                l1_ratio=updated_parameters["l1_ratio"],
                fit_intercept=updated_parameters["fit_intercept"], 
                random_state=updated_parameters["random_state"],
                learning_rate=updated_parameters["learning_rate"])

        elif model_type == "svr":
            updated_parameters = self.__update_parameters(parameters, self.svr_parameters)
            model = SVR(
                kernel=updated_parameters["kernel"],
                degree=updated_parameters["degree"],
                gamma=updated_parameters["gamma"],
                coef0=updated_parameters["coef0"],
                tol=updated_parameters["tol"],
                C=updated_parameters["C"],
                epsilon=updated_parameters["epsilon"],
                shrinking=updated_parameters["shrinking"],
                max_iter=updated_parameters["max_iter"],
            )

        elif model_type == "decision_tree":
            updated_parameters = self.__update_parameters(parameters, self.decision_tree_parameters)
            model = DecisionTreeRegressor(
                criterion=updated_parameters['criterion'],
                splitter=updated_parameters['splitter'],
                max_depth=updated_parameters['max_depth'],
                min_samples_split=updated_parameters['min_samples_split'],
                min_samples_leaf=updated_parameters['min_samples_leaf'],
                min_weight_fraction_leaf=updated_parameters['min_weight_fraction_leaf'],
                random_state=updated_parameters['random_state'],
                max_features=updated_parameters['max_features'])

        elif model_type == 'lasso':
            updated_parameters = self.__update_parameters(parameters, self.lasso_parameters)
            model = Lasso(
                normalize=updated_parameters['normalize'], 
                selection=updated_parameters['selection'],
                alpha = updated_parameters['alpha'],
                copy_X = updated_parameters['copy_X'],
                max_iter = updated_parameters['max_iter'],
                fit_intercept = updated_parameters['fit_intercept'],
                warm_start = updated_parameters['warm_start'],
                random_state = updated_parameters['random_state'],
                positive = updated_parameters['positive'])

        elif model_type == 'ridge':
            updated_parameters = self.__update_parameters(parameters, self.ridge_parameters)
            model = Ridge(
                normalize=updated_parameters['normalize'], 
                solver=updated_parameters['solver'],
                alpha = updated_parameters['alpha'],
                copy_X = updated_parameters['copy_X'],
                max_iter = updated_parameters['max_iter'],
                fit_intercept = updated_parameters['fit_intercept'],
                random_state = updated_parameters['random_state']
            )
        else:
            raise ValueError("Given Unknown Model Type: {}".format(model_type))

        return (model, updated_parameters)

    def __update_parameters(self, new_parameters, old_parameters):
        if new_parameters == None:
            return old_parameters

        for key in new_parameters.keys():
            if key in old_parameters.keys():
                old_parameters[key] = new_parameters[key]
            else:
                raise ValueError("Unknown Parameter: {}".format(key))

            return old_parameters


