# Import your libraries:
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.feature_selection import RFE
import statsmodels.api as sm


# Challenge 1 - Explore the Scikit-Learn Datasets

diabetes = load_diabetes()

print(diabetes.keys())

print(diabetes["DESCR"])

print(diabetes["data"].shape)


# Challenge 2 - Perform Supervised Learning on the Dataset

diabetes_model = LinearRegression()

diabetes_model.fit(diabetes["data"], diabetes["target"])
print("Intercept:", diabetes_model.intercept_)
print("Coefficients:", diabetes_model.coef_)


# Bonus Challenge 1 - Conduct a Hypothesis Test on the Model

X_diabetes = sm.add_constant(diabetes["data"])
model_sm = sm.OLS(diabetes["target"], X_diabetes).fit()
print(model_sm.summary())


# Challenge 3 - Perform Supervised Learning on a Pandas Dataframe

auto = pd.read_csv("../auto-mpg.csv")

print(auto.head())

numeric_columns = ["mpg", "cylinders", "displacement", "horse_power", "weight", "acceleration", "model_year"]
for column in numeric_columns:
    auto[column] = pd.to_numeric(auto[column], errors="coerce")

print(auto.dtypes)

print("Newest model year:", auto["model_year"].max())
print("Oldest model year:", auto["model_year"].min())

print(auto.isnull().sum())
auto = auto.dropna()

print(auto["cylinders"].value_counts())
print("Number of possible values:", auto["cylinders"].nunique())

X = auto.drop(columns=["car_name", "mpg"], errors="ignore")
y = auto["mpg"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

auto_model = LinearRegression()
auto_model.fit(X_train, y_train)


# Challenge 4 - Evaluate the Model

y_pred = auto_model.predict(X_train)
print("Train R^2:", r2_score(y_train, y_pred))

y_test_pred = auto_model.predict(X_test)
print("Test R^2:", r2_score(y_test, y_test_pred))


# Challenge 5 - Improve the Model Fit

X_train09, X_test09, y_train09, y_test09 = train_test_split(
    X, y, test_size=0.1, random_state=1
)

auto_model09 = LinearRegression().fit(X_train09, y_train09)

y_pred09 = auto_model09.predict(X_train09)
print("Train R^2 90% split:", r2_score(y_train09, y_pred09))

y_test_pred09 = auto_model09.predict(X_test09)
print("Test R^2 90% split:", r2_score(y_test09, y_test_pred09))

old_test_r2 = auto_model.score(X_test, y_test)
new_test_r2 = auto_model09.score(X_test09, y_test09)

print("Old test R^2:", old_test_r2)
print("New test R^2:", new_test_r2)

if new_test_r2 > old_test_r2:
    print("Yes, there is an improvement.")
else:
    print("No, there is no improvement.")


# Bonus Challenge 2 - Backward Elimination

rfe = RFE(estimator=auto_model, n_features_to_select=3)
rfe.fit(X_train, y_train)

print(rfe.ranking_)
print(X_train.columns)
print(pd.Series(rfe.ranking_, index=X_train.columns))
print(pd.Series(rfe.estimator_.coef_, index=X_train.columns[rfe.support_]))

selected_features = X_train.columns[rfe.support_]
print("Selected features:", selected_features)

X_reduced = X[selected_features]

X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(
    X_reduced, y, test_size=0.2, random_state=1
)

auto_model_reduced = LinearRegression().fit(X_train_reduced, y_train_reduced)

print("Reduced train R^2:", auto_model_reduced.score(X_train_reduced, y_train_reduced))
print("Reduced test R^2:", auto_model_reduced.score(X_test_reduced, y_test_reduced))

if auto_model_reduced.score(X_test_reduced, y_test_reduced) > auto_model.score(X_test, y_test):
    print("Yes, there is an improvement.")
else:
    print("No, there is no improvement.")
