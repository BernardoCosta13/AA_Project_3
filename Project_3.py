from sklearn.impute import KNNImputer
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.decomposition import PCA

# Load the dataset
X_train, X_ivs, y_train, col_names = pickle.load(
    open("drd2_data.pickle", "rb"))

# Assuming X_ivs is a Pandas DataFrame
X_ivs_df = pd.DataFrame(X_ivs)

# Ensure that the length of X_train is equal to the length of y_train
if len(X_train) != len(y_train):
    raise ValueError(
        "Inconsistent number of samples between X_train and y_train.")

y_train = pd.Series(y_train)

# Impute missing values using KNN
imputer = KNNImputer(n_neighbors=2)
X_train_imputed = imputer.fit_transform(X_train)

# Apply PCA
pca = PCA(n_components=100)  # Adjust n_components to your needs
X_train_pca = pca.fit_transform(X_train_imputed)

# Ensure that the PCA data has the same number of samples as the original data
if len(X_train_pca) != len(y_train):
    raise ValueError(
        "Inconsistent number of samples between PCA X_train and y_train.")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_train_pca, y_train, test_size=0.2, random_state=42)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "ElasticNet": ElasticNet(),
    "Polynomial Regression": LinearRegression(),
    "SVM": SVR(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "KNN": KNeighborsRegressor(),
    "Gaussian Process": GaussianProcessRegressor()
}

# Initialize variables to store the best model and the lowest RMSE
best_model = None
lowest_rmse = float('inf')

# Train and evaluate each model
for name, model in models.items():
    if name == "Polynomial Regression":
        poly = PolynomialFeatures(degree=2)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        model.fit(X_train_poly, y_train)
        y_pred = model.predict(X_test_poly)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"{name} - RMSE: {rmse}")

    # Update the best model and the lowest RMSE
    if rmse < lowest_rmse:
        best_model = model
        lowest_rmse = rmse


# Apply PCA to X_ivs_df
X_ivs_pca = pca.transform(X_ivs_df)

# Make predictions for X_ivs dataset using the best model
if best_model is not None:
    if name == "Polynomial Regression":
        X_ivs_pca_poly = poly.transform(X_ivs_pca)
        final_predictions = best_model.predict(X_ivs_pca_poly)
    else:
        final_predictions = best_model.predict(X_ivs_pca)

# Save predictions to a text file
group_number = 4
file_name = f"{group_number:02d}.txt"

with open(file_name, "w") as file:
    for prediction in final_predictions:
        file.write(f"{prediction}\n")

# Plot actual vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.show()
