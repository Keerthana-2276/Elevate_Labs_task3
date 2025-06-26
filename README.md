# 🏠 Housing Price Prediction using Linear Regression

This project uses a **Linear Regression model** to predict house prices based on features like area and location. The process includes data preprocessing, model training, evaluation, and visualization.

---

## 📁 Dataset

- Input File: `task 3.csv`
- Target column: `price`

---

## 🧪 Complete Code

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Import and preprocess the dataset
df = pd.read_csv("/content/task 3.csv")
df_encoded = pd.get_dummies(df, drop_first=True)

# 2. Split data into train-test sets
X = df_encoded.drop("price", axis=1)
y = df_encoded["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Fit a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)

# 5. Plot regression line for 'area' and interpret coefficients
plt.figure(figsize=(8, 6))
sns.regplot(x=X_test["area"], y=y_test, label='Actual', scatter_kws={'alpha':0.5})
sns.lineplot(x=X_test["area"], y=model.predict(X_test), color='red', label='Predicted')
plt.title("Regression Line: Area vs Price")
plt.xlabel("Area")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.show()

# Print model coefficients
coefficients = pd.Series(model.coef_, index=X.columns)
print("\nIntercept:", model.intercept_)
print("\nCoefficients:")
print(coefficients.sort_values(ascending=False))
```

---

## 📊 Evaluation Metrics

- **MAE** – Mean Absolute Error
- **MSE** – Mean Squared Error
- **R²** – Coefficient of determination

---

## 📈 Output

- Console output of evaluation metrics
- Graph showing actual vs predicted prices (based on area)
- Model intercept and feature coefficients

---

## 💡 Conclusion

This project demonstrates a simple and effective way to predict housing prices using Linear Regression. You can further improve the model with feature engineering, polynomial terms, or advanced regressors like Random Forest.
