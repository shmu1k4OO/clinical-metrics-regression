import pandas as pd
import numpy as np
import sklearn
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import os

def clinic_value(row):
    if row == "СТ":
        return 1
    elif row == "Т":
        return 2
    elif row == "Л":
        return 0

def show_spearmen(data):
    os.makedirs("images", exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(method='spearman'), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title("Корреляционная матрица Спирмена")
    plt.savefig("images/spearman_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

def show_pearson(data):
    os.makedirs("images", exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title("Корреляционная матрица Пирсона")
    plt.savefig("images/person_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()


database = pd.read_excel("data/DataBase.xlsx")
dataset = pd.read_excel("data/DataSigns.xlsx")

database["Result_Value"] = database["Клин_Оценка"].apply(clinic_value)

features = ['Age', 'Temperature.', 'SatO2', 'Neutral.1', 'Ctreating. 1', 'SRB', 'fibrinogen']

X = dataset[features].fillna(dataset[features].mean())

X = sm.add_constant(X)

y = database["Result_Value"]

corr_df = X.copy()
corr_df["Result_Value"] = y.values

show_pearson(corr_df)
show_spearmen(corr_df)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.30, random_state=100)

model = sm.OLS(y_train, X_train)

results = model.fit()

print(results.summary())

y_pred_train_continuous = results.predict(X_train)

y_pred_test_continuous = results.predict(X_test)

mae_train = sklearn.metrics.mean_absolute_error(y_train, y_pred_train_continuous)

r2_train = sklearn.metrics.r2_score(y_train, y_pred_train_continuous)

mae_test = sklearn.metrics.mean_absolute_error(y_test, y_pred_test_continuous)

r2_test = sklearn.metrics.r2_score(y_test, y_pred_test_continuous)

print(f"Для обучения: {mae_train}, {r2_train}", end='\n')
print(f"Для экзамена: {mae_test}, {r2_test}", end='\n')

#Дискретизация данных -> перехожу к классификации
y_pred_train_rounded = np.clip(np.round(y_pred_train_continuous), 0, 2)

y_pred_test_rounded = np.clip(np.round(y_pred_test_continuous), 0, 2)

acc_train = sklearn.metrics.accuracy_score(y_train, y_pred_train_rounded)
acc_test = sklearn.metrics.accuracy_score(y_test, y_pred_test_rounded)

print("--- Точность предсказаний ---")
print(f"На обучающей выборке: {acc_train * 100:.1f}%")
print(f"На тестовой выборке: {acc_test * 100:.1f}%")