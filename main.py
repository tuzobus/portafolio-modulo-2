import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

columns = [
    "Area",
    "Perimeter",
    "Major Axis Length",
    "Minor Axis Length",
    "Eccentricity",
    "Convex Area",
    "Extent",
    "Class",
]
df = pd.read_csv(
    "Rice_Cammeo_Osmancik.arff",
    skiprows=16,
    names=columns,
)

# EXTRACT y EDA
# Información general del dataset apenas importado
print("\nInformación general:")
df.info()
print(df.head())
print(df.tail())
print(df.shape)
print(df.columns)

# Valores faltantes
print("\nValores faltantes:")
print(df.isnull().sum())

# Filas duplicadas
print("\nFilas duplicadas:")
print(df.duplicated().sum())

# Clases existentes
print("\nClases existentes:")
print(df["Class"].unique())

# Cantidad de observaciones por clase
print("\nDistribución de clases:")
print(df["Class"].value_counts())

# Porcentaje de cada clase
print("\nDistribución porcentual:")
print(df["Class"].value_counts(normalize=True) * 100)

# Estadística descriptiva de las variables numéricas
print("\nEstadística descriptiva:")
print(df.describe())

# Comparación entre clases
print("\nPromedios por clase:")
print(df.groupby("Class").mean())

print("\nMedianas por clase:")
print(df.groupby("Class").median())


for i in columns[:-1]:
    cammeo = df[df["Class"] == "Cammeo"][i]
    osmancik = df[df["Class"] == "Osmancik"][i]

    plt.figure(figsize=(8, 5))

    plt.hist(
        cammeo,
        bins=30,
        alpha=0.5,
        label="Cammeo",
    )

    plt.hist(
        osmancik,
        bins=30,
        alpha=0.5,
        label="Osmancik",
    )

    plt.xlabel(i)
    plt.ylabel("Frequency")
    plt.title(f"Distribución de {i} por clase")
    plt.legend()

    plt.savefig("histograms/" + i + ".png")


numeric_df = df.drop(columns=["Class"])
correlation_matrix = numeric_df.corr()
print("\nMatriz de correlación:")
print(correlation_matrix.round(3))

plt.figure(figsize=(9, 7))

plt.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=45,
    ha="right",
)

plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)

for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        plt.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}", ha="center", va="center")

plt.title("Matriz de correlación")
plt.tight_layout()
plt.savefig("histograms/correlation_matrix.png")
plt.close()

pairs = [
    ("Area", "Convex Area"),
    ("Perimeter", "Major Axis Length"),
    ("Major Axis Length", "Minor Axis Length"),
]

for x_feature, y_feature in pairs:
    plt.figure(figsize=(8, 5))

    for class_name in ["Cammeo", "Osmancik"]:
        subset = df[df["Class"] == class_name]

        plt.scatter(
            subset[x_feature],
            subset[y_feature],
            alpha=0.4,
            label=class_name,
        )

    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.title(f"{x_feature} vs {y_feature}")
    plt.legend()
    plt.savefig("histograms/scatterplots-" + x_feature + "-vs-" + y_feature + ".png")

# TRANSFORM
# Codificación de clases
# 0: Osmancik, 1: Cammeo

df["Class"] = df["Class"].map({"Osmancik": 0, "Cammeo": 1})
print(df["Class"].value_counts())

# Como el dataset viene ordenado, se lleva a cabo un shuffle
np.random.seed(67)
indices = np.random.permutation(len(df))
df_shuffled = df.iloc[indices].reset_index(drop=True)
print("\nDataset después del shuffle:")
print(df_shuffled.head())

# Separación de X y Y / train y test
train_size = int(len(df_shuffled) * 0.8)

train_df = df_shuffled.iloc[:train_size]
test_df = df_shuffled.iloc[train_size:]

X_train = train_df.drop(columns=["Class"]).to_numpy()
X_test = test_df.drop(columns=["Class"]).to_numpy()

y_train = train_df["Class"].to_numpy()
y_test = test_df["Class"].to_numpy()

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# MODELO DE REGRESIÓN LOGÍSTICA
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict_probability(X, weights, bias):
    z = np.dot(X, weights) + bias
    return sigmoid(z)


def binary_cross_entropy(y, y_hat):
    epsilon = 1e-15
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)

    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


def mean_scaling(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    max_val = np.max(X_train, axis=0)

    X_train_scaled = (X_train - mean) / max_val
    X_test_scaled = (X_test - mean) / max_val

    return X_train_scaled, X_test_scaled


X_train, X_test = mean_scaling(X_train, X_test)


def train_logistic_regression(X, y, learning_rate, epochs):
    n_samples, n_features = X.shape

    weights = np.zeros(n_features)
    bias = 0.0

    losses = []

    for epoch in range(epochs):

        # 1. Hipótesis
        y_hat = predict_probability(X, weights, bias)

        # 2. Costo
        loss = binary_cross_entropy(y, y_hat)
        losses.append(loss)

        # 3. Gradientes
        error = y_hat - y

        dw = (1 / n_samples) * np.dot(X.T, error)
        db = (1 / n_samples) * np.sum(error)

        # 4. Actualización de parámetros
        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, " f"Loss: {loss:.6f}")

    return weights, bias, losses


def predict(X, weights, bias):
    probabilities = predict_probability(X, weights, bias)

    return (probabilities >= 0.5).astype(int)


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100


### Prueba

learning_rate = 0.03
epochs = 3000

weights, bias, losses = train_logistic_regression(
    X_train, y_train, learning_rate, epochs
)

print("\nPesos finales:")
print(weights)

print("\nBias:")
print(bias)

y_pred_train = predict(X_train, weights, bias)

y_pred_test = predict(X_test, weights, bias)

print("\nAccuracy train:", accuracy(y_train, y_pred_train))

print("Accuracy test:", accuracy(y_test, y_pred_test))

plt.figure(figsize=(8, 5))

plt.plot(losses)

plt.xlabel("Epoch")
plt.ylabel("Binary Cross Entropy")
plt.title("Evolución del costo durante el entrenamiento")

plt.savefig("training_loss.png")
plt.close()


# Predicciones
probabilities = predict_probability(X_test, weights, bias)

for i in range(10):
    predicted_class = y_pred_test[i]
    real_class = y_test[i]

    predicted_name = "Cammeo" if predicted_class == 1 else "Osmancik"

    real_name = "Cammeo" if real_class == 1 else "Osmancik"

    print(
        f"Ejemplo {i + 1}: "
        f"Probabilidad Cammeo = {probabilities[i]:.4f}, "
        f"Predicción = {predicted_name}, "
        f"Real = {real_name}"
    )
