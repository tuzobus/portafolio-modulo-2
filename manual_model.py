import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("histograms")
OUTPUT_DIR.mkdir(exist_ok=True)

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

# Dataset cargado desde archivo arff
# (leído con read_csv pero salto las primeras líneas con metadatos)
df = pd.read_csv(
    "Rice_Cammeo_Osmancik.arff",
    skiprows=16,
    names=columns,
)

print("\nInfo general del dataset:")
df.info()

print("\nForma del dataset:")
print(df.shape)


# EDA Y REVISIÓN DE CALIDAD DE LOS DATOS
# Valores faltantes que requieran limpieza o imputación
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

print("\nDistribución porcentual de clases:")
print(df["Class"].value_counts(normalize=True) * 100)

# Estadística descriptiva de las variables numéricas
print("\nEstadística descriptiva:")
print(df.describe())

# Comparación entre clases
print("\nPromedios por clase:")
print(df.groupby("Class").mean())

print("\nMedianas por clase:")
print(df.groupby("Class").median())


# EDA: DISTRIBUCIÓN DE FEATURES POR CLASE
# Histogramas para comparar distribuciones de cada
# característica entre las 2 clases de arroz
for feature in columns[:-1]:
    cammeo = df[df["Class"] == "Cammeo"][feature]
    osmancik = df[df["Class"] == "Osmancik"][feature]

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

    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.title(f"Distribución de {feature} por clase")
    plt.legend()

    plt.savefig(f"histograms/{feature}.png")
    plt.close()


# EDA: CORRELACIÓN ENTRE VARIABLES
# Class se excluye porque contiene etiquetas categóricas
# y no es numérica
numeric_df = df.drop(columns=["Class"])

# Matriz de correlación de Pearson para detectar
# relaciones lineales y posibles variables redundantes
correlation_matrix = numeric_df.corr()
print("\nMatriz de correlación:")
print(correlation_matrix.round(3))

# Plot de la matriz de correlación
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


# EDA: RELACIONES ENTRE FEATURES SELECCIONADAS
# Relaciones relevantes identificadas mediante la matriz de correlación
pairs = [
    ("Area", "Convex Area"),
    ("Perimeter", "Major Axis Length"),
    ("Major Axis Length", "Eccentricity"),
    ("Extent", "Major Axis Length"),
    ("Eccentricity", "Minor Axis Length"),
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

    plt.savefig(f"histograms/scatterplot-{x_feature}-vs-{y_feature}.png")
    plt.close()


# TRANSFORM
# Para la regresión logística se le asigna 0 a Osmancik y 1 a Cammeo
df["Class"] = df["Class"].map({"Osmancik": 0, "Cammeo": 1})

# Separación manual estratificada del dataset
# 60% entrenamiento, 20% validación y 20% prueba
np.random.seed(67)

train_indices = []
val_indices = []
test_indices = []

for class_value in [0, 1]:
    class_indices = np.where(df["Class"].to_numpy() == class_value)[0]

    np.random.shuffle(class_indices)

    n = len(class_indices)

    train_end = int(n * 0.6)
    val_end = train_end + int(n * 0.2)

    train_indices.extend(class_indices[:train_end])
    val_indices.extend(class_indices[train_end:val_end])
    test_indices.extend(class_indices[val_end:])

# Shuffle dentro de cada conjunto
np.random.shuffle(train_indices)
np.random.shuffle(val_indices)
np.random.shuffle(test_indices)

train_df = df.iloc[train_indices].reset_index(drop=True)
val_df = df.iloc[val_indices].reset_index(drop=True)
test_df = df.iloc[test_indices].reset_index(drop=True)

# Separación de las variables predictoras X y de la variable objetivo y
X_train = train_df.drop(columns=["Class"]).to_numpy()
X_val = val_df.drop(columns=["Class"]).to_numpy()
X_test = test_df.drop(columns=["Class"]).to_numpy()

y_train = train_df["Class"].to_numpy()
y_val = val_df["Class"].to_numpy()
y_test = test_df["Class"].to_numpy()

print("\nDimensiones de los conjuntos:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ESCALAMIENTO
# Ya que las variables presentan escalas distintas,
# se aplica z-score scaling para estabilizar la gradiente descendente
def z_score_scaling(X_train, X_val, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    # Reemplaza ceros en std con 1 para evitar división por cero
    std[std == 0] = 1

    X_train_scaled = (X_train - mean) / std
    X_val_scaled = (X_val - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled, X_val_scaled, X_test_scaled


X_train, X_val, X_test = z_score_scaling(X_train, X_val, X_test)


# MODELO DE REGRESIÓN LOGÍSTICA
print("\n---MODELO---")


# Función sigmoide (convierte valores a una probabilidad entre 0 y 1)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Función para calcular la combinación lineal de las features y aplicar la sigmoide
def predict_probability(X, weights, bias):
    z = np.dot(X, weights) + bias
    return sigmoid(z)


# Función para calcular el costo mediante cross entropy
def binary_cross_entropy(y, y_hat):
    epsilon = 1e-15

    # Se evitan valores iguales a 0 o 1 para no calcular log(0)
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)

    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


# Función para entrenar los pesos y el bias mediante gradiente descendente por batch
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


# Función para convertir las probabilidades generadas por la sigmoide en clases
# Umbral de clasificación utilizado: 0.5
def predict(X, weights, bias):
    probabilities = predict_probability(X, weights, bias)

    return (probabilities >= 0.5).astype(int)


# Función para calcular el porcentaje de observaciones clasificadas correctamente
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100


# ENTRENAMIENTO Y EVALUACIÓN

learning_rate = 0.03
epochs = 3000

weights, bias, losses = train_logistic_regression(
    X_train, y_train, learning_rate, epochs
)

print("\nPesos finales:")
print(weights)

print("\nBias:")
print(bias)

# Predicciones sobre datos utilizados y no utilizados durante el entrenamiento
y_pred_train = predict(X_train, weights, bias)
y_pred_val = predict(X_val, weights, bias)
y_pred_test = predict(X_test, weights, bias)

print("\nAccuracy train:", accuracy(y_train, y_pred_train))
print("Accuracy validation:", accuracy(y_val, y_pred_val))
print("Accuracy test:", accuracy(y_test, y_pred_test))


# VISUALIZACIÓN DEL ENTRENAMIENTO
# Se puede saber que la gradiente descendente se ajusta progresivamente
# a los parámetros por la disminución del costo
plt.figure(figsize=(8, 5))

plt.plot(losses)

plt.xlabel("Epoch")
plt.ylabel("Binary Cross Entropy")
plt.title("Evolución del costo durante el entrenamiento")

plt.savefig("training_loss.png")
plt.close()


# PREDICCIONES DE EJEMPLO
# Para el ejemplo, se muestran diez predicciones individuales del conjunto de prueba
# y se comparan con sus valores reales.
# Además, se muestran las probabilidades de que cada observación pertenezca a la clase Cammeo.
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
