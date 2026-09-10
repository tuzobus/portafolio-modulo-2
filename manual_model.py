import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import erfc, sqrt
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


MODEL_FEATURES = [
    "Area",
    "Perimeter",
    "Major Axis Length",
    "Minor Axis Length",
    "Eccentricity",
    "Extent",
]


# TRANSFORM
# Para la regresión logística se le asigna 0 a Osmancik y 1 a Cammeo
df["Class"] = df["Class"].map({"Osmancik": 0, "Cammeo": 1})


# Separación manual estratificada del dataset
# 60% entrenamiento, 20% validación y 20% prueba
def stratified_split_indices(y, seed=67):
    rng = np.random.RandomState(seed)
    train_indices = []
    val_indices = []
    test_indices = []

    for class_value in [0, 1]:
        class_indices = np.where(y == class_value)[0].copy()

        rng.shuffle(class_indices)
        n = len(class_indices)

        train_end = int(n * 0.60)
        val_end = train_end + int(n * 0.20)

        train_indices.extend(class_indices[:train_end])
        val_indices.extend(class_indices[train_end:val_end])
        test_indices.extend(class_indices[val_end:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    return (
        np.array(train_indices),
        np.array(val_indices),
        np.array(test_indices),
    )


y_all = df["Class"].to_numpy()

train_indices, val_indices, test_indices = stratified_split_indices(
    y_all,
    seed=67,
)

X_all = df[MODEL_FEATURES].to_numpy(dtype=float)

X_train = X_all[train_indices]
X_val = X_all[val_indices]
X_test = X_all[test_indices]

y_train = y_all[train_indices]
y_val = y_all[val_indices]
y_test = y_all[test_indices]

print("\nDimensiones de los conjuntos:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\nDistribución de clases:")

for name, y_subset in [
    ("Train", y_train),
    ("Validation", y_val),
    ("Test", y_test),
]:
    values, counts = np.unique(y_subset, return_counts=True)

    print(name, dict(zip(values, counts)))


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
def train_logistic_regression(
    X_train,
    y_train,
    X_val,
    y_val,
    learning_rate,
    epochs,
):
    n_samples, n_features = X_train.shape

    weights = np.zeros(n_features)
    bias = 0.0

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Predicción en train
        train_probabilities = predict_probability(
            X_train,
            weights,
            bias,
        )

        # Loss de train
        train_loss = binary_cross_entropy(
            y_train,
            train_probabilities,
        )

        train_losses.append(train_loss)

        # Loss de validation
        val_probabilities = predict_probability(
            X_val,
            weights,
            bias,
        )

        val_loss = binary_cross_entropy(
            y_val,
            val_probabilities,
        )

        val_losses.append(val_loss)

        # Gradientes
        error = train_probabilities - y_train

        dw = (1 / n_samples) * np.dot(X_train.T, error)
        db = (1 / n_samples) * np.sum(error)

        # Actualización de parámetros
        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db

        if epoch % 100 == 0:
            print(
                f"Epoch {epoch}, "
                f"Train loss: {train_loss:.6f}, "
                f"Validation loss: {val_loss:.6f}"
            )

    return (
        weights,
        bias,
        train_losses,
        val_losses,
    )


# Función para convertir las probabilidades generadas por la sigmoide en clases
# Umbral de clasificación utilizado: 0.5
def predict(X, weights, bias):
    probabilities = predict_probability(X, weights, bias)

    return (probabilities >= 0.5).astype(int)


# Función para calcular el porcentaje de observaciones clasificadas correctamente
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100


# Función para calcular precisión, recall y F1 para una clase específica
def precision_recall_f1_for_class(
    y_true,
    y_pred,
    class_value,
):
    tp = np.sum((y_true == class_value) & (y_pred == class_value))

    fp = np.sum((y_true != class_value) & (y_pred == class_value))

    fn = np.sum((y_true == class_value) & (y_pred != class_value))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return precision, recall, f1


def macro_f1(y_true, y_pred):
    _, _, f1_osmancik = precision_recall_f1_for_class(
        y_true,
        y_pred,
        0,
    )

    _, _, f1_cammeo = precision_recall_f1_for_class(
        y_true,
        y_pred,
        1,
    )

    return (f1_osmancik + f1_cammeo) / 2


def confusion_matrix_manual(y_true, y_pred):
    osmancik_correct = np.sum((y_true == 0) & (y_pred == 0))

    osmancik_as_cammeo = np.sum((y_true == 0) & (y_pred == 1))

    cammeo_as_osmancik = np.sum((y_true == 1) & (y_pred == 0))

    cammeo_correct = np.sum((y_true == 1) & (y_pred == 1))

    return np.array(
        [
            [
                osmancik_correct,
                osmancik_as_cammeo,
            ],
            [
                cammeo_as_osmancik,
                cammeo_correct,
            ],
        ]
    )


def evaluate_predictions(
    y_true,
    y_pred,
):
    precision_0, recall_0, f1_0 = precision_recall_f1_for_class(
        y_true,
        y_pred,
        0,
    )

    precision_1, recall_1, f1_1 = precision_recall_f1_for_class(
        y_true,
        y_pred,
        1,
    )

    return {
        "Accuracy": accuracy(
            y_true,
            y_pred,
        )
        / 100,
        "Macro-F1": macro_f1(
            y_true,
            y_pred,
        ),
        "Osmancik Precision": precision_0,
        "Osmancik Recall": recall_0,
        "Osmancik F1": f1_0,
        "Cammeo Precision": precision_1,
        "Cammeo Recall": recall_1,
        "Cammeo F1": f1_1,
    }


# ENTRENAMIENTO Y EVALUACIÓN
learning_rate = 0.03
epochs = 3000

weights, bias, train_losses, val_losses = train_logistic_regression(
    X_train,
    y_train,
    X_val,
    y_val,
    learning_rate,
    epochs,
)

print("\nPesos finales:")
print(weights)

print("\nBias:")
print(bias)


# ANÁLISIS ESTADÍSTICO DE LOS COEFICIENTES
# Se utiliza una aproximación de Wald basada en
# la matriz de información observada.

X_design = np.column_stack(
    [
        np.ones(X_train.shape[0]),
        X_train,
    ]
)

beta = np.concatenate(
    (
        [bias],
        weights,
    )
)

train_probabilities = predict_probability(
    X_train,
    weights,
    bias,
)

w_values = train_probabilities * (1 - train_probabilities)

information_matrix = X_design.T @ (X_design * w_values[:, None])

covariance_matrix = np.linalg.pinv(information_matrix)

standard_errors = np.sqrt(
    np.clip(
        np.diag(covariance_matrix),
        0,
        None,
    )
)

z_scores = np.divide(
    beta,
    standard_errors,
    out=np.full_like(
        beta,
        np.nan,
        dtype=float,
    ),
    where=standard_errors > 0,
)

p_values = np.array(
    [
        erfc(abs(z_value) / sqrt(2)) if np.isfinite(z_value) else np.nan
        for z_value in z_scores
    ]
)

odds_ratios = np.exp(beta)

coefficient_names = [
    "Intercept",
    *MODEL_FEATURES,
]

coefficient_table = pd.DataFrame(
    {
        "Variable": coefficient_names,
        "Coefficient": beta,
        "Std. Error": standard_errors,
        "Z": z_scores,
        "p-value": p_values,
        "Odds Ratio": odds_ratios,
    }
)

coefficient_table["Significant (p < 0.05)"] = coefficient_table["p-value"] < 0.05


print("\nCOEFICIENTES DEL MODELO")
print(coefficient_table.round(6).to_string(index=False))

# Predicciones sobre datos utilizados y no utilizados durante el entrenamiento
y_pred_train = predict(X_train, weights, bias)
y_pred_val = predict(X_val, weights, bias)
y_pred_test = predict(X_test, weights, bias)

train_metrics = evaluate_predictions(
    y_train,
    y_pred_train,
)

val_metrics = evaluate_predictions(
    y_val,
    y_pred_val,
)

test_metrics = evaluate_predictions(
    y_test,
    y_pred_test,
)


results = pd.DataFrame(
    [
        {
            "Set": "Train",
            **train_metrics,
        },
        {
            "Set": "Validation",
            **val_metrics,
        },
        {
            "Set": "Test",
            **test_metrics,
        },
    ]
)


print("\nRESULTADOS DEL MODELO MANUAL")
print(results.round(4).to_string(index=False))


print("\nMatriz de confusión - Train:")
print(
    confusion_matrix_manual(
        y_train,
        y_pred_train,
    )
)

print("\nMatriz de confusión - Validation:")
print(
    confusion_matrix_manual(
        y_val,
        y_pred_val,
    )
)

print("\nMatriz de confusión - Test:")
print(
    confusion_matrix_manual(
        y_test,
        y_pred_test,
    )
)


train_val_gap = train_metrics["Macro-F1"] - val_metrics["Macro-F1"]

print(
    "\nGeneralization gap " "(Train Macro-F1 - Validation Macro-F1):",
    round(train_val_gap, 4),
)


# VISUALIZACIÓN DEL ENTRENAMIENTO
# Se puede saber que la gradiente descendente se ajusta progresivamente
# a los parámetros por la disminución del costo
plt.figure(figsize=(8, 5))

plt.plot(
    train_losses,
    label="Train",
)

plt.plot(
    val_losses,
    label="Validation",
)

plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy")

plt.title("Training vs Validation Loss")

plt.legend()

plt.tight_layout()

plt.savefig(
    "training_validation_loss.png",
    dpi=300,
    bbox_inches="tight",
)

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
