import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.ensemble import RandomForestClassifier

from manual_model import plot_confusion_matrix

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


MODEL_FEATURES = [
    "Area",
    "Perimeter",
    "Major Axis Length",
    "Minor Axis Length",
    "Eccentricity",
    "Extent",
]


# TRANSFORM
# Se hace encoding de la variable objetivo para random forest,
# se le asigna 0 a Osmancik y 1 a Cammeo
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

X = df[MODEL_FEATURES]
y = df["Class"]

X_train = X.iloc[train_indices]
X_val = X.iloc[val_indices]
X_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]
y_val = y.iloc[val_indices]
y_test = y.iloc[test_indices]


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
    print(name, y_subset.value_counts().sort_index().to_dict())


# MODELO DE RANDOM FOREST
# Este primer modelo no tiene hiperparámetros ajustados
base_model = RandomForestClassifier(random_state=67)

base_model.fit(X_train, y_train)

base_train_pred = base_model.predict(X_train)
base_val_pred = base_model.predict(X_val)

base_train_accuracy = accuracy_score(y_train, base_train_pred)

base_val_accuracy = accuracy_score(y_val, base_val_pred)

base_train_f1 = f1_score(
    y_train,
    base_train_pred,
    average="macro",
)

base_val_f1 = f1_score(
    y_val,
    base_val_pred,
    average="macro",
)

base_f1_gap = base_train_f1 - base_val_f1

base_val_confusion = confusion_matrix(y_val, base_val_pred)


print("\n-----")
print("RANDOM FOREST BASE")

print("Train Macro-F1:", round(base_train_f1, 4))

print("Validation Macro-F1:", round(base_val_f1, 4))

print("Generalization gap (Macro-F1):", round(base_f1_gap, 4))

print("\nTrain accuracy:", round(base_train_accuracy, 4))

print("Validation accuracy:", round(base_val_accuracy, 4))


print("\nValidation classification report:")

print(
    classification_report(
        y_val,
        base_val_pred,
        target_names=[
            "Osmancik",
            "Cammeo",
        ],
        digits=4,
    )
)


print("\nValidation confusion matrix:")

print(base_val_confusion)


# RANDOM FOREST AJUSTADO
#
# Esta configuración se modifica manualmente entre ejecuciones
# de acuerdo con los resultados obtenidos en train y validation.

adjusted_model = RandomForestClassifier(
    max_depth=2,
    random_state=67,
)

adjusted_model.fit(X_train, y_train)


adjusted_train_pred = adjusted_model.predict(X_train)

adjusted_val_pred = adjusted_model.predict(X_val)


adjusted_train_accuracy = accuracy_score(
    y_train,
    adjusted_train_pred,
)

adjusted_val_accuracy = accuracy_score(
    y_val,
    adjusted_val_pred,
)


adjusted_train_f1 = f1_score(
    y_train,
    adjusted_train_pred,
    average="macro",
)

adjusted_val_f1 = f1_score(
    y_val,
    adjusted_val_pred,
    average="macro",
)

adjusted_f1_gap = adjusted_train_f1 - adjusted_val_f1

adjusted_val_confusion = confusion_matrix(
    y_val,
    adjusted_val_pred,
)


print("\n-----")
print("RANDOM FOREST AJUSTADO")

print("Train Macro-F1:", round(adjusted_train_f1, 4))

print("Validation Macro-F1:", round(adjusted_val_f1, 4))

print("Generalization gap (Macro-F1):", round(adjusted_f1_gap, 4))

print("\nTrain accuracy:", round(adjusted_train_accuracy, 4))

print("Validation accuracy:", round(adjusted_val_accuracy, 4))


print("\nValidation classification report:")

print(
    classification_report(
        y_val,
        adjusted_val_pred,
        target_names=[
            "Osmancik",
            "Cammeo",
        ],
        digits=4,
    )
)


print("\nValidation confusion matrix:")

print(adjusted_val_confusion)


comparison_table = pd.DataFrame(
    [
        {
            "Model": "Random Forest Base",
            "Train Macro-F1": base_train_f1,
            "Validation Macro-F1": base_val_f1,
            "F1 Gap": base_f1_gap,
            "Train Accuracy": base_train_accuracy,
            "Validation Accuracy": base_val_accuracy,
        },
        {
            "Model": "Random Forest Adjusted",
            "Train Macro-F1": adjusted_train_f1,
            "Validation Macro-F1": adjusted_val_f1,
            "F1 Gap": adjusted_f1_gap,
            "Train Accuracy": adjusted_train_accuracy,
            "Validation Accuracy": adjusted_val_accuracy,
        },
    ]
)


print("\n-----")
print("Comparación")

print(comparison_table.round(4).to_string(index=False))


# GRÁFICA COMPARATIVA DE MACRO-F1
sets = [
    "Train",
    "Validation",
]

base_scores = [
    base_train_f1,
    base_val_f1,
]

adjusted_scores = [
    adjusted_train_f1,
    adjusted_val_f1,
]

x = np.arange(len(sets))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))

base_bars = ax.bar(
    x - width / 2,
    base_scores,
    width,
    label="Modelo base",
)

adjusted_bars = ax.bar(
    x + width / 2,
    adjusted_scores,
    width,
    label="Modelo ajustado",
)

ax.set_ylabel("Macro-F1")

ax.set_title("Desempeño antes y después del ajuste")

ax.set_xticks(x)
ax.set_xticklabels(sets)

ax.set_ylim(0, 1.08)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25,
)

ax.bar_label(
    base_bars,
    labels=[f"{score:.4f}" for score in base_scores],
    padding=3,
)

ax.bar_label(
    adjusted_bars,
    labels=[f"{score:.4f}" for score in adjusted_scores],
    padding=3,
)

fig.tight_layout()

plt.savefig(
    "f1_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# MATRICES DE CONFUSIÓN COMPARATIVAS
fig, axes = plt.subplots(
    1,
    2,
    figsize=(10, 4),
)

base_display = ConfusionMatrixDisplay(
    confusion_matrix=base_val_confusion,
    display_labels=[
        "Osmancik",
        "Cammeo",
    ],
)

base_display.plot(
    ax=axes[0],
    cmap="Blues",
    colorbar=False,
)

axes[0].set_title("Modelo base - Validation")
axes[0].set_xlabel("Clase predicha")
axes[0].set_ylabel("Clase real")


adjusted_display = ConfusionMatrixDisplay(
    confusion_matrix=adjusted_val_confusion,
    display_labels=[
        "Osmancik",
        "Cammeo",
    ],
)

adjusted_display.plot(
    ax=axes[1],
    cmap="Blues",
    colorbar=False,
)

axes[1].set_title("Modelo ajustado - Validation")
axes[1].set_xlabel("Clase predicha")
axes[1].set_ylabel("Clase real")

fig.tight_layout()

plt.savefig(
    "validation_confusion_matrices.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

test_pred = adjusted_model.predict(X_test)

test_macro_f1 = f1_score(
    y_test,
    test_pred,
    average="macro",
)

test_accuracy = accuracy_score(
    y_test,
    test_pred,
)

print("\n-----")
print("FINAL TEST")

print("Test Macro-F1:", round(test_macro_f1, 4))

print("Test accuracy:", round(test_accuracy, 4))

print("\nTest classification report:")
print(
    classification_report(
        y_test,
        test_pred,
        target_names=[
            "Osmancik",
            "Cammeo",
        ],
        digits=4,
    )
)

final_test_confusion = confusion_matrix(
    y_test,
    test_pred,
)

plot_confusion_matrix(
    final_test_confusion,
    "Random Forest final - Test",
    "random_forest_test_confusion.png",
)

print("\nTest confusion matrix:")
print(final_test_confusion)

# MATRIZ DE CONFUSIÓN DEL TEST FINAL
test_display = ConfusionMatrixDisplay(
    confusion_matrix=final_test_confusion,
    display_labels=[
        "Osmancik",
        "Cammeo",
    ],
)

fig, ax = plt.subplots(figsize=(5, 4))

test_display.plot(
    ax=ax,
    cmap="Blues",
    colorbar=False,
)

ax.set_title("Random Forest final - Test")

ax.set_xlabel("Clase predicha")

ax.set_ylabel("Clase real")

fig.tight_layout()

plt.savefig(
    "random_forest_test_confusion.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()
