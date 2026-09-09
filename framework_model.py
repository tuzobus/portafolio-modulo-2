import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.ensemble import RandomForestClassifier

RUN_GRID_SEARCH = False

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


# TRANSFORM
# Se hace encoding de la variable objetivo para random forest,
# se le asigna 0 a Osmancik y 1 a Cammeo
df["Class"] = df["Class"].map({"Osmancik": 0, "Cammeo": 1})


# Separación del dataset en columnas x y y.
X = df.drop(columns=["Class"])
y = df["Class"]

# Separación del dataset en 60% para entrenamiento y 40% para validation y test
# Como el dataset viene ordenado, se tiene que hacer shuffle, pero ahora se
# puede aprovechar train_test_split de sklearn en vez de hacerlo manualmente
# con la librería random y numpy. Esto también permite usar stratify, que
# asegura que la proporción de clases se mantenga en los conjuntos de train,
# validation y test (pues el dataset original de 2 clases está desbalanceado,
# 57% de Osmanik y un 43% de Cammeo)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=67, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=67, stratify=y_train
)


# MODELO DE RANDOM FOREST
# Este primer modelo no tiene hiperparámetros ajustados
model = RandomForestClassifier(random_state=67)

model.fit(X_train, y_train)

train_pred = model.predict(X_train)
val_pred = model.predict(X_val)

train_accuracy = accuracy_score(y_train, train_pred)
val_accuracy = accuracy_score(y_val, val_pred)

print("Train accuracy:", train_accuracy)
print("Validation accuracy:", val_accuracy)
print("Generalization gap:", train_accuracy - val_accuracy)

print("\nClassification report:")
print(classification_report(y_val, val_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_val, val_pred))

base_train_accuracy = train_accuracy
base_val_accuracy = val_accuracy
base_val_confusion = confusion_matrix(y_val, val_pred)

if RUN_GRID_SEARCH:
    param_grid = {
        "n_estimators": [100, 300, 500],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.5],
        "class_weight": [None, "balanced"],
    }

    search_model = RandomForestClassifier(random_state=67)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=67,
    )

    grid_search = GridSearchCV(
        search_model,
        param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0,
    )

    grid_search.fit(X_train, y_train)

    model = grid_search.best_estimator_

    print("Best parameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)

else:
    model = RandomForestClassifier(
        class_weight=None,
        max_depth=5,
        max_features="sqrt",
        min_samples_leaf=2,
        min_samples_split=10,
        n_estimators=300,
        random_state=67,
    )

    model.fit(X_train, y_train)


train_pred = model.predict(X_train)
val_pred = model.predict(X_val)

train_accuracy = accuracy_score(y_train, train_pred)
val_accuracy = accuracy_score(y_val, val_pred)

print("\nTUNED MODEL")
print("Train accuracy:", train_accuracy)
print("Validation accuracy:", val_accuracy)
print("Generalization gap:", train_accuracy - val_accuracy)

print("\nValidation classification report:")
print(classification_report(y_val, val_pred))

print("\nValidation confusion matrix:")
print(confusion_matrix(y_val, val_pred))


tuned_train_accuracy = train_accuracy
tuned_val_accuracy = val_accuracy
tuned_val_confusion = confusion_matrix(y_val, val_pred)


# Actualización del modelo con mejores hiperparámetros encontrados
test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_pred)

print("\nFINAL TEST")
print("Test accuracy:", test_accuracy)

print("\nClassification report:")
print(classification_report(y_test, test_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, test_pred))


# 10 PREDICCIONES DE EJEMPLO
test_probabilities = model.predict_proba(X_test)

print("\n10 PREDICCIONES DE EJEMPLO")

for i in range(10):
    predicted_class = test_pred[i]
    real_class = y_test.iloc[i]

    predicted_name = "Cammeo" if predicted_class == 1 else "Osmancik"

    real_name = "Cammeo" if real_class == 1 else "Osmancik"

    cammeo_probability = test_probabilities[i, 1]

    print(
        f"Ejemplo {i + 1}: "
        f"Probabilidad Cammeo = {cammeo_probability:.4f}, "
        f"Predicción = {predicted_name}, "
        f"Real = {real_name}"
    )


# GRÁFICA COMPARATIVA DE ACCURACY
sets = ["Train", "Validation"]

base_scores = [
    base_train_accuracy,
    base_val_accuracy,
]

tuned_scores = [
    tuned_train_accuracy,
    tuned_val_accuracy,
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

tuned_bars = ax.bar(
    x + width / 2,
    tuned_scores,
    width,
    label="Modelo ajustado",
)

ax.set_ylabel("Exactitud (accuracy)")
ax.set_title("Desempeño antes y después del ajuste")
ax.set_xticks(x)
ax.set_xticklabels(sets)
ax.set_ylim(0, 1.08)
ax.legend()
ax.grid(axis="y", alpha=0.25)

ax.bar_label(
    base_bars,
    labels=[f"{score * 100:.2f}%" for score in base_scores],
    padding=3,
)

ax.bar_label(
    tuned_bars,
    labels=[f"{score * 100:.2f}%" for score in tuned_scores],
    padding=3,
)

fig.tight_layout()

plt.savefig(
    "accuracy_comparison.png",
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
    display_labels=["Osmancik", "Cammeo"],
)

base_display.plot(
    ax=axes[0],
    cmap="Blues",
    colorbar=False,
)

axes[0].set_title("Modelo base - Validation")
axes[0].set_xlabel("Clase predicha")
axes[0].set_ylabel("Clase real")


tuned_display = ConfusionMatrixDisplay(
    confusion_matrix=tuned_val_confusion,
    display_labels=["Osmancik", "Cammeo"],
)

tuned_display.plot(
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
