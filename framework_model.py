import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

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


# ESCALAMIENTO CON FRAMEWORK
# Ya que las variables presentan escalas distintas,
# se aplica StandardScaler para estabilizar las medias
# y varianzas de las variables predictoras.
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# MODELO DE RANDOM FOREST
# Este primer modelo no tiene hiperparámetros ajustados
model = RandomForestClassifier(random_state=67)

model.fit(X_train, y_train)

val_pred = model.predict(X_val)

val_accuracy = accuracy_score(y_val, val_pred)

print("Validation accuracy:", val_accuracy)

print("\nClassification report:")
print(classification_report(y_val, val_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_val, val_pred))
