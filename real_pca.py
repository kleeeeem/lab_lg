from typing import Tuple
from matrix import Matrix
from sklearn.datasets import load_wine
from pca import pca

def sklearn_data_to_matrix(data) -> Matrix:
    return Matrix([list(row) for row in data])

def apply_pca_to_dataset(dataset_name: str, k: int) -> Tuple[Matrix, float]:
    if dataset_name == "wine":
        print("📦 Загружаем встроенный датасет: wine")
        data = load_wine()
        X = sklearn_data_to_matrix(data.data)
    else:
        raise ValueError(f"Неизвестный датасет: {dataset_name}")

    print(f"Размерность данных: {X.rows}×{X.cols}")
    X_proj, gamma = pca(X, k)
    print(f"📊 Объяснённая дисперсия (γ) при k={k}: {gamma:.4f}")
    return X_proj, gamma
