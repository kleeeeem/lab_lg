from typing import Tuple, List
from matrix import Matrix
from center import center_data
from covariance import covariance_matrix
from eigen import find_eigenvalues
from eigenvectors import find_eigenvectors
from explained import explained_variance_ratio

#реализация полного PCA
def pca(X: Matrix, k: int) -> Tuple[Matrix, float]:
    """
    X: матрица данных (n × m)
    k: число главных компонент
    ---
    Возвращает:
    - проекцию данных на первые k компонент (матрица размером n × k)
    - долю объяснённой дисперсии
    """

    # 1. Центрируем данные
    X_centered = center_data(X)

    # 2. Вычисляем ковариационную матрицу
    C = covariance_matrix(X_centered)

    # 3. Находим собственные значения и векторы
    eigenvalues = find_eigenvalues(C)
    eigenvectors = find_eigenvectors(C, eigenvalues)
    if len(eigenvectors) < k:
        print(f"⚠️ Недостаточно собственных векторов для k = {k}")
        return Matrix([]), 0.0

    # 4. Сортируем собственные значения и соответствующие векторы по убыванию
    eig_pairs = list(zip(eigenvalues, eigenvectors))
    eig_pairs.sort(reverse=True, key=lambda x: x[0])  # по убыванию λ

    # 5. Берём первые k векторов — формируем матрицу компонент (m × k)
    selected_vectors = [vec for (_, vec) in eig_pairs[:k]]  # каждый — Matrix (m × 1)

    components = []
    for j in range(k):
        column = [selected_vectors[j][i][0] for i in range(X.cols)]
        components.append(column)  # собираем столбцы как строки (транспонируем потом)

    #транспонируем: теперь компоненты — это матрица W размера m × k
    W = Matrix(list(map(list, zip(*components))))

    # 6. Проецируем X_centered (n × m) на W (m × k) → получим X_proj (n × k)
    n = X_centered.rows
    X_proj_data = []

    for i in range(n):
        row = []
        for j in range(k):
            dot = sum(X_centered[i][t] * W[t][j] for t in range(X.cols))
            row.append(dot)
        X_proj_data.append(row)

    X_proj = Matrix(X_proj_data)

    # 7. Вычисляем долю объяснённой дисперсии
    gamma = explained_variance_ratio(eigenvalues, k)

    return X_proj, gamma
