from typing import List
from matrix import Matrix
from gauss import gauss_solver_with_general_solution, print_solution_as_latex
from center import center_data
from covariance import covariance_matrix
from eigen import find_eigenvalues
from eigenvectors import find_eigenvectors
from explained import explained_variance_ratio
from pca import pca
#from visualize import plot_pca_projection
from reconstruction import reconstruction_error

from tests import run_tests, test_center_data, test_covariance_matrix, test_eigenvalues, test_eigenvectors, test_explained_variance, test_pca, test_reconstruction_error

# === Ввод матрицы с клавиатуры ===
def input_matrix_from_keyboard_safe():
    def safe_input_number(prompt):
        while True:
            try:
                return int(input(prompt))
            except ValueError:
                print("❗ Нужно ввести целое число.")

    def safe_input_row(prompt, expected_len):
        while True:
            raw = input(prompt)
            parts = raw.strip().split()
            if len(parts) != expected_len:
                print(f"❗ Введите ровно {expected_len} чисел.")
                continue
            try:
                return list(map(float, parts))
            except ValueError:
                print("❗ Ввод должен содержать только числа.")

    n = safe_input_number("Введите количество строк (уравнений): ")
    m = safe_input_number("Введите количество столбцов (переменных): ")

    A_data = []
    b_data = []

    print("\nВведите коэффициенты уравнений и правую часть:")
    for i in range(n):
        row = safe_input_row(f"  Строка {i+1} матрицы A (через пробел): ", m)
        A_data.append(row)
        rhs = safe_input_row(f"  Правая часть уравнения {i+1}: ", 1)[0]
        b_data.append([rhs])

    return Matrix(A_data), Matrix(b_data)

# === Основная программа ===

#вводим A и b
A, b = input_matrix_from_keyboard_safe()

#выводим решение СЛАУ
print("\n=== РЕШЕНИЕ СЛАУ (метод Гаусса) ===")
print_solution_as_latex(A, b)

#центрирование
X_centered = center_data(A)
print("\n=== ЦЕНТРИРОВАННАЯ МАТРИЦА A ===")
print(X_centered)

#ковариационная матрица
C = covariance_matrix(X_centered)
print("\n=== МАТРИЦА КОВАРИАЦИЙ ===")
print(C)

#собственные значения
print("\n=== СОБСТВЕННЫЕ ЗНАЧЕНИЯ ===")
eigenvalues = find_eigenvalues(C)
for i, val in enumerate(eigenvalues):
    print(f"λ{i+1} = {val:.6f}")

#собственные векторы
print("\n=== СОБСТВЕННЫЕ ВЕКТОРЫ ===")
eigenvectors = find_eigenvectors(C, eigenvalues)
for i, vec in enumerate(eigenvectors):
    print(f"v{i+1} =")
    print(vec)

#объясненная дисперсия
print("\n=== ОБЪЯСНЁННАЯ ДИСПЕРСИЯ ===")
for k in range(1, len(eigenvalues)+1):
    ratio = explained_variance_ratio(eigenvalues, k)
    print(f"γ({k}) = {ratio:.4f}")

#рса
print("\n=== PCA ===")
X_proj, gamma = pca(A, k=1)
print("Проекция данных (X_proj):")
print(X_proj)
print(f"Доля объяснённой дисперсии γ: {gamma:.4f}")

#ВОССТАНОВЛЕНИЕ данных из рса-проекции
def reconstruct_data(X_proj: Matrix, W: Matrix, means: List[float]) -> Matrix:
    """
    X_proj: матрица проекций (n×k)
    W: матрица главных компонент (m×k)
    means: список средних по колонкам (m)

    Возвращает восстановленную матрицу X_reconstructed (n×m)
    """
    n = X_proj.rows
    k = X_proj.cols
    m = len(means)

    # Транспонируем W (m×k → k×m)
    W_T = [[W[j][i] for j in range(k)] for i in range(m)]

    # Умножаем X_proj (n×k) × W_T (k×m) → (n×m)
    recon_data = []
    for i in range(n):
        row = []
        for j in range(m):
            dot = sum(X_proj[i][t] * W_T[t][j] for t in range(k))
            row.append(dot + means[j])
        recon_data.append(row)

    return Matrix(recon_data)


#визуализация проекции
#if X_proj.cols >= 2:
#    fig = plot_pca_projection(X_proj)
#    fig.show()  # открывает окно с графиком
#else:
#    print("⚠️ Нельзя построить график — нужно хотя бы 2 компоненты.")

# Средние значения по признакам
means = [sum(A[i][j] for i in range(A.rows)) / A.rows for j in range(A.cols)]

# Повторное получение C, λ и собственных векторов
C = covariance_matrix(center_data(A))
eigenvalues = find_eigenvalues(C)
eigenvectors = find_eigenvectors(C, eigenvalues)

# Выбираем первые k собственных векторов
k = 1
eig_pairs = sorted(zip(eigenvalues, eigenvectors), reverse=True, key=lambda x: x[0])
selected_vectors = [v for _, v in eig_pairs[:k]]

# Строим матрицу W (m × k)
if len(selected_vectors) < k:
    print(f"⚠️ Невозможно построить W: недостаточно векторов для k = {k}")
else:
    W = Matrix([
        [selected_vectors[j][i][0] for j in range(k)]
        for i in range(A.cols)
    ])
# Восстановление данных
    X_reconstructed = reconstruct_data(X_proj, W, means)
    mse = reconstruction_error(A, X_reconstructed)
    print(f"\nMSE восстановления: {mse:.6f}")

#запуск тестов
print("\n=== АВТОТЕСТЫ ===")
run_tests()
test_center_data()
test_covariance_matrix()
test_eigenvalues()
test_eigenvectors()
test_explained_variance()
test_pca()
#test_plot_pca_projection()
test_reconstruction_error()