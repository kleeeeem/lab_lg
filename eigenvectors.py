from matrix import Matrix
from gauss import gauss_solver_with_general_solution
from typing import List

# Находит собственные векторы для каждого λ
def find_eigenvectors(C: Matrix, eigenvalues: List[float]) -> List[Matrix]:
    m = C.cols  # размерность матрицы C
    vectors = []

    for lam in eigenvalues:
        # Строим матрицу (C - λI)
        A = []
        for i in range(m):
            row = []
            for j in range(m):
                if i == j:
                    row.append(C[i][j] - lam)  # на диагонали: C[i][j] - λ
                else:
                    row.append(C[i][j])        # вне диагонали: просто C[i][j]
            A.append(row)

        # Правая часть = нулевая (однородная система)
        b = [[0] for _ in range(m)]

        # Решаем (C - λI)v = 0
        solution = gauss_solver_with_general_solution(Matrix(A), Matrix(b))

        # Если решение — строка, то это ошибка или несовместность
        if isinstance(solution, str):
            print(f"❗ Не удалось найти вектор для λ = {lam}")
            print("Причина:", solution)
            continue

        # Парсим результат (строки вида: x1 = ..., x2 = ..., ...)
        lines = solution.strip().split("\n")
        vector_values = []

        for line in lines:
            parts = line.split("=")
            expr = parts[1].strip()
            if any(char.isalpha() for char in expr):  # выражено через параметр
                vector_values.append(1.0)  # просто подставим 1 для свободной переменной
            else:
                try:
                    vector_values.append(float(expr))
                except ValueError:
                    print(f"⚠️ Не удалось разобрать строку: {line}")
                    break

        # Нормализуем вектор (по желанию)
        norm = sum(v**2 for v in vector_values) ** 0.5
        if norm != 0:
            vector_values = [v / norm for v in vector_values]

        vectors.append(Matrix([[val] for val in vector_values]))  # превращаем в вектор-столбец

    return vectors

