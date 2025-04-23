from matrix import Matrix
from gauss import gauss_solver_with_general_solution
from typing import List

#находит собственные векторы для каждого λ
def find_eigenvectors(C: Matrix, eigenvalues: List[float]) -> List[Matrix]:
    m = C.cols  #размерность матрицы C
    vectors = []

    for lam in eigenvalues:
        #строим матрицу (C - λI)
        A = []
        for i in range(m):
            row = []
            for j in range(m):
                if i == j:
                    row.append(C[i][j] - lam)  #на диагонали: C[i][j] - λ
                else:
                    row.append(C[i][j])        #вне диагонали: просто C[i][j]
            A.append(row)

        #правая часть = нулевая (однородная система)
        b = [[0] for _ in range(m)]

        #решаем (C - λI)v = 0
        solution = gauss_solver_with_general_solution(Matrix(A), Matrix(b))

        #если решение — строка, то это ошибка или несовместность
        if isinstance(solution, str):
            print(f"❗ Не удалось найти вектор для λ = {lam}")
            continue

        #парсим результат (строки вида: x1 = ..., x2 = ..., ...)
        lines = solution.strip().split("\n")
        vector_values = []

        for line in lines:
            parts = line.split("=")
            expr = parts[1].strip()
            if any(char.isalpha() for char in expr):  #выражено через параметр
                vector_values.append(1.0)  #просто подставим 1 для свободной переменной
            else:
                vector_values.append(float(expr))

        vectors.append(Matrix([[val] for val in vector_values]))  # превращаем в вектор-столбец

    return vectors
