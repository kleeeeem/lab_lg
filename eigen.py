from matrix import Matrix

#удаляем i-ю строку и j-й столбец из матрицы — нужно для вычисления определителя
def minor(matrix: list, i: int, j: int) -> list:
    return [row[:j] + row[j+1:] for k, row in enumerate(matrix) if k != i]

#вычисляем определитель матрицы рекурсивно (разложение по первой строке)
def determinant(matrix: list) -> float:
    size = len(matrix)
    if size == 1:
        return matrix[0][0]
    if size == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

    det = 0
    for j in range(size):
        sign = (-1) ** j
        det += sign * matrix[0][j] * determinant(minor(matrix, 0, j))
    return det

#метод бисекции — находит корни det(C - λI) = 0
def find_eigenvalues(C: Matrix, tol: float = 1e-6) -> list:
    m = C.cols

    #начинаем с отрезка поиска (можно шире, если значения большие)
    a, b = -100, 100
    steps = 100  #на сколько частей делим отрезок

    eigenvalues = []

    #проверяем каждый отрезок [x, x+step] на смену знака определителя
    for i in range(steps):
        x1 = a + i * (b - a) / steps
        x2 = a + (i + 1) * (b - a) / steps

        #считаем определитель на концах отрезка
        det1 = determinant([
            [C[i][j] - x1 if i == j else C[i][j] for j in range(m)]
            for i in range(m)
        ])
        det2 = determinant([
            [C[i][j] - x2 if i == j else C[i][j] for j in range(m)]
            for i in range(m)
        ])

        #если знак поменялся — там есть корень (собственное значение)
        if det1 * det2 < 0:
            #применяем бисекцию
            while abs(x2 - x1) > tol:
                mid = (x1 + x2) / 2
                det_mid = determinant([
                    [C[i][j] - mid if i == j else C[i][j] for j in range(m)]
                    for i in range(m)
                ])
                if det1 * det_mid < 0:
                    x2 = mid
                    det2 = det_mid
                else:
                    x1 = mid
                    det1 = det_mid
            eigenvalues.append(round((x1 + x2) / 2, 6))

    # Удаляем близкие повторяющиеся значения (если разница < tol)
    unique_eigenvalues = []
    for val in eigenvalues:
        if all(abs(val - u) > tol for u in unique_eigenvalues):
            unique_eigenvalues.append(val)

    return unique_eigenvalues
