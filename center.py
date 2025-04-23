from matrix import Matrix  #подключаем класс Matrix из matrix.py

#функция, которая выполняет центрирование матрицы
def center_data(X: Matrix) -> Matrix:
    #получаем размеры матрицы
    n = X.rows
    m = X.cols
    #считаем среднее значение по каждому столбцу
    means = []
    for j in range(m):
        col_sum = 0
        for i in range(n):
            col_sum += X[i][j]  #складываем все значения в j-м столбце
        means.append(col_sum / n)  #делим сумму на количество строк

    #вычитаем среднее из каждого элемента — получаем центрированную матрицу
    centered = []
    for i in range(n):
        row = []
        for j in range(m):
            row.append(X[i][j] - means[j])  #каждый элемент минус среднее по его столбцу
        centered.append(row)

    #возвращаем результат как объект Matrix
    return Matrix(centered)
