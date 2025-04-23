from matrix import Matrix  #подключаем наш класс Matrix из matrix.py

#функция для вычисления ковариационной матрицы
def covariance_matrix(X_centered: Matrix) -> Matrix:
    #получаем количество строк и столбцов
    n = X_centered.rows
    m = X_centered.cols

    #транспонируем матрицу: меняем местами строки и столбцы
    X_T = []
    for j in range(m):  #проходим по каждому столбцу
        col = []
        for i in range(n):
            col.append(X_centered[i][j])
        X_T.append(col)  #каждая новая строка — это старый столбец

    #создаём матрицу m*m (такой же размер, как количество признаков)
    result = []
    for i in range(m):
        row = []
        for j in range(m):
            dot_product = 0
            for k in range(n):
                #скалярное произведение i-й строки X^T и j-го столбца X
                dot_product += X_T[i][k] * X_centered[k][j]
            #делим на (n - 1)(формула)
            row.append(dot_product / (n - 1))
        result.append(row)

    #возвращаем как объект Matrix
    return Matrix(result)
