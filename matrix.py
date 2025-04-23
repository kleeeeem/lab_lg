from typing import List

#чтобы удобно хранить и обрабатывать матрицы
class Matrix:
    def __init__(self, data: List[List[float]]):
        #сохраняем саму матрицу (список списков)
        self.data = data

        #количество строк в матрице
        self.rows = len(data)

        #количество столбцов в первой строке /
        #если матрица пустая, то столбцов нет
        self.cols = len(data[0]) if self.rows > 0 else 0

    def __getitem__(self, index):
        #это чтобы можно было писать X[0], X[1] и получать строки
        return self.data[index]

    def __str__(self):
        #это чтобы красиво печаталась матрица
        return '\n'.join([
            ' '.join([f'{val:.2f}' for val in row])  #каждое число округляется до 2 знаков
            for row in self.data
        ])
