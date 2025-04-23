from matrix import Matrix
import math

def handle_missing_values(X: Matrix) -> Matrix:
    """
    Заполняет пропущенные значения NaN средним значением по каждому столбцу.
    """
    n = X.rows
    m = X.cols

    means = []
    for j in range(m):
        values = [X[i][j] for i in range(n) if not math.isnan(X[i][j])]
        mean_val = sum(values) / len(values) if values else 0.0
        means.append(mean_val)

    filled_data = []
    for i in range(n):
        row = []
        for j in range(m):
            value = X[i][j]
            row.append(means[j] if math.isnan(value) else value)
        filled_data.append(row)

    return Matrix(filled_data)
