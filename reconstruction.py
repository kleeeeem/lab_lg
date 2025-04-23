from matrix import Matrix

#вычисляет среднеквадратичную ошибку между X_orig и X_recon
def reconstruction_error(X_orig: Matrix, X_recon: Matrix) -> float:
    n = X_orig.rows
    m = X_orig.cols

    total = 0.0
    for i in range(n):
        for j in range(m):
            diff = X_orig[i][j] - X_recon[i][j]
            total += diff ** 2

    return total / (n * m)
