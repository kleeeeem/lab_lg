import random
from matrix import Matrix
from pca import pca
from explained import explained_variance_ratio

def add_noise_and_compare(X: Matrix, noise_level: float = 0.1):
    n, m = X.rows, X.cols

    # === PCA ДО ШУМА ===
    print("📊 PCA до добавления шума:")
    X_proj_before, gamma_before = pca(X, k=-1)
    print(f"Объяснённая дисперсия (до): γ = {gamma_before:.4f}")

    # === Добавляем шум ===
    print(f"\n🌪️ Добавляем шум: уровень = {noise_level}")
    noisy_data = []
    for i in range(n):
        row = []
        for j in range(m):
            val = X[i][j]
            noise = random.gauss(0, noise_level)
            row.append(val + noise)
        noisy_data.append(row)
    X_noisy = Matrix(noisy_data)

    # === PCA ПОСЛЕ ШУМА ===
    print("\n📊 PCA после добавления шума:")
    X_proj_after, gamma_after = pca(X_noisy, k=-1)
    print(f"Объяснённая дисперсия (после): γ = {gamma_after:.4f}")

    # === Анализ различий ===
    delta = abs(gamma_after - gamma_before)
    print(f"\n📉 Разница в объяснённой дисперсии: Δγ = {delta:.6f}")
    if delta < 0.05:
        print("✅ PCA устойчиво к данному уровню шума.")
    else:
        print("⚠️ Влияние шума заметно — снижение качества.")
