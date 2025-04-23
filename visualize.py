from matrix import Matrix
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Рисуем scatter-график по первым двум главным компонентам
def plot_pca_projection(X_proj: Matrix) -> Figure:
    # Проверяем размерность: должно быть n × 2
    if X_proj.cols != 2:
        raise ValueError("Для визуализации нужны ровно 2 компоненты (столбца)")

    # Разбиваем на два списка: x и y
    x_vals = [X_proj[i][0] for i in range(X_proj.rows)]
    y_vals = [X_proj[i][1] for i in range(X_proj.rows)]

    # Строим график
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x_vals, y_vals, color='blue', alpha=0.7)
    ax.set_title("PCA: проекция на первые 2 компоненты")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True)

    fig.show()

    return fig
