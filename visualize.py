from matrix import Matrix
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

#рисуем scatter-график по первым двум глав. компонентам
def plot_pca_projection(X_proj: Matrix) -> Figure:
    #проверяем размерность: должно быть n × 2
    if X_proj.cols != 2:
        raise ValueError("Для визуализации нужны ровно 2 компоненты (столбца)")

    #разбиваем на два списка: x и y
    x_vals = [X_proj[i][0] for i in range(X_proj.rows)]
    y_vals = [X_proj[i][1] for i in range(X_proj.rows)]

    #строим график
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x_vals, y_vals, color='blue', alpha=0.7)
    ax.set_title("PCA: проекция на первые 2 компоненты")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True)

    return fig
