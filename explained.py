from typing import List

#функция для вычисления доли объяснённой дисперсии
def explained_variance_ratio(eigenvalues: List[float], k: int) -> float:
    #сортируем значения по убыванию (самые важные — первые)
    sorted_values = sorted(eigenvalues, reverse=True)

    #сумма первых k (главных компонент)
    top_k_sum = sum(sorted_values[:k])

    #сумма всех собственных значений
    total_sum = sum(sorted_values)

    #делим и возвращаем результат
    return top_k_sum / total_sum if total_sum != 0 else 0.0
