from matrix import Matrix
from string import ascii_lowercase
from typing import List, Union

#проверяем, нет ли одинаковых строк (чтобы убрать лишние или найти противоречие)
def remove_linearly_dependent_rows_strict(matrix: List[List[float]]) -> Union[List[List[float]], str]:
    def is_proportional_strict(row1, row2):
        ratio = None
        for a, b in zip(row1[:-1], row2[:-1]):
            if abs(b) < 1e-10:
                if abs(a) > 1e-10:
                    return False, None
            else:
                if ratio is None:
                    ratio = a / b
                elif abs(a - ratio * b) > 1e-10:
                    return False, None
        return True, ratio

    filtered = []
    for row in matrix:
        for existing in filtered:
            proportional, ratio = is_proportional_strict(row, existing)
            if proportional:
                rhs1 = row[-1]
                rhs2 = existing[-1]
                if ratio is None:
                    if abs(rhs1 - rhs2) > 1e-10:
                        return "Система несовместна: одинаковые коэффициенты, но разные правые части"
                else:
                    if abs(rhs1 - rhs2 * ratio) > 1e-10:
                        return "Система несовместна: строки пропорциональны, но правая часть не совпадает"
                break
        else:
            filtered.append(row)
    return filtered

#решаем СЛАУ методом Гаусса
def gauss_solver_with_general_solution(A: Matrix, b: Matrix) -> Union[str, List[str]]:
    n = A.rows
    m = A.cols

    #склеиваем A и b в одну общую матрицу
    Ab = [A[i] + b[i] for i in range(n)]

    #удаляем лишние нулевые строки или противоречия
    filtered_Ab = []
    for row in Ab:
        coeffs = row[:-1]
        rhs = row[-1]
        if all(abs(x) < 1e-10 for x in coeffs):
            if abs(rhs) < 1e-10:
                continue
            else:
                return "Система несовместна: строка вида 0 + 0 + ... = неноль"
        filtered_Ab.append(row)

    Ab = remove_linearly_dependent_rows_strict(filtered_Ab)
    if isinstance(Ab, str):
        return Ab
    n = len(Ab)

    #сам метод Гаусса: идём по главной диагонали
    rank = 0
    for i in range(min(n, m)):
        max_row = max(range(i, n), key=lambda r: abs(Ab[r][i]))
        if abs(Ab[max_row][i]) < 1e-10:
            continue
        Ab[i], Ab[max_row] = Ab[max_row], Ab[i]
        lead = Ab[i][i]
        Ab[i] = [x / lead for x in Ab[i]]
        for j in range(n):
            if j != i:
                factor = Ab[j][i]
                Ab[j] = [Ab[j][k] - factor * Ab[i][k] for k in range(m + 1)]
        rank += 1

    #определяем где свободные переменные, а где главные
    pivot_positions = [-1] * n
    pivot_cols = []
    for i in range(n):
        for j in range(m):
            if abs(Ab[i][j] - 1) < 1e-10:
                if all(abs(Ab[r][j]) < 1e-10 for r in range(n) if r != i):
                    pivot_positions[i] = j
                    pivot_cols.append(j)
                    break

    #свободные переменные — те, которых нет в pivot_cols
    free_vars = [j for j in range(m) if j not in pivot_cols]
    var_names = [f'x{j+1}' for j in range(m)]
    param_names = [ascii_lowercase[i] for i in range(len(free_vars))]

    #собираем общее решение
    general_solution = [""] * m
    for j in range(m):
        if j in free_vars:
            idx = free_vars.index(j)
            general_solution[j] = f"{param_names[idx]}"
        else:
            row_index = pivot_positions.index(j)
            value = Ab[row_index][-1]
            expr = [f"{value:.2f}"]
            for k in free_vars:
                coeff = -Ab[row_index][k]
                if abs(coeff) > 1e-10:
                    term = f"{'-' if coeff > 0 else '+'} {abs(coeff):.2f}{param_names[free_vars.index(k)]}"
                    expr.append(term)
            general_solution[j] = ' '.join(expr)

    result = "\n".join(f"{var_names[j]} = {general_solution[j]}" for j in range(m))
    return result

#красивый вывод решения в формате вектора 
def print_solution_as_vector_form(A: Matrix, b: Matrix):
    result = gauss_solver_with_general_solution(A, b)
    if isinstance(result, str):
        print(result)
        return

    lines = result.strip().split("\n")
    m = len(lines)

    const = []
    param_vectors = {}

    for line in lines:
        parts = line.split("=")
        expr = parts[1].strip()
        tokens = expr.split()

        value = 0.0
        terms = {}
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token.replace('.', '', 1).replace('-', '', 1).isdigit():
                number = float(token)
                if i == 0:
                    value = number
                else:
                    sign = -1 if tokens[i - 1] == "-" else 1
                    var = tokens[i + 1]
                    terms[var] = sign * number
                    i += 1
            i += 1

        const.append(value)
        for param in terms:
            if param not in param_vectors:
                param_vectors[param] = [0.0] * m
            param_vectors[param][len(const)-1] = terms[param]

    # Печатаем как LaTeX-вектора
    print("\nОбщее решение в векторной форме:")
    print("x =")
    print("\\[")
    print("\\begin{bmatrix}")
    for i in range(m):
        end = " \\\\" if i < m - 1 else ""
        print(f"{const[i]:6.2f}{end}")
    print("\\end{bmatrix}", end="")

    for param in sorted(param_vectors):
        print(f" + {param} \\cdot \\begin{{bmatrix}}")
        for i in range(m):
            end = " \\\\" if i < m - 1 else ""
            print(f"{param_vectors[param][i]:6.2f}{end}")
        print("\\end{bmatrix}", end="")

    print("\n\\]")

    lines = result.strip().split("\n")
    m = len(lines)
    const = []
    param_vectors = {}

    for line in lines:
        parts = line.split("=")
        expr = parts[1].strip()
        tokens = expr.split()
        value = 0.0
        terms = {}
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token.replace('.', '', 1).replace('-', '', 1).isdigit():
                number = float(token)
                if i == 0:
                    value = number
                else:
                    sign = -1 if tokens[i - 1] == "-" else 1
                    var = tokens[i + 1]
                    terms[var] = sign * number
                    i += 1
            i += 1
        const.append(value)
        for param in terms:
            if param not in param_vectors:
                param_vectors[param] = [0.0] * m
            param_vectors[param][len(const)-1] = terms[param]

    print("\nРешение в формате LaTeX (ФСР):")
    print("\\[")
    print("x = ")
    print("\\begin{bmatrix}")
    for i in range(m):
        end = " \\\\" if i < m - 1 else ""
        print(f"{const[i]:.2f}{end}")
    print("\\end{bmatrix}", end="")

    for param in sorted(param_vectors):
        print(f" + {param} \\cdot \\begin{{bmatrix}}")
        for i in range(m):
            end = " \\\\" if i < m - 1 else ""
            print(f"{param_vectors[param][i]:.2f}{end}")
        print("\\end{bmatrix}", end="")

    print("\n\\]")
