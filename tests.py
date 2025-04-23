from matrix import Matrix
from gauss import gauss_solver_with_general_solution
from center import center_data
from covariance import covariance_matrix
from eigen import find_eigenvalues
from eigenvectors import find_eigenvectors
from explained import explained_variance_ratio
from pca import pca
from visualize import plot_pca_projection
from matplotlib.figure import Figure
from reconstruction import reconstruction_error

# Тесты для метода Гаусса
def run_tests():
    test_cases = [
        {
            "name": "Единственное решение",
            "A": [[2, 1], [5, 7]],
            "b": [[11], [13]],
            "expected_contains": "x1 ="
        },
        {
            "name": "ФСР (бесконечно много решений)",
            "A": [[1, 2, -1], [2, 4, -2]],
            "b": [[3], [6]],
            "expected_contains": "x1 ="
        },
        {
            "name": "Несовместная система (пропорциональные строки)",
            "A": [[1, 2], [1, 2]],
            "b": [[4], [3]],
            "expected_contains": "Система несовместна"
        },
        {
            "name": "Удаляется нулевая строка",
            "A": [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
            "b": [[1], [2], [0]],
            "expected_contains": "x3 ="
        },
        {
            "name": "Противоречие: 0 0 = 5",
            "A": [[1, 0], [0, 1], [0, 0]],
            "b": [[2], [3], [5]],
            "expected_contains": "Система несовместна"
        },
        {
            "name": "ЛЗ одинаковые строки, совпадают правые части",
            "A": [[1, -1], [1, -1], [1, -1]],
            "b": [[0], [0], [0]],
            "expected_contains": "x1 ="
        },
        {
            "name": "ЛЗ одинаковые строки, правые части разные",
            "A": [[1, -1], [1, -1], [1, -1]],
            "b": [[0], [1], [2]],
            "expected_contains": "Система несовместна"
        }
    ]

    print("\n=== РЕЗУЛЬТАТЫ АВТОТЕСТОВ ДЛЯ МЕТОДА ГАУССА ===")
    for case in test_cases:
        A = Matrix(case["A"])
        b = Matrix(case["b"])
        result = gauss_solver_with_general_solution(A, b)
        if isinstance(result, str):
            passed = case["expected_contains"] in result
        else:
            passed = case["expected_contains"] in str(result)
        print(f"{case['name']}: {'✅ Passed' if passed else f'❌ Failed'}")


# Тесты для центрирования
def test_center_data():
    tests = [
        {
            "name": "Простая 3x2 матрица",
            "input": [[2, 4], [4, 8], [6, 6]],
            "expected": [[-2.0, -2.0], [0.0, 2.0], [2.0, 0.0]]
        },
        {
            "name": "Все элементы равны",
            "input": [[5, 5], [5, 5], [5, 5]],
            "expected": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        },
        {
            "name": "Нулевая матрица",
            "input": [[0, 0], [0, 0], [0, 0]],
            "expected": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        },
        {
            "name": "С отрицательными значениями",
            "input": [[1, -1], [3, -3], [5, -5]],
            "expected": [[-2.0, 2.0], [0.0, 0.0], [2.0, -2.0]]
        },
        {
            "name": "2x3 с разными числами",
            "input": [[1, 2, 3], [4, 5, 6]],
            "expected": [[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]]
        }
    ]

    print("\n=== РЕЗУЛЬТАТЫ АВТОТЕСТОВ ДЛЯ ЦЕНТРИРОВАНИЯ ===")
    for test in tests:
        X = Matrix(test["input"])
        centered = center_data(X).data
        expected = test["expected"]
        passed = all(
            abs(centered[i][j] - expected[i][j]) < 1e-6
            for i in range(len(expected)) for j in range(len(expected[0]))
        )
        print(f"{test['name']}: {'✅ Passed' if passed else f'❌ Failed'}")


# Тесты для ковариационной матрицы
def test_covariance_matrix():
    tests = [
        {
            "name": "Простая 3x2 матрица",
            "input": [[1, 2], [3, 4], [5, 6]],
            "expected": [[4.0, 4.0], [4.0, 4.0]]
        },
        {
            "name": "Одинаковые строки",
            "input": [[5, 5], [5, 5], [5, 5]],
            "expected": [[0.0, 0.0], [0.0, 0.0]]
        },
        {
            "name": "Нулевая матрица",
            "input": [[0, 0], [0, 0], [0, 0]],
            "expected": [[0.0, 0.0], [0.0, 0.0]]
        },
        {
            "name": "Сильная дисперсия",
            "input": [[1, 10], [2, 20], [3, 30]],
            "expected": [[1.0, 10.0], [10.0, 100.0]]
        }
    ]

    print("\n=== РЕЗУЛЬТАТЫ АВТОТЕСТОВ ДЛЯ КОВАРИАЦИЙ ===")
    for test in tests:
        X = Matrix(test["input"])
        X_centered = center_data(X)
        C = covariance_matrix(X_centered).data
        expected = test["expected"]
        passed = all(
            abs(C[i][j] - expected[i][j]) < 1e-6
            for i in range(len(expected)) for j in range(len(expected[0]))
        )
        print(f"{test['name']}: {'✅ Passed' if passed else f'❌ Failed'}")

# Тесты для собственных значений
def test_eigenvalues():
    tests = [
        {
            "name": "Матрица 2x2 с целыми λ",
            "input": [[2, 0], [0, 3]],
            "expected": [2.0, 3.0]
        },
        {
            "name": "Матрица 2x2 с одинаковыми λ",
            "input": [[4, 0], [0, 4]],
            "expected": [4.0]
        },
        {
            "name": "Диагональная 3x3",
            "input": [[1, 0, 0], [0, 5, 0], [0, 0, 10]],
            "expected": [1.0, 5.0, 10.0]
        }
    ]

    print("\n=== РЕЗУЛЬТАТЫ АВТОТЕСТОВ ДЛЯ СОБСТВЕННЫХ ЗНАЧЕНИЙ ===")
    for test in tests:
        C = Matrix(test["input"])
        found = find_eigenvalues(C)
        expected = test["expected"]

        # Проверка: у найденных значений разница с ожидаемыми < 1e-3
        passed = (
                all(any(abs(ev - f) < 1e-3 for f in found) for ev in expected) and
                all(any(abs(ev - f) < 1e-3 for ev in expected) for f in found)
        )
        print(f"{test['name']}: {'✅ Passed' if passed else f'❌ Failed (найдено: {found})'}")


# Тесты для собственных векторов
def test_eigenvectors():
    print("\n=== РЕЗУЛЬТАТЫ АВТОТЕСТОВ ДЛЯ СОБСТВЕННЫХ ВЕКТОРОВ ===")

    # Простой тест: λ = 2, 3, матрица диагональная
    C = Matrix([
        [2, 0],
        [0, 3]
    ])
    eigenvalues = [2.0, 3.0]

    vectors = find_eigenvectors(C, eigenvalues)

    passed = True

    # Проверяем: количество найденных векторов = количество λ
    if len(vectors) != len(eigenvalues):
        passed = False
        print("❌ Неверное количество собственных векторов")
    else:
        for i, vec in enumerate(vectors):
            if not isinstance(vec, Matrix) or vec.rows != 2 or vec.cols != 1:
                passed = False
                print(f"❌ v{i+1} не является вектором-столбцом нужного размера")

    if passed:
        print("✅ Passed: найдены векторы для диагональной матрицы")


# Тесты для объясненной дисперсии
def test_explained_variance():
    print("\n=== РЕЗУЛЬТАТЫ АВТОТЕСТОВ ДЛЯ ОБЪЯСНЁННОЙ ДИСПЕРСИИ ===")

    tests = [
        {
            "name": "3 значения, k = 1",
            "eigenvalues": [4.0, 3.0, 2.0],
            "k": 1,
            "expected": 4 / 9
        },
        {
            "name": "3 значения, k = 2",
            "eigenvalues": [4.0, 3.0, 2.0],
            "k": 2,
            "expected": (4 + 3) / 9
        },
        {
            "name": "k = 3 (вся сумма)",
            "eigenvalues": [4.0, 3.0, 2.0],
            "k": 3,
            "expected": 1.0
        },
        {
            "name": "Нулевые значения",
            "eigenvalues": [0.0, 0.0, 0.0],
            "k": 2,
            "expected": 0.0
        }
    ]

    for test in tests:
        result = explained_variance_ratio(test["eigenvalues"], test["k"])
        passed = abs(result - test["expected"]) < 1e-6
        print(f"{test['name']}: {'✅ Passed' if passed else f'❌ Failed (получено: {result:.6f})'}")

# Тесты для РСА
def test_pca():
    print("\n=== РЕЗУЛЬТАТЫ АВТОТЕСТОВ ДЛЯ PCA ===")

    #пример: простая матрица с понятной структурой
    X = Matrix([
        [2.5, 2.4],
        [0.5, 0.7],
        [2.2, 2.9],
        [1.9, 2.2],
        [3.1, 3.0],
        [2.3, 2.7],
        [2.0, 1.6],
        [1.0, 1.1],
        [1.5, 1.6],
        [1.1, 0.9]
    ])

    #запускаем PCA на 1 компоненту
    X_proj, gamma = pca(X, k=1)

    #проверяем:
    # 1. Размерность проекции — должно быть 10 строк и 1 столбец
    correct_shape = (X_proj.rows == 10 and X_proj.cols == 1)

    # 2. Доля объяснённой дисперсии — число от 0 до 1
    correct_ratio = (0 <= gamma <= 1)

    if correct_shape and correct_ratio:
        print("✅ Passed: PCA работает корректно (размерность + γ)")
    else:
        print("❌ Failed:")
        if not correct_shape:
            print(f"   - Ожидалась размерность 10×1, а получена {X_proj.rows}×{X_proj.cols}")
        if not correct_ratio:
            print(f"   - Неверное значение γ: {gamma}")


#mse
def test_reconstruction_error():
    print("\n=== РЕЗУЛЬТАТЫ АВТОТЕСТОВ ДЛЯ MSE ВОССТАНОВЛЕНИЯ ===")

    tests = [
        {
            "name": "Идеальное восстановление (ошибка = 0)",
            "X_orig": [[1, 2], [3, 4]],
            "X_recon": [[1, 2], [3, 4]],
            "expected": 0.0
        },
        {
            "name": "Одинаковое смещение на 1",
            "X_orig": [[1, 2], [3, 4]],
            "X_recon": [[2, 3], [4, 5]],
            "expected": 1.0
        },
        {
            "name": "Разные значения",
            "X_orig": [[1, 1], [1, 1]],
            "X_recon": [[2, 2], [2, 2]],
            "expected": 1.0
        }
    ]

    for test in tests:
        X1 = Matrix(test["X_orig"])
        X2 = Matrix(test["X_recon"])
        result = reconstruction_error(X1, X2)

        passed = abs(result - test["expected"]) < 1e-6
        print(f"{test['name']}: {'✅ Passed' if passed else f'❌ Failed (MSE = ' + str(result) + ')'}")

# ТЕсты для визуализации
def test_plot_pca_projection():
    print("\n=== РЕЗУЛЬТАТЫ АВТОТЕСТА ДЛЯ ВИЗУАЛИЗАЦИИ PCA ===")

    try:
        # Создаём простую матрицу проекций (3 строки, 2 компоненты)
        X_proj = Matrix([
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0]
        ])

        # Пытаемся построить график
        fig = plot_pca_projection(X_proj)

        # Проверяем тип
        if isinstance(fig, Figure):
            print("✅ Passed: график успешно построен")
            fig.show()  # Открываем окно с графиком
        else:
            print("❌ Failed: результат — не объект Figure")

    except Exception as e:
        print(f"❌ Failed: произошла ошибка при построении графика — {e}")