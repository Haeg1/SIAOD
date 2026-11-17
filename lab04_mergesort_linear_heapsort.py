#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ЛР 4. MergeSort, Counting, Radix и HeapSort: оценка производительности (решение).

КИМ-04, вариант задаётся флагом ``--variant``; seed — ``30 + вариант``.
QuickSort перенесён из собственной ЛР 3 (``lab03_simple_sorts_quicksort.py``).

Запуск:
    python lab04_mergesort_linear_heapsort.py --variant 15
    python lab04_mergesort_linear_heapsort.py --variant 15 --out .

Что делает:
  1) MergeSort со стабильным слиянием (key-функция, пары «ключ + номер»);
  2) CountingSort — устойчивый вариант с префиксными суммами, + опыт «время и
     память как функции k = max − min»;
  3) RadixSort LSD по десятичным разрядам поверх устойчивого поразрядного прохода,
     + демонстрация того, что неустойчивый проход даёт неверный результат;
  4) HeapSort in-place через просейку вниз (построение кучи за O(n));
  5) бенчмарк шести сортировок против QuickSort и sorted() на пяти
     распределениях данных (методика docs/reproducibility.md);
  6) верификация по docs/ai-verification.md (инварианты, границы, сверка с sorted()).
"""
from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

# ---------------------------------------------------------------------------
# Параметры эксперимента
# ---------------------------------------------------------------------------

REPEATS = 5
SIZES = (1_000, 3_000, 10_000, 30_000, 100_000)
SMALL_KEY_MAX = 100          # узкий диапазон: counting уместен
WIDE_KEY_MAX = 10 ** 9       # широкий диапазон: counting теряет смысл
QUICK_SMALL = 16             # порог вставок внутри QuickSort (как в ЛР 3)

# Опыт «counting_sort: цена диапазона ключей»
RANGE_SIZES = (10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000)
RANGE_N = 100_000

DISTRIBUTIONS = ("random_wide", "random_small", "sorted", "reversed", "almost_sorted")


def _identity(value):
    """Ключ по умолчанию: сортировка по самому элементу."""
    return value


# ---------------------------------------------------------------------------
# 1. MergeSort со стабильным слиянием
# ---------------------------------------------------------------------------


def merge(left: list, right: list, key=_identity) -> list:
    """Стабильное слияние двух отсортированных списков в новый список.

    Устойчивость обеспечивает нестрогое сравнение ``key(left[i]) <= key(right[j])``:
    при равных ключах элемент левой половины ставится раньше правого, а левая
    половина содержит более «ранние» элементы исходного массива.

    Время Θ(n), память Θ(n) на результат слияния.
    """
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if key(left[i]) <= key(right[j]):
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def merge_sort(values: list, key=_identity) -> list:
    """MergeSort: возвращает новый отсортированный список.

    Время Θ(n log n) на любом входе (деление пополам + линейное слияние),
    память Θ(n). Рекурсия глубины log₂ n.
    """
    a = list(values)
    if len(a) <= 1:
        return a
    mid = len(a) // 2
    return merge(merge_sort(a[:mid], key), merge_sort(a[mid:], key), key)


# ---------------------------------------------------------------------------
# 2. CountingSort: устойчивый вариант с префиксными суммами
# ---------------------------------------------------------------------------


def counting_sort(values: list, key_max: int, key=_identity) -> list:
    """Устойчивая сортировка подсчётом: ключи — целые из [0, key_max].

    Время Θ(n + k), память Θ(n + k), где k = key_max + 1. Ключи вне диапазона —
    ValueError: расширять диапазон «на лету» нельзя, иначе теряется сама идея
    адресации по значению ключа.

    Порядок устойчивости: массив счётчиков -> префиксные суммы (позиции конца
    блока) -> проход расстановки **с конца массива**. Проход с начала по тем же
    префиксным суммам дал бы обратный порядок внутри равных ключей.
    """
    a = list(values)
    n = len(a)
    if n == 0:
        return []
    counts = [0] * (key_max + 1)
    for x in a:
        k = key(x)
        if not 0 <= k <= key_max:
            raise ValueError(f"ключ {k} вне диапазона [0, {key_max}]")
        counts[k] += 1
    for i in range(1, key_max + 1):
        counts[i] += counts[i - 1]
    out = [None] * n
    for x in reversed(a):
        k = key(x)
        counts[k] -= 1
        out[counts[k]] = x
    return out


# ---------------------------------------------------------------------------
# 3. RadixSort LSD по десятичным разрядам
# ---------------------------------------------------------------------------


def _counting_pass_by_digit(a: list, exp: int, key) -> list:
    """Один устойчивый проход по разряду: цифра = (key(x) // exp) % 10.

    Устойчивость прохода обязательна: LSD обрабатывает разряды от младшего к
    старшему, и порядок, наведённый на младших разрядах, должен сохраниться при
    сортировке по старшим. Именно устойчивость «запоминает» результат
    предыдущего прохода внутри одинаковых старших цифр.
    """
    counts = [0] * 10
    for x in a:
        counts[(key(x) // exp) % 10] += 1
    for d in range(1, 10):
        counts[d] += counts[d - 1]
    out = [None] * len(a)
    for x in reversed(a):
        d = (key(x) // exp) % 10
        counts[d] -= 1
        out[counts[d]] = x
    return out


def radix_sort_lsd(values: list[int], key=_identity) -> list[int]:
    """RadixSort LSD по десятичным разрядам, неотрицательные целые.

    Время Θ(d·(n + 10)) = Θ(d·n), где d — число десятичных разрядов максимума;
    память Θ(n + 10) — на порядок меньше, чем у CountingSort при большом k.

    Отрицательные ключи — ValueError: разряды LSD для них не определены, а
    молча вернуть неотсортированный массив хуже явной ошибки.
    """
    a = list(values)
    if not a:
        return []
    min_key, max_key = min(key(x) for x in a), max(key(x) for x in a)
    if min_key < 0:
        raise ValueError(f"radix_sort_lsd: отрицательный ключ {min_key}; ожидались целые >= 0")
    exp = 1
    while max_key // exp > 0:
        a = _counting_pass_by_digit(a, exp, key)
        exp *= 10
    return a


def _counting_pass_by_digit_unstable(a: list, exp: int, key) -> list:
    """Намеренно неустойчивый проход — только для демонстрации ошибки."""
    buckets: list[list] = [[] for _ in range(10)]
    for x in a:
        buckets[(key(x) // exp) % 10].append(x)
    out: list = []
    for bucket in buckets:
        out.extend(reversed(bucket))   # ломает порядок внутри равных цифр
    return out


def radix_sort_lsd_unstable(values: list[int], key=_identity) -> list[int]:
    """RadixSort с неустойчивым внутренним проходом — даёт неверный результат."""
    a = list(values)
    if not a:
        return []
    min_key, max_key = min(key(x) for x in a), max(key(x) for x in a)
    if min_key < 0:
        raise ValueError(
            f"radix_sort_lsd_unstable: отрицательный ключ {min_key}; ожидались целые >= 0")
    exp = 1
    while max_key // exp > 0:
        a = _counting_pass_by_digit_unstable(a, exp, key)
        exp *= 10
    return a


# ---------------------------------------------------------------------------
# 4. HeapSort: in-place, просейка вниз
# ---------------------------------------------------------------------------


def sift_down(a: list[int], i: int, n: int) -> None:
    """Просейка вниз элемента i в куче a[0:n] (max-heap), на месте."""
    while True:
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n and a[left] > a[largest]:
            largest = left
        if right < n and a[right] > a[largest]:
            largest = right
        if largest == i:
            return
        a[i], a[largest] = a[largest], a[i]
        i = largest


def build_heap(a: list[int]) -> None:
    """Построение max-кучи за Θ(n): просейки от последнего узла с потомками вверх."""
    for i in range(len(a) // 2 - 1, -1, -1):
        sift_down(a, i, len(a))


def heap_sort(a: list[int]) -> None:
    """HeapSort in-place: сортирует список на месте, без вспомогательного массива.

    Время Θ(n log n) на любом входе (Θ(n) на построение кучи + n извлечений по
    Θ(log n)), память Θ(1) сверх массива. Нестабильна: обмен вершины с
    последним элементом переставляет равные ключи через весь массив.
    """
    n = len(a)
    build_heap(a)
    for end in range(n - 1, 0, -1):
        a[0], a[end] = a[end], a[0]
        sift_down(a, 0, end)


# ---------------------------------------------------------------------------
# 5. QuickSort — перенесено из ЛР 3 (КИМ-03)
# ---------------------------------------------------------------------------


def _partition_three_way(a: list, lo: int, hi: int, pivot_index: int, key,
                         counters: list[int]) -> tuple[int, int]:
    """Трёхпутевое разбиение вокруг pivot_index; возвращает границы блока равных."""
    a[pivot_index], a[hi] = a[hi], a[pivot_index]
    pivot = a[hi]
    lt, i, gt = lo, lo, hi - 1
    while i <= gt:
        if key(a[i]) < key(pivot):
            counters[0] += 1
            a[lt], a[i] = a[i], a[lt]
            counters[1] += 1
            lt += 1
            i += 1
        else:
            counters[0] += 2
            if key(a[i]) > key(pivot):
                a[i], a[gt] = a[gt], a[i]
                counters[1] += 1
                gt -= 1
            else:
                i += 1
    a[gt + 1], a[hi] = a[hi], a[gt + 1]
    counters[1] += 1
    return lt, gt + 1


def _insertion_range(a: list, lo: int, hi: int, key, counters: list[int]) -> None:
    """Вставки на подмассиве a[lo:hi+1] — хвост QuickSort для малых отрезков."""
    for i in range(lo + 1, hi + 1):
        current = a[i]
        j = i - 1
        while j >= lo:
            counters[0] += 1
            if key(a[j]) <= key(current):
                break
            a[j + 1] = a[j]
            counters[1] += 1
            j -= 1
        a[j + 1] = current


def quick_sort(values: list, rng: random.Random | None = None, key=_identity,
               counters: list[int] | None = None) -> list:
    """QuickSort с рандомизированным опорным элементом (перенесён из ЛР 3).

    Среднее Θ(n log n), худшее Θ(n²) — рандомизация делает его маловероятным.
    """
    if rng is None:
        rng = random.Random()
    if counters is None:
        counters = [0, 0]
    a = list(values)
    stack = [(0, len(a) - 1)]
    while stack:
        lo, hi = stack.pop()
        while hi - lo + 1 > QUICK_SMALL:
            pivot_index = rng.randrange(lo, hi + 1)
            lt, gt = _partition_three_way(a, lo, hi, pivot_index, key, counters)
            if lt - lo < hi - gt:
                stack.append((gt + 1, hi))
                hi = lt - 1
            else:
                stack.append((lo, lt - 1))
                lo = gt + 1
        if hi > lo:
            _insertion_range(a, lo, hi, key, counters)
    return a


# ---------------------------------------------------------------------------
# 6. Верификация: инварианты + сверка с эталоном sorted()
# ---------------------------------------------------------------------------


class SelfCheckError(Exception):
    """Нарушен контракт или инвариант реализации."""


def expect(condition: bool, message: str) -> None:
    """Явная проверка вместо assert: assert-ы исчезают при запуске python -O."""
    if not condition:
        raise SelfCheckError(message)


def heap_sorted(a: list[int]) -> list[int]:
    """heap_sort сортирует на месте: проверяется тот список, который ему передали."""
    buf = list(a)
    heap_sort(buf)
    return buf


# Каждая функция возвращает отсортированный список; ключи — целые из [0, 999].
SORTS = {
    "merge_sort": lambda a: merge_sort(a),
    "counting_sort": lambda a: counting_sort(a, key_max=999),
    "radix_sort_lsd": lambda a: radix_sort_lsd(a),
    "heap_sort": lambda a: heap_sorted(a),
    "quick_sort": lambda a: quick_sort(a),
}


def check_sort(fn) -> None:
    """Граничные случаи, инварианты сортировки и сверка с эталоном sorted()."""
    boundary = [
        [],                    # пустой массив
        [7],                   # один элемент
        [0],                   # нулевой ключ
        [1, 0],                # два элемента
        [5, 5, 5],             # все элементы равны
        [1, 2, 3, 4, 5],       # уже отсортирован
        [5, 4, 3, 2, 1],       # обратный порядок
        [10, 0, 10, 0, 5, 5],  # дубликаты
    ]
    for a in boundary:
        res = fn(list(a))
        expect(isinstance(res, list), f"на {a} возвращён {type(res).__name__}, а не список")
        expect(sorted(res) == sorted(a), f"не перестановка входа {a}: получено {res}")
        expect(res == sorted(a), f"на {a} получено {res}, ожидалось {sorted(a)}")

    # Случайные тесты: инвариант перестановки + сверка с эталоном
    rng = random.Random(0)
    for _ in range(200):
        a = [rng.randint(0, 999) for _ in range(rng.randint(0, 60))]
        res = fn(list(a))
        expect(sorted(res) == sorted(a), f"не перестановка входа {a}: получено {res}")
        expect(res == sorted(a), f"расходится с sorted() на {a}: получено {res}")


def check_merge_stability() -> None:
    """Стабильность слияния на парах (ключ, порядковый номер).

    Эталон sorted() стабилен, поэтому при равных ключах порядок номеров обязан
    совпасть. 200 наборов, а не один: стабильность — инвариант.
    """
    rng = random.Random(0)
    for _ in range(200):
        pairs = [(rng.randint(0, 5), i) for i in range(rng.randint(0, 100))]
        got = merge_sort(pairs, key=lambda p: p[0])
        expect(got == sorted(pairs, key=lambda p: p[0]),
               f"merge_sort нестабилен на {pairs[:10]}…: нарушен порядок номеров")


def check_radix_stability() -> None:
    """Устойчивость RadixSort LSD на парах (ключ, номер) — тот же эталон."""
    rng = random.Random(1)
    for _ in range(100):
        pairs = [(rng.randint(0, 999), i) for i in range(rng.randint(0, 80))]
        got = radix_sort_lsd(pairs, key=lambda p: p[0])
        expect(got == sorted(pairs, key=lambda p: p[0]),
               "radix_sort_lsd неустойчив: нарушен порядок номеров при равных ключах")


def check_counting_stability() -> None:
    """CountingSort устойчив: расстановка идёт с конца массива по префиксным суммам."""
    rng = random.Random(4)
    for _ in range(100):
        pairs = [(rng.randint(0, 9), i) for i in range(rng.randint(0, 80))]
        got = counting_sort(pairs, key_max=9, key=lambda p: p[0])
        expect(got == sorted(pairs, key=lambda p: p[0]),
               "counting_sort неустойчив: нарушен порядок номеров при равных ключах")


def check_radix_bounds() -> None:
    """RadixSort отвергает отрицательные ключи: они вне области применимости."""
    for fn in (radix_sort_lsd, radix_sort_lsd_unstable):
        try:
            fn([3, -1, 2])
        except ValueError:
            pass
        else:
            raise SelfCheckError(f"{fn.__name__} принял отрицательный ключ")
    expect(radix_sort_lsd([0, 0, 0]) == [0, 0, 0], "radix_sort_lsd: нулевые ключи")


def check_radix_unstable_fails() -> None:
    """Неустойчивый проход обязан давать неверный результат — иначе тест пуст."""
    broken = 0
    rng = random.Random(2)
    for _ in range(50):
        a = [rng.randint(0, 999) for _ in range(rng.randint(2, 40))]
        if radix_sort_lsd_unstable(a) != sorted(a):
            broken += 1
    expect(broken > 0,
           "неустойчивый вариант radix дал верный результат — демонстрация несостоятельна")

    # Показательный пример из отчёта.
    example = [21, 1, 11, 2, 12, 3]
    expect(radix_sort_lsd(example) == sorted(example), "radix: ошибка на примере")
    expect(radix_sort_lsd_unstable(example) != sorted(example),
           "radix: неустойчивый проход случайно дал верный результат на примере")


def check_heap_property() -> None:
    """После build_heap выполняется свойство max-кучи, heap_sort не тратит память."""
    rng = random.Random(3)
    for _ in range(100):
        a = [rng.randint(-100, 100) for _ in range(rng.randint(0, 50))]
        build_heap(a)
        for i in range(1, len(a)):
            parent = (i - 1) // 2
            expect(a[parent] >= a[i], f"свойство кучи нарушено в узле {i}: {a}")
        # heap_sort обязан изменить переданный список на месте и вернуть None
        buf = list(a)
        expect(heap_sort(buf) is None, "heap_sort вернул значение — нарушен контракт in-place")
        expect(buf == sorted(a), f"heap_sort не отсортировал на месте: {buf}")


def check_counting_bounds() -> None:
    """CountingSort отвергает ключи вне заявленного диапазона, а не портит память."""
    try:
        counting_sort([1, 2, 1000], key_max=999)
    except ValueError:
        pass
    else:
        raise SelfCheckError("counting_sort не отверг ключ вне диапазона")
    expect(counting_sort([], key_max=999) == [], "counting_sort: пустой вход")
    expect(counting_sort([0, 0, 0], key_max=0) == [0, 0, 0], "counting_sort: k = 0")


def self_check() -> None:
    """Все проверки по разделам; при любой ошибке замеры не выполняются."""
    # fn=fn фиксирует функцию текущего шага цикла (иначе все лямбды увидят последнюю)
    checks = [(name, lambda fn=fn: check_sort(fn)) for name, fn in SORTS.items()]
    checks.append(("стабильность merge_sort", check_merge_stability))
    checks.append(("устойчивость counting_sort", check_counting_stability))
    checks.append(("устойчивость radix_sort_lsd", check_radix_stability))
    checks.append(("границы radix_sort_lsd", check_radix_bounds))
    checks.append(("неустойчивый radix ломает результат", check_radix_unstable_fails))
    checks.append(("свойство кучи и in-place heap_sort", check_heap_property))
    checks.append(("границы диапазона counting_sort", check_counting_bounds))
    failed = 0
    print("self_check:")
    for name, check in checks:
        try:
            check()
        except SelfCheckError as err:
            print(f"  {name}: ОШИБКА — {err}")
            failed += 1
        except Exception:  # noqa: BLE001 — падение реализации тоже результат проверки
            print(f"  {name}: ИСКЛЮЧЕНИЕ")
            raise
        else:
            print(f"  {name}: OK")
    if failed:
        raise SystemExit(f"self_check: не пройдено разделов — {failed}; "
                         f"замеры выполняются только после всех проверок.")
    print("self_check: OK")


# ---------------------------------------------------------------------------
# 7. Бенчмарк (методика — docs/reproducibility.md)
# ---------------------------------------------------------------------------


def make_datasets(rng: random.Random, n: int) -> dict[str, list[int]]:
    """Массивы разных распределений и диапазонов ключей для размера n."""
    almost = list(range(n))
    for _ in range(max(1, n // 100)):  # ~1 % случайных обменов
        i, j = rng.randrange(n), rng.randrange(n)
        almost[i], almost[j] = almost[j], almost[i]
    return {
        "random_wide": [rng.randint(0, WIDE_KEY_MAX) for _ in range(n)],
        "random_small": [rng.randint(0, SMALL_KEY_MAX) for _ in range(n)],
        "sorted": list(range(n)),
        "reversed": list(range(n, 0, -1)),
        "almost_sorted": almost,
    }


def bench(fn, data: list[int]) -> float:
    """Медиана времени fn на копии data по REPEATS запускам, с прогревом.

    Каждый запуск получает свежую копию: сортировка на месте не должна
    превращать последующие запуски в сортировку уже отсортированного массива.
    """
    fn(list(data))  # прогрев — не учитывается
    times = []
    for _ in range(REPEATS):
        buf = list(data)
        t0 = time.perf_counter()
        fn(buf)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def log_log_slope(points: list[tuple[int, float]]) -> float:
    """Наклон log-log прямой (МНК): 1 — линейный рост, 2 — квадратичный."""
    if len(points) < 2:
        raise ValueError("для наклона log-log нужно минимум две точки")
    xs = [math.log10(n) for n, _ in points]
    ys = [math.log10(t) for _, t in points]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denominator = sum((x - mx) ** 2 for x in xs)
    return numerator / denominator


ALGORITHM_ORDER = (
    "merge_sort", "counting_sort", "radix_sort_lsd", "heap_sort",
    "quick_sort (ЛР 3)", "sorted() (эталон)",
)


def algorithm_table() -> list[tuple[str, object]]:
    """Шесть сортировок бенчмарка: четыре свои, QuickSort из ЛР 3 и эталон."""
    return [
        ("merge_sort", merge_sort),
        ("counting_sort", lambda a: counting_sort(a, key_max=max(a, default=0))),
        ("radix_sort_lsd", radix_sort_lsd),
        ("heap_sort", heap_sort),
        ("quick_sort (ЛР 3)", quick_sort),
        ("sorted() (эталон)", sorted),
    ]


def collect_times(seed: int) -> dict[str, dict[str, list[tuple[int, float]]]]:
    """Время: алгоритм -> распределение -> [(n, медиана)].

    CountingSort на random_wide пропускается: диапазон ключей ~10⁹ означает
    массив счётчиков в 8 ГБ — это и есть граница применимости алгоритма.
    """
    rng = random.Random(seed)
    datasets = {n: make_datasets(rng, n) for n in SIZES}
    results: dict[str, dict[str, list[tuple[int, float]]]] = {
        name: {dist: [] for dist in DISTRIBUTIONS} for name in ALGORITHM_ORDER
    }
    for name, fn in algorithm_table():
        for dist in DISTRIBUTIONS:
            if name == "counting_sort" and dist == "random_wide":
                continue
            for n in SIZES:
                results[name][dist].append((n, bench(fn, datasets[n][dist])))
    return results


def print_properties_table() -> None:
    """Сводная таблица свойств шести сортировок бенчмарка (КИМ-04, п. 6)."""
    print("\n## Сводная таблица свойств\n")
    print("| Алгоритм | Лучший | Средний | Худший | Доп. память | Стабильность "
          "| Ограничения |")
    print("| --- | :---: | :---: | :---: | :---: | :---: | --- |")
    rows = [
        ("merge_sort", "Θ(n log n)", "Θ(n log n)", "Θ(n log n)", "Θ(n)", "да",
         "нужен вспомогательный массив на слияние"),
        ("counting_sort", "Θ(n + k)", "Θ(n + k)", "Θ(n + k)", "Θ(n + k)", "да",
         "целые ключи из [0, k]; при k ≫ n память и время растут как k"),
        ("radix_sort_lsd", "Θ(d·n)", "Θ(d·n)", "Θ(d·n)", "Θ(n)", "да",
         "неотрицательные целые; d — число разрядов максимума"),
        ("heap_sort", "Θ(n log n)", "Θ(n log n)", "Θ(n log n)", "Θ(1)", "нет",
         "in-place, но нестабилен и хуже по локальности памяти"),
        ("quick_sort (ЛР 3)", "Θ(n log n)", "Θ(n log n)", "Θ(n²)", "O(log n)",
         "нет", "худший случай маловероятен только при рандомизации пивота"),
        ("sorted() (эталон)", "Θ(n)", "Θ(n log n)", "Θ(n log n)", "Θ(n)", "да",
         "Timsort: использует упорядоченные подотрезки входа"),
    ]
    for name, best, average, worst, memory, stable, limits in rows:
        print(f"| `{name}` | {best} | {average} | {worst} | {memory} | {stable} "
              f"| {limits} |")
    print("\nНижняя граница Ω(n log n) относится к сортировкам сравнением; "
          "counting и radix её обходят, потому что используют значение ключа "
          "как адрес, а не сравнивают ключи друг с другом.\n")


def print_time_tables(results) -> None:
    """Таблицы времени по размерам и распределениям + наклоны log-log."""
    for n in SIZES:
        print(f"\n### n = {n:,}\n".replace(",", " "))
        print("| Алгоритм | random_wide | random_small | sorted | reversed | almost_sorted |")
        print("| --- | ---: | ---: | ---: | ---: | ---: |")
        for name in ALGORITHM_ORDER:
            row = []
            for dist in DISTRIBUTIONS:
                points = results[name][dist]
                if not points:
                    row.append("—")
                    continue
                value = next(t for m, t in points if m == n)
                row.append(f"{value:.4f}")
            print(f"| `{name}` | " + " | ".join(row) + " |")

    print("\n### Наклоны log-log\n")
    print("| Алгоритм | random_wide | random_small | sorted | reversed | almost_sorted |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name in ALGORITHM_ORDER:
        row = []
        for dist in DISTRIBUTIONS:
            points = results[name][dist]
            row.append("—" if not points else f"{log_log_slope(points):.2f}")
        print(f"| `{name}` | " + " | ".join(row) + " |")


def measure_counting_range(rng: random.Random) -> list[tuple[int, float, int]]:
    """Цена диапазона ключей: время и пиковая память CountingSort как функции k.

    n фиксировано, меняется только k = max ключа. Ожидание: время Θ(n + k) и
    память Θ(n + k) — обе величины растут линейно по k, пока k не станет
    сравним с n и больше.

    Время и память меряются в разных прогонах: tracemalloc замедляет работу в
    несколько раз и исказил бы замер времени.
    """
    data = [rng.randint(0, RANGE_SIZES[-1]) for _ in range(RANGE_N)]
    rows = []
    for k in RANGE_SIZES:
        subset = [x % (k + 1) for x in data]

        times = []
        out = None
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            out = counting_sort(subset, key_max=k)
            times.append(time.perf_counter() - t0)
        assert out == sorted(subset), "counting_sort: расхождение в опыте с k"

        tracemalloc.start()
        counting_sort(subset, key_max=k)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rows.append((k, statistics.median(times), peak))
    return rows


def print_range_table(rows) -> None:
    """Таблица «k -> время, память» и сравнение с radix на тех же данных."""
    print("\n## CountingSort: цена диапазона ключей\n")
    print(f"n = {_num(RANGE_N)}, значения в [0, k]; пиковая память — tracemalloc, "
          f"время — медиана по {REPEATS} прогонам без tracemalloc.\n")
    print("| k | время, с | пиковая память, МБ | время/элемент, нс |")
    print("| ---: | ---: | ---: | ---: |")
    for k, elapsed, peak in rows:
        print(f"| {_num(k)} | {elapsed:.4f} | {peak / 2 ** 20:.2f} | "
              f"{elapsed / RANGE_N * 1e9:.1f} |")
    print("\nСравнение с RadixSort на том же входе (k = 10⁷, n = 100 000):")
    rng = random.Random(7)
    data = [rng.randint(0, 10_000_000) for _ in range(RANGE_N)]
    times = []
    out = None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        out = radix_sort_lsd(data)
        times.append(time.perf_counter() - t0)
    assert out == sorted(data), "radix_sort_lsd: расхождение в опыте с k"
    tracemalloc.start()
    radix_sort_lsd(data)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  radix_sort_lsd: {statistics.median(times):.4f} c, "
          f"пиковая память {peak / 2 ** 20:.2f} МБ")
    print(f"  counting_sort при том же k: {rows[-1][1]:.4f} c, "
          f"{rows[-1][2] / 2 ** 20:.2f} МБ")


def _num(value: int) -> str:
    """Число с пробелами между разрядами для читаемости таблиц (12 345)."""
    return f"{value:,}".replace(",", " ")


def plot_results(results, range_rows, out_dir: Path) -> None:
    """Log-log графики по распределениям + график цены диапазона ключей."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("matplotlib не установлен — графики пропущены.")
        return

    styles = {
        "merge_sort": ("o", "#264653"),
        "counting_sort": ("v", "#2a9d8f"),
        "radix_sort_lsd": ("D", "#e9c46a"),
        "heap_sort": ("s", "#f4a261"),
        "quick_sort (ЛР 3)": ("^", "#e76f51"),
        "sorted() (эталон)": ("x", "grey"),
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True, sharey=True)
    for ax, dist in zip(axes.flat, DISTRIBUTIONS):
        for name in ALGORITHM_ORDER:
            points = results[name][dist]
            if not points:
                continue
            marker, color = styles[name]
            ax.plot([n for n, _ in points], [t for _, t in points], marker=marker,
                    markersize=3.5, linewidth=1.1, color=color,
                    label=f"{name} ({log_log_slope(points):.2f})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(dist)
        ax.grid(True, which="both", linewidth=0.25)
    for ax in axes[1]:
        ax.set_xlabel("размер входа n")
    for ax in axes[:, 0]:
        ax.set_ylabel("время, с")

    # Легенда строится по всем алгоритмам, а не только по первой панели:
    # counting_sort отсутствует на панели random_wide, но должен быть в подписи.
    legend_ax = axes.flat[-1]
    legend_ax.axis("off")
    handles = [Line2D([], [], marker=styles[name][0], color=styles[name][1],
                      linewidth=1.2, markersize=5, label=name)
               for name in ALGORITHM_ORDER]
    legend_ax.legend(handles=handles, fontsize=9, loc="center", title="алгоритм")
    legend_ax.text(0.5, 0.08, "наклоны log-log — в таблице отчёта",
                   transform=legend_ax.transAxes, ha="center", fontsize=8, color="dimgray")
    fig.suptitle("Сортировки: время в осях log-log по распределениям данных")
    fig.tight_layout()
    fig.savefig(out_dir / "lab04_loglog.png", dpi=150)
    plt.close(fig)

    # Цена диапазона ключей: время и память как функции k
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ks = [row[0] for row in range_rows]
    axes[0].plot(ks, [row[1] for row in range_rows], marker="o", color="#264653")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("k = максимальный ключ")
    axes[0].set_ylabel("время, с")
    axes[0].set_title("CountingSort: время от k (n = 100 000)")
    axes[0].grid(True, which="both", linewidth=0.3)

    axes[1].plot(ks, [row[2] / 2 ** 20 for row in range_rows], marker="s", color="#e76f51")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("k = максимальный ключ")
    axes[1].set_ylabel("пиковая память, МБ")
    axes[1].set_title("CountingSort: память от k (n = 100 000)")
    axes[1].grid(True, which="both", linewidth=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "lab04_counting_range.png", dpi=150)
    plt.close(fig)

    print(f"\nГрафики сохранены в {out_dir}")


def run(seed: int, out_dir: Path) -> None:
    """Полный прогон: замеры, таблицы, графики."""
    print("\n## Параметры\n")
    print(f"- seed = 30 + вариант = {seed}")
    print(f"- размеры: {', '.join(_num(n) for n in SIZES)}")
    print(f"- распределения: {', '.join(DISTRIBUTIONS)}")
    print(f"- повторов на точку: {REPEATS}, медиана, прогрев перед серией")
    print(f"- узкий диапазон ключей: [0, {SMALL_KEY_MAX}], "
          f"широкий: [0, {_num(WIDE_KEY_MAX)}]")

    results = collect_times(seed)
    print_time_tables(results)
    print_properties_table()
    rows = measure_counting_range(random.Random(seed))
    print_range_table(rows)
    plot_results(results, rows, out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", type=int, required=True, help="номер варианта (seed = 30 + N)")
    ap.add_argument("--out", type=Path, default=Path.cwd(), help="каталог для графиков")
    args = ap.parse_args()

    seed = 30 + args.variant
    random.seed(seed)
    args.out.mkdir(parents=True, exist_ok=True)

    self_check()
    run(seed, args.out)


if __name__ == "__main__":
    sys.setrecursionlimit(10_000)
    main()