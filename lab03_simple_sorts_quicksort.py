#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ЛР 3. Простые сортировки и QuickSort с рандомизацией опорного элемента (решение).

КИМ-03, вариант задаётся флагом ``--variant``; seed генератора — ``30 + вариант``
(см. ``docs/reproducibility.md``). Замеры — по методике того же документа:
``time.perf_counter``, прогрев, 5 повторов, медиана.

Запуск:
    python lab03_simple_sorts_quicksort.py --variant 15
    python lab03_simple_sorts_quicksort.py --variant 15 --out .

Соглашение о счётчиках (одно и то же во всех таблицах отчёта):

* ``comparisons`` — число сравнений ключей (сравнение вида ``key(a) < key(b)``);
* ``moves`` — число обменов/перемещений: у пузырька и выбора это один обмен пары,
  у вставок — один сдвиг элемента на соседнюю позицию;
* «перестановка ключа на своё место» в счётчик не входит: на упорядоченном входе
  вставки обязаны иметь ноль перемещений (лучший случай Θ(n) по сравнениям).
"""
from __future__ import annotations

import argparse
import math
import random
import statistics
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Параметры эксперимента (docs/reproducibility.md, п. 4)
# ---------------------------------------------------------------------------

REPEATS = 5                      # повторов на точку, берётся медиана
SIZES_QUADRATIC = (500, 1_000, 2_000, 4_000, 8_000)
SIZES_QUICK = (1_000, 3_000, 10_000, 30_000, 100_000)
SIZES_PIVOT = (1_000, 2_000, 4_000)   # опыт «детерминированный vs случайный пивот»
QUICK_SMALL = 16                 # подмассивы короче — сортируются вставками

CLASSES = ("упорядоченный", "случайный", "обратный")


def _identity(value):
    """Ключ по умолчанию: сортировка по самому элементу."""
    return value


# ---------------------------------------------------------------------------
# 1. Простые сортировки со счётчиками (сортируют копию, вход не меняют)
# ---------------------------------------------------------------------------


def bubble_sort(values: list, key=_identity) -> tuple[list, int, int]:
    """Сортировка пузырьком с досрочным выходом.

    Сложность: худший и средний случаи Θ(n²) сравнений и обменов; лучший —
    Θ(n) сравнений и 0 обменов (один проход без обмена завершает работу).
    """
    a = list(values)
    comparisons = moves = 0
    for end in range(len(a) - 1, 0, -1):
        swapped = False
        for i in range(end):
            comparisons += 1
            if key(a[i]) > key(a[i + 1]):
                a[i], a[i + 1] = a[i + 1], a[i]
                moves += 1
                swapped = True
        if not swapped:
            break
    return a, comparisons, moves


def insertion_sort(values: list, key=_identity) -> tuple[list, int, int]:
    """Сортировка вставками.

    Худший и средний случаи Θ(n²); лучший (упорядоченный вход) — Θ(n) сравнений
    и 0 сдвигов. Именно последнее отличает её от сортировки выбором.
    """
    a = list(values)
    comparisons = moves = 0
    for i in range(1, len(a)):
        current = a[i]
        j = i - 1
        while j >= 0:
            comparisons += 1
            if key(a[j]) <= key(current):
                break
            a[j + 1] = a[j]
            moves += 1
            j -= 1
        a[j + 1] = current
    return a, comparisons, moves


def selection_sort(values: list, key=_identity) -> tuple[list, int, int]:
    """Сортировка выбором.

    Ровно n(n-1)/2 сравнений на любом входе — Θ(n²) независимо от данных;
    обменов не более n-1, поэтому по перемещениям это самый «экономный» из
    простых алгоритмов. Нестабильна: обмен минимума с a[i] переносит элемент
    через равные ему ключи.
    """
    a = list(values)
    comparisons = moves = 0
    n = len(a)
    for i in range(n - 1):
        best = i
        for j in range(i + 1, n):
            comparisons += 1
            if key(a[j]) < key(a[best]):
                best = j
        if best != i:
            a[i], a[best] = a[best], a[i]
            moves += 1
    return a, comparisons, moves


# ---------------------------------------------------------------------------
# 2. QuickSort с рандомизированным опорным элементом
# ---------------------------------------------------------------------------


def _partition_three_way(a: list, lo: int, hi: int, pivot_index: int, key,
                         counters: list[int]) -> tuple[int, int]:
    """Трёхпутевое разбиение вокруг элемента pivot_index (на месте).

    Возвращает границы блока элементов, равных опорному: ``[lt, gt]``.
    Элементы ``a[lo:lt]`` строго меньше опорного, ``a[gt+1:hi+1]`` — строго
    больше. Трёхпутевая схема не деградирует на входах с большим числом
    дубликатов (в том числе «все элементы равны»): такой блок отсекается
    за один проход, а не за n.
    """
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
            counters[0] += 2       # проверены оба условия: < и >
            if key(a[i]) > key(pivot):
                a[i], a[gt] = a[gt], a[i]
                counters[1] += 1
                gt -= 1
            else:
                i += 1
    # опорный элемент — на границу блока равных
    a[gt + 1], a[hi] = a[hi], a[gt + 1]
    counters[1] += 1
    return lt, gt + 1


def _insertion_range(a: list, lo: int, hi: int, key, counters: list[int]) -> None:
    """Вставки на подмассиве a[lo:hi+1] — «хвост» QuickSort для малых отрезков."""
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


def _quick_sort_impl(values: list, pick_pivot, key, counters: list[int]) -> list:
    """Общий каркас QuickSort: итеративный, со стеком отрезков.

    Обрабатывается меньший отрезок, больший кладётся в стек, поэтому глубина
    стека — O(log n) даже при неудачных разбиениях (рекурсия не используется,
    переполнение стека вызовов невозможно). Отрезки короче QUICK_SMALL
    сортируются вставками: на малых n они быстрее и снимают основную часть
    рекурсивных вызовов.
    """
    a = list(values)
    stack = [(0, len(a) - 1)]
    while stack:
        lo, hi = stack.pop()
        while hi - lo + 1 > QUICK_SMALL:
            lt, gt = _partition_three_way(a, lo, hi, pick_pivot(lo, hi), key, counters)
            if lt - lo < hi - gt:            # меньший отрезок — сразу в работу
                stack.append((gt + 1, hi))
                hi = lt - 1
            else:
                stack.append((lo, lt - 1))
                lo = gt + 1
        if hi > lo:
            _insertion_range(a, lo, hi, key, counters)
    return a


def quick_sort(values: list, rng: random.Random | None = None, key=_identity,
               counters: list[int] | None = None) -> list:
    """QuickSort с рандомизированным опорным элементом (``rng.randrange``).

    Среднее время Θ(n log n), худшее Θ(n²); рандомизация делает худший случай
    маловероятным для любого фиксированного входа. Дополнительная память —
    O(log n) под стек отрезков.

    ``counters`` — необязательный список ``[сравнения, перемещения]``, который
    заполняется по ходу работы (нужен для таблиц отчёта).
    """
    if rng is None:
        rng = random.Random()
    if counters is None:
        counters = [0, 0]
    return _quick_sort_impl(values, lambda lo, hi: rng.randrange(lo, hi + 1),
                            key, counters)


def quick_sort_deterministic(values: list, key=_identity,
                             counters: list[int] | None = None) -> list:
    """QuickSort с фиксированным опорным элементом (последний в отрезке).

    Нужен только для эксперимента: показывает, на каком входе детерминированный
    выбор пивота даёт Θ(n²), и как рандомизация это снимает.
    """
    if counters is None:
        counters = [0, 0]
    return _quick_sort_impl(values, lambda lo, hi: hi, key, counters)


# ---------------------------------------------------------------------------
# 3. Демонстрация стабильности на парах (ключ, метка)
# ---------------------------------------------------------------------------

STABILITY_PAIRS = [
    (key, label)
    for key, label in zip(
        (2, 1, 2, 1, 3, 2, 1, 2, 4, 3, 1, 2, 3, 4, 2, 1, 3, 2, 4, 1, 2, 3, 1, 2),
        "abcdefghijklmnopqrstuvwx",
    )
]


def _labels(pairs: list) -> str:
    """Метки пар в текущем порядке — по ним видно, сохранён ли исходный порядок."""
    return " ".join(f"{p[1]}({p[0]})" for p in pairs)


def stability_demo() -> None:
    """Показать, какие сортировки сохраняют порядок равных ключей, а какие нет."""
    print("\n## Стабильность на парах (ключ, метка)\n")
    print(f"Вход: {_labels(STABILITY_PAIRS)}")
    reference = sorted(STABILITY_PAIRS, key=lambda p: p[0])
    print(f"Эталон (устойчивый sorted): {_labels(reference)}\n")

    runs = [
        ("bubble_sort", lambda a: bubble_sort(a, key=lambda p: p[0])[0]),
        ("insertion_sort", lambda a: insertion_sort(a, key=lambda p: p[0])[0]),
        ("selection_sort", lambda a: selection_sort(a, key=lambda p: p[0])[0]),
        ("quick_sort", lambda a: quick_sort(a, key=lambda p: p[0],
                                           rng=random.Random(0))),
    ]
    print("| Алгоритм | Результат | Стабильна |")
    print("| --- | --- | :---: |")
    for name, fn in runs:
        got = fn(list(STABILITY_PAIRS))
        stable = "да" if got == reference else "**нет**"
        print(f"| `{name}` | {_labels(got)} | {stable} |")

    # Быстрая проверка на 200 случайных наборах пар: стабильность — инвариант,
    # а не единичное наблюдение.
    rng = random.Random(0)
    verdict = {}
    for name, fn in runs[:2]:
        ok = True
        for _ in range(200):
            pairs = [(rng.randint(0, 3), i) for i in range(rng.randint(0, 30))]
            ok &= fn(list(pairs)) == sorted(pairs, key=lambda p: p[0])
        verdict[name] = ok
    print("\nСтабильность как инвариант (200 случайных наборов пар, "
          "сверка с `sorted(key=...)`):")
    for name, ok in verdict.items():
        print(f"  {name}: {'OK' if ok else 'НАРУШЕНА'}")


# ---------------------------------------------------------------------------
# 4. Классы входов (данные по варианту, детерминированно по seed)
# ---------------------------------------------------------------------------


def make_inputs(n: int, seed: int) -> dict[str, list[int]]:
    """Три класса входов размера n: упорядоченный, случайный, обратный."""
    rng = random.Random(seed + n)  # свой поток для каждого размера
    data = [rng.randint(-1_000_000, 1_000_000) for _ in range(n)]
    return {
        "упорядоченный": sorted(data),
        "случайный": data,
        "обратный": sorted(data, reverse=True),
    }


# ---------------------------------------------------------------------------
# 5. Верификация (docs/ai-verification.md, шаги 1–3)
# ---------------------------------------------------------------------------


class SelfCheckError(Exception):
    """Нарушен контракт или инвариант реализации."""


def expect(condition: bool, message: str) -> None:
    """Явная проверка вместо assert: assert-ы исчезают при запуске python -O."""
    if not condition:
        raise SelfCheckError(message)


def is_sorted(a: list, key=_identity) -> bool:
    """Инвариант 1: неубывающий порядок ключей."""
    return all(key(a[i]) <= key(a[i + 1]) for i in range(len(a) - 1))


def check_algorithm(fn) -> None:
    """Граничные случаи, инварианты сортировки и сверка с эталоном sorted()."""
    boundary = [
        [],                    # пустой массив
        [7],                   # один элемент
        [2, 1],                # два элемента
        [5, 5, 5, 5],          # все элементы равны
        [1, 2, 3, 4, 5],       # уже отсортирован
        [5, 4, 3, 2, 1],       # обратный порядок
        [-3, 0, -3, 7, -3],    # отрицательные числа и дубликаты
    ]
    for a in boundary:
        res = fn(list(a))
        expect(isinstance(res, list), f"на {a} возвращён {type(res).__name__}, а не список")
        expect(is_sorted(res), f"нарушен порядок на {a}: получено {res}")
        expect(sorted(res) == sorted(a), f"не перестановка входа {a}: получено {res}")
        expect(res == sorted(a), f"расходится с sorted() на {a}: получено {res}")

    # Сверка с эталоном на сотнях случайных входов (шаг 3 методики).
    rng = random.Random(0)
    for _ in range(300):
        a = [rng.randint(-100, 100) for _ in range(rng.randint(0, 80))]
        res = fn(list(a))
        expect(res == sorted(a), f"расходится с sorted() на {a}: получено {res}")


def check_key_variant(fn) -> None:
    """Инвариант с ключом: результат равен sorted(a, key=...) по ключам."""
    rng = random.Random(1)
    for _ in range(100):
        a = [(rng.randint(0, 9), i) for i in range(rng.randint(0, 40))]
        res = fn(list(a))
        expect([p[0] for p in res] == sorted(p[0] for p in a),
               f"неверный порядок ключей на {a}: получено {res}")
        expect(is_sorted(res, key=lambda p: p[0]), f"ключи не упорядочены на {a}")


def check_counters() -> None:
    """Счётчики согласуются с теорией о числе сравнений и перемещений."""
    n = 5
    _, comparisons, _ = selection_sort([3, 1, 2, 5, 4])
    expect(comparisons == n * (n - 1) // 2,
           f"selection_sort: сравнений {comparisons}, ожидалось {n * (n - 1) // 2}")

    # Выбор: ровно n(n-1)/2 сравнений на любом входе.
    for values in ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5], [3, 3, 3, 3, 3]):
        _, comparisons, _ = selection_sort(values)
        expect(comparisons == n * (n - 1) // 2,
               f"selection_sort: на {values} сравнений {comparisons}")

    # Пузырёк на упорядоченном входе: ровно n-1 сравнений, 0 обменов (досрочный выход).
    ordered = list(range(20))
    _, comparisons, moves = bubble_sort(ordered)
    expect((comparisons, moves) == (19, 0),
           f"bubble_sort на упорядоченном: сравнений {comparisons}, обменов {moves}")

    # Пузырёк на обратном входе: n(n-1)/2 обменов.
    n = 20
    _, _, moves = bubble_sort(list(range(n, 0, -1)))
    expect(moves == n * (n - 1) // 2,
           f"bubble_sort на обратном: обменов {moves}, ожидалось {n * (n - 1) // 2}")

    # Вставки: лучший случай — n-1 сравнений и 0 сдвигов; худший — n(n-1)/2 сравнений.
    _, comparisons, moves = insertion_sort(ordered)
    expect((comparisons, moves) == (n - 1, 0),
           f"insertion_sort на упорядоченном: сравнений {comparisons}, сдвигов {moves}")
    _, comparisons, _ = insertion_sort(list(range(n, 0, -1)))
    expect(comparisons == n * (n - 1) // 2,
           f"insertion_sort на обратном: сравнений {comparisons}")

    # QuickSort: на упорядоченном входе с рандомизацией — O(n log n), не n².
    n = 2_048
    counters = [0, 0]
    quick_sort(list(range(n)), rng=random.Random(0), counters=counters)
    limit = 4 * n * math.log2(n)
    expect(counters[0] < limit,
           f"quick_sort: сравнений {counters[0]} при пороге {limit:.0f} — похоже на Θ(n²)")


def check_quick_randomization() -> None:
    """Разные seed дают одинаковый результат; детерминированный пивот бьётся о n²."""
    # Поток случайных чисел создаётся один раз: random.Random(5) внутри
    # генератора списка каждый раз возвращал бы одно и то же первое число,
    # и проверка шла бы на массиве из одинаковых элементов.
    data_rng = random.Random(5)
    a = [data_rng.randint(-50, 50) for _ in range(500)]
    expect(quick_sort(a, random.Random(1)) == sorted(a), "quick_sort: расхождение при rng=1")
    expect(quick_sort(a, random.Random(2)) == sorted(a), "quick_sort: расхождение при rng=2")

    # Худший случай детерминированного пивота: упорядоченный вход.
    n = 1_024
    det = [0, 0]
    rnd = [0, 0]
    quick_sort_deterministic(list(range(n)), counters=det)
    quick_sort(list(range(n)), rng=random.Random(0), counters=rnd)
    expect(det[0] > rnd[0],
           f"детерминированный пивот на упорядоченном входе не деградировал: "
           f"{det[0]} против {rnd[0]}")


ALGORITHMS = {
    "bubble_sort": lambda a: bubble_sort(a)[0],
    "insertion_sort": lambda a: insertion_sort(a)[0],
    "selection_sort": lambda a: selection_sort(a)[0],
    "quick_sort": lambda a: quick_sort(a, random.Random(0)),
}


def self_check() -> None:
    """Все проверки по разделам; при любой ошибке замеры не выполняются."""
    # fn=fn фиксирует функцию текущего шага цикла (иначе все лямбды увидят последнюю)
    checks = [(name, lambda fn=fn: check_algorithm(fn)) for name, fn in ALGORITHMS.items()]
    checks += [(f"{name} (ключ)", lambda fn=fn: check_key_variant(fn))
               for name, fn in ALGORITHMS.items()]
    checks.append(("счётчики и их согласие с теорией", check_counters))
    checks.append(("рандомизация пивота", check_quick_randomization))
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
# 6. Бенчмарк (методика — docs/reproducibility.md)
# ---------------------------------------------------------------------------


def bench(fn, data: list) -> float:
    """Медиана времени выполнения fn(копия data) по REPEATS запускам, с прогревом."""
    fn(list(data))  # прогрев — не учитывается
    times = []
    for _ in range(REPEATS):
        arg = list(data)  # свежая копия: сортировка не должна получать свой результат
        t0 = time.perf_counter()
        fn(arg)
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


def collect_times(seed: int) -> dict[tuple[str, str], list[tuple[int, float]]]:
    """Замеры простых сортировок: ключ (алгоритм, класс входа) -> [(n, медиана)].

    QuickSort меряется отдельно, на своём ряде размеров (см. collect_quick).
    """
    results: dict[tuple[str, str], list[tuple[int, float]]] = {}
    plans = (
        ("bubble_sort", ALGORITHMS["bubble_sort"]),
        ("insertion_sort", ALGORITHMS["insertion_sort"]),
        ("selection_sort", ALGORITHMS["selection_sort"]),
    )
    for name, fn in plans:
        for cls in CLASSES:
            points = []
            for n in SIZES_QUADRATIC:
                points.append((n, bench(fn, make_inputs(n, seed)[cls])))
            results[(name, cls)] = points
    return results


def collect_quick(seed: int) -> tuple[dict[str, list[tuple[int, float]]],
                                     list[tuple[int, float, float]]]:
    """Замеры QuickSort по классам входов и опыт с детерминированным пивотом.

    Возвращает (ряды по классам, [(n, случайный пивот, фиксированный пивот)])
    для упорядоченного входа. Опыт с фиксированным пивотом ведётся на коротком
    ряде размеров: на упорядоченном входе он квадратичен, и размеры 10⁴–10⁵
    дали бы часы счёта.
    """
    results: dict[str, list[tuple[int, float]]] = {cls: [] for cls in CLASSES}
    for cls in CLASSES:
        for n in SIZES_QUICK:
            data = make_inputs(n, seed)[cls]
            results[cls].append((n, bench(lambda a: quick_sort(a, random.Random(seed)), data)))

    pivot_demo = []
    for n in SIZES_PIVOT:
        data = make_inputs(n, seed)["упорядоченный"]
        t_rand = bench(lambda a: quick_sort(a, random.Random(seed)), data)
        t_det = bench(lambda a: quick_sort_deterministic(a), data)
        pivot_demo.append((n, t_rand, t_det))
    return results, pivot_demo


def _num(value: int) -> str:
    """Число с пробелами между разрядами для читаемости таблиц (12 345)."""
    return f"{value:,}".replace(",", " ")


def print_counter_table() -> None:
    """Таблица счётчиков сравнений и перемещений (без замеров времени)."""
    print("\n## Счётчики операций\n")
    for title, index in (("Сравнения ключей", 1), ("Обмены / перемещения", 2)):
        print(f"### {title}\n")
        print("| n | вход | bubble | insertion | selection |")
        print("| ---: | --- | ---: | ---: | ---: |")
        for n in SIZES_QUADRATIC:
            for cls, data in make_inputs(n, SEED).items():
                b = bubble_sort(data)[index]
                i = insertion_sort(data)[index]
                s = selection_sort(data)[index]
                print(f"| {_num(n)} | {cls} | {_num(b)} | {_num(i)} | {_num(s)} |")
        print()

    print("\nТеоретические значения: выбор — n(n-1)/2 сравнений на любом входе; "
          "пузырёк и вставки на обратном входе — n(n-1)/2 сравнений, "
          "у вставок лучший случай n-1 сравнений и 0 сдвигов, "
          "у пузырька на упорядоченном входе n-1 сравнение и 0 обменов.\n")


def print_time_tables(results, quick, pivot_demo) -> None:
    """Таблицы времени и наклонов log-log."""
    print("\n## Время: простые сортировки\n")
    for name in ("bubble_sort", "insertion_sort", "selection_sort"):
        print(f"### `{name}`\n")
        print("| n | упорядоченный, с | случайный, с | обратный, с |")
        print("| ---: | ---: | ---: | ---: |")
        for idx, (n, _) in enumerate(results[(name, "случайный")]):
            cells = " | ".join(f"{results[(name, cls)][idx][1]:.4f}" for cls in CLASSES)
            print(f"| {_num(n)} | {cells} |")
        print()

    print("\n### Наклоны log-log (ряд размеров простых сортировок)\n")
    print("| Алгоритм | упорядоченный | случайный | обратный | теория (худший) |")
    print("| --- | ---: | ---: | ---: | :---: |")
    for name in ("bubble_sort", "insertion_sort", "selection_sort"):
        slopes = [log_log_slope(results[(name, cls)]) for cls in CLASSES]
        print(f"| `{name}` | {slopes[0]:.2f} | {slopes[1]:.2f} | {slopes[2]:.2f} | 2 |")

    print("\n## Время: QuickSort\n")
    print("| n | упорядоченный, с | случайный, с | обратный, с | сравнений (случайный) |")
    print("| ---: | ---: | ---: | ---: | ---: |")
    for idx, (n, _) in enumerate(quick["случайный"]):
        counters = [0, 0]
        quick_sort(make_inputs(n, SEED)["случайный"], rng=random.Random(SEED),
                   counters=counters)
        print(f"| {_num(n)} | {quick['упорядоченный'][idx][1]:.4f} | "
              f"{quick['случайный'][idx][1]:.4f} | {quick['обратный'][idx][1]:.4f} | "
              f"{_num(counters[0])} |")

    print("\n### QuickSort: рандомизация пивота на упорядоченном входе\n")
    print("| n | случайный пивот, с | фиксированный пивот, с | замедление |")
    print("| ---: | ---: | ---: | ---: |")
    for n, t_rand, t_det in pivot_demo:
        print(f"| {_num(n)} | {t_rand:.4f} | {t_det:.4f} | ×{t_det / t_rand:.1f} |")

    print("\n### Наклоны log-log (QuickSort)\n")
    print("| Ряд | наклон | теория |")
    print("| --- | ---: | :---: |")
    for cls in CLASSES:
        print(f"| {cls} | {log_log_slope(quick[cls]):.2f} | ≈1 (n log n) |")
    det_points = [(n, t_det) for n, _, t_det in pivot_demo]
    print(f"| фиксированный пивот на упорядоченном | "
          f"{log_log_slope(det_points):.2f} | 2 (n²) |")


def plot_results(results, quick, pivot_demo, out_dir: Path) -> None:
    """Log-log графики: простые сортировки, QuickSort, счётчики сравнений."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib не установлен — графики пропущены.")
        return

    markers = {"bubble_sort": "o", "insertion_sort": "s", "selection_sort": "^"}
    colors = {"упорядоченный": "#2a9d8f", "случайный": "#264653", "обратный": "#e76f51"}

    # --- Рисунок 1: простые сортировки, три класса входов -------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name in ("bubble_sort", "insertion_sort", "selection_sort"):
        for cls in CLASSES:
            points = results[(name, cls)]
            ax.plot([n for n, _ in points], [t for _, t in points],
                    marker=markers[name], markersize=4, linewidth=1.2,
                    linestyle="--" if cls == "обратный" else "-",
                    color=colors[cls], label=f"{name}, {cls} ({log_log_slope(points):.2f})")
    guide = results[("bubble_sort", "случайный")]
    n0 = guide[0][0]
    t0 = guide[0][1]
    ax.plot([n for n, _ in guide], [t0 * (n / n0) ** 2 for n, _ in guide],
            color="grey", linewidth=0.8, linestyle=":", label="наклон 2 (n²)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("размер входа n")
    ax.set_ylabel("время, с")
    ax.set_title("Простые сортировки: время в осях log-log")
    ax.grid(True, which="both", linewidth=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "lab03_simple_sorts_loglog.png", dpi=150)
    plt.close(fig)

    # --- Рисунок 2: QuickSort, включая детерминированный пивот --------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for cls in CLASSES:
        points = quick[cls]
        ax.plot([n for n, _ in points], [t for _, t in points],
                marker="o", markersize=4, color=colors[cls], label=f"quick_sort, {cls}")
    det_points = [(n, t_det) for n, _, t_det in pivot_demo]
    ax.plot([n for n, _ in det_points], [t for _, t in det_points], marker="x",
            color="#8d0801", linestyle="--",
            label=f"фиксированный пивот, упорядоченный ({log_log_slope(det_points):.2f})")
    ref = quick["случайный"]
    ax.plot([n for n, _ in det_points],
            [ref[0][1] * (n / ref[0][0]) * (math.log2(n) / math.log2(ref[0][0]))
             for n, _ in det_points], color="grey", linewidth=0.8, linestyle=":",
            label="наклон ≈1 (n log n)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("размер входа n")
    ax.set_ylabel("время, с")
    ax.set_title("QuickSort: рандомизация пивота против фиксированного")
    ax.grid(True, which="both", linewidth=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "lab03_quicksort_loglog.png", dpi=150)
    plt.close(fig)

    # --- Рисунок 3: счётчики сравнений --------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name, sort_fn in (("bubble_sort", bubble_sort),
                          ("insertion_sort", insertion_sort),
                          ("selection_sort", selection_sort)):
        for cls in CLASSES:
            points = [(n, sort_fn(make_inputs(n, SEED)[cls])[1])
                      for n in SIZES_QUADRATIC]
            ax.plot([n for n, _ in points], [c for _, c in points],
                    marker=markers[name], markersize=4, linewidth=1.1,
                    linestyle="--" if cls == "обратный" else "-",
                    color=colors[cls], label=f"{name}, {cls}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("размер входа n")
    ax.set_ylabel("число сравнений ключей")
    ax.set_title("Счётчики сравнений: совпадение с n² и вырождение на упорядоченном входе")
    ax.grid(True, which="both", linewidth=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "lab03_counters_loglog.png", dpi=150)
    plt.close(fig)

    print(f"\nГрафики сохранены в {out_dir}")


SEED = 0  # переопределяется в main() из --variant


def run(seed: int, out_dir: Path) -> None:
    """Полный прогон: замеры, таблицы, графики."""
    print("\n## Параметры\n")
    print(f"- seed = 30 + вариант = {seed}")
    print(f"- размеры (простые сортировки): {', '.join(str(n) for n in SIZES_QUADRATIC)}")
    print(f"- размеры (QuickSort): {', '.join(str(n) for n in SIZES_QUICK)}")
    print(f"- повторов на точку: {REPEATS}, медиана, прогрев перед серией")

    results = collect_times(seed)
    quick, pivot_demo = collect_quick(seed)
    print_time_tables(results, quick, pivot_demo)
    plot_results(results, quick, pivot_demo, out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", type=int, required=True, help="номер варианта (seed = 30 + N)")
    ap.add_argument("--out", type=Path, default=Path.cwd(), help="каталог для графиков")
    args = ap.parse_args()

    seed = 30 + args.variant
    random.seed(seed)
    global SEED
    SEED = seed
    args.out.mkdir(parents=True, exist_ok=True)

    self_check()
    stability_demo()
    print_counter_table()
    run(seed, args.out)


if __name__ == "__main__":
    main()