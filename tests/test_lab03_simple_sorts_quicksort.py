#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Тесты решения ЛР 3 «Простые сортировки и QuickSort».

Проверяется то, что требует рубрика-03 и методика docs/ai-verification.md:

1) корректность четырёх сортировок на граничных и сотнях случайных входов;
2) сортировка не меняет входной список (алгоритмы работают с копией);
3) счётчики сравнений и перемещений совпадают с теоремой;
4) QuickSort рандомизирован, воспроизводим при фиксированном seed и не
   деградирует до Θ(n²) на упорядоченном входе и на дубликатах;
5) устойчивость: пузырёк и вставки стабильны, выбор — нет;
6) встроенная самопроверка проходит, а полный прогон пишет графики.

Модуль решения лежит рядом с заготовкой и содержит дефис в имени, поэтому
загружается через importlib, а не обычным ``import``.
"""
from __future__ import annotations

import importlib.util
import math
import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LAB_PATH = REPO_ROOT / "lab03_simple_sorts_quicksort.py"

BOUNDARY_INPUTS = [
    [],
    [7],
    [2, 1],
    [1, 2],
    [5, 5, 5, 5],
    [1, 2, 3, 4, 5],
    [5, 4, 3, 2, 1],
    [-3, 0, -3, 7, -3],
]


def _load_lab():
    """Загрузить решение как модуль Python, несмотря на дефис в имени файла."""
    assert LAB_PATH.is_file(), f"нет файла решения: {LAB_PATH}"
    spec = importlib.util.spec_from_file_location("lab03_solution", LAB_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


lab = _load_lab()

# Функции, возвращающие только отсортированный список.
SORTS = {
    "bubble_sort": lambda a: lab.bubble_sort(a)[0],
    "insertion_sort": lambda a: lab.insertion_sort(a)[0],
    "selection_sort": lambda a: lab.selection_sort(a)[0],
    "quick_sort": lambda a: lab.quick_sort(a, rng=random.Random(0)),
}

KEYED_SORTS = {
    "bubble_sort": lambda a: lab.bubble_sort(a, key=lambda p: p[0])[0],
    "insertion_sort": lambda a: lab.insertion_sort(a, key=lambda p: p[0])[0],
    "selection_sort": lambda a: lab.selection_sort(a, key=lambda p: p[0])[0],
    "quick_sort": lambda a: lab.quick_sort(a, key=lambda p: p[0],
                                           rng=random.Random(0)),
}


@pytest.mark.parametrize("name", sorted(SORTS))
def test_boundary_inputs(name: str) -> None:
    """Пустой массив, один элемент, дубликаты, крайние порядки — всё корректно."""
    sort = SORTS[name]
    for data in BOUNDARY_INPUTS:
        result = sort(list(data))
        assert isinstance(result, list), f"{name}: возвращён не список"
        assert result == sorted(data), f"{name}: на {data} получено {result}"


@pytest.mark.parametrize("name", sorted(SORTS))
def test_random_inputs_match_sorted(name: str) -> None:
    """Сверка с эталоном sorted() на трёхстах случайных входах."""
    sort = SORTS[name]
    rng = random.Random(0)
    for _ in range(300):
        data = [rng.randint(-100, 100) for _ in range(rng.randint(0, 80))]
        assert sort(list(data)) == sorted(data), f"{name}: расхождение на {data}"


@pytest.mark.parametrize("name", sorted(SORTS))
def test_input_is_not_mutated(name: str) -> None:
    """Алгоритмы сортируют копию и не портят входной список."""
    sort = SORTS[name]
    data = [5, -2, 7, 0, 3, 7]
    snapshot = list(data)
    sort(data)
    assert data == snapshot, f"{name}: изменил входной список"


@pytest.mark.parametrize("name", sorted(KEYED_SORTS))
def test_key_function_orders_keys(name: str) -> None:
    """Сортировка по ключу упорядочивает ключи и сохраняет элементы."""
    sort = KEYED_SORTS[name]
    rng = random.Random(1)
    for _ in range(100):
        pairs = [(rng.randint(0, 9), i) for i in range(rng.randint(0, 40))]
        result = sort(list(pairs))
        assert [p[0] for p in result] == sorted(p[0] for p in pairs)
        assert sorted(result) == sorted(pairs), f"{name}: потеряны элементы"


def test_selection_sort_comparisons_are_always_quadratic() -> None:
    """Выбор: ровно n(n-1)/2 сравнений на любом входе."""
    for n in (0, 1, 5, 20):
        inputs = [list(range(n)), list(range(n, 0, -1)), [3] * n]
        for data in inputs:
            _, comparisons, moves = lab.selection_sort(data)
            assert comparisons == n * (n - 1) // 2
            assert moves <= max(n - 1, 0)


def test_bubble_sort_counters() -> None:
    """Пузырёк: на упорядоченном входе n-1 сравнение и 0 обменов, на обратном n(n-1)/2."""
    n = 20
    _, comparisons, moves = lab.bubble_sort(list(range(n)))
    assert (comparisons, moves) == (n - 1, 0)

    _, _, moves = lab.bubble_sort(list(range(n, 0, -1)))
    assert moves == n * (n - 1) // 2


def test_insertion_sort_counters() -> None:
    """Вставки: лучший случай n-1 сравнение и 0 сдвигов, худший — n(n-1)/2."""
    n = 20
    _, comparisons, moves = lab.insertion_sort(list(range(n)))
    assert (comparisons, moves) == (n - 1, 0)

    _, comparisons, moves = lab.insertion_sort(list(range(n, 0, -1)))
    assert comparisons == n * (n - 1) // 2
    assert moves == n * (n - 1) // 2


def test_quick_sort_is_randomized_but_correct() -> None:
    """Разные seed дают корректный результат; один seed — воспроизводим."""
    data_rng = random.Random(5)
    data = [data_rng.randint(-50, 50) for _ in range(500)]
    assert len(set(data)) > 10, "тестовые данные выродились в константу"

    for seed in range(5):
        assert lab.quick_sort(data, rng=random.Random(seed)) == sorted(data)

    counters_a, counters_b = [0, 0], [0, 0]
    first = lab.quick_sort(data, rng=random.Random(42), counters=counters_a)
    second = lab.quick_sort(data, rng=random.Random(42), counters=counters_b)
    assert first == second
    assert counters_a == counters_b and counters_a[0] > 0


def test_quick_sort_sorted_input_is_n_log_n() -> None:
    """Рандомизация не даёт Θ(n²) на упорядоченном входе."""
    n = 4_096
    counters = [0, 0]
    result = lab.quick_sort(list(range(n)), rng=random.Random(0),
                            counters=counters)
    assert result == list(range(n))
    assert counters[0] < 4 * n * math.log2(n), (
        f"сравнений {counters[0]} — похоже на Θ(n²)")


def test_quick_sort_handles_many_duplicates() -> None:
    """На массиве из одинаковых элементов трёхпутевое разбиение линейно."""
    n = 5_000
    counters = [0, 0]
    result = lab.quick_sort([7] * n, rng=random.Random(0), counters=counters)
    assert result == [7] * n
    assert counters[0] < 10 * n, (
        f"сравнений {counters[0]} на n={n} — трёхпутевое разбиение не работает")


def test_deterministic_pivot_degrades_on_sorted_input() -> None:
    """Опыт рубрики: фиксированный пивот на упорядоченном входе квадратичен."""
    n = 1_024
    deterministic, randomized = [0, 0], [0, 0]
    lab.quick_sort_deterministic(list(range(n)), counters=deterministic)
    lab.quick_sort(list(range(n)), rng=random.Random(0), counters=randomized)
    assert deterministic[0] > randomized[0]


@pytest.mark.parametrize("name", ["bubble_sort", "insertion_sort"])
def test_stable_sorts_keep_equal_key_order(name: str) -> None:
    """Пузырёк и вставки устойчивы на 200 случайных наборах пар."""
    sort = KEYED_SORTS[name]
    rng = random.Random(0)
    for _ in range(200):
        pairs = [(rng.randint(0, 3), i) for i in range(rng.randint(0, 30))]
        assert sort(list(pairs)) == sorted(pairs, key=lambda p: p[0]), (
            f"{name}: нарушен порядок равных ключей")


def test_selection_sort_is_not_stable() -> None:
    """Выбор обменивает элемент через равные ключи — на паре это видно."""
    pairs = [(1, "a"), (1, "b"), (0, "c")]
    got = lab.selection_sort(pairs, key=lambda p: p[0])[0]
    assert got != sorted(pairs, key=lambda p: p[0])
    assert got[0] == (0, "c")


def test_make_inputs_returns_three_classes() -> None:
    """Классы входов: упорядоченный, обратный и тот же набор в случайном порядке."""
    inputs = lab.make_inputs(200, seed=45)
    assert set(inputs) == {"упорядоченный", "случайный", "обратный"}
    assert inputs["упорядоченный"] == sorted(inputs["упорядоченный"])
    assert inputs["обратный"] == sorted(inputs["обратный"], reverse=True)
    assert sorted(inputs["случайный"]) == inputs["упорядоченный"]
    assert inputs == lab.make_inputs(200, seed=45), "входы не детерминированы"


def test_log_log_slope_matches_growth() -> None:
    """Наклон log-log: 2 у n², около 1 у n log n."""
    sizes = [100, 200, 400, 800]
    assert lab.log_log_slope([(n, n ** 2) for n in sizes]) == pytest.approx(2.0)
    slope = lab.log_log_slope([(n, n * math.log2(n)) for n in sizes])
    assert 1.0 < slope < 1.3


def test_self_check_passes(capsys: pytest.CaptureFixture[str]) -> None:
    """Встроенная самопроверка решения не находит нарушений."""
    lab.self_check()
    assert "self_check: OK" in capsys.readouterr().out


def test_full_run_writes_plots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                               capsys: pytest.CaptureFixture[str]) -> None:
    """Полный прогон на уменьшенных размерах пишет три графика и таблицы."""
    pytest.importorskip("matplotlib")
    monkeypatch.setattr(lab, "REPEATS", 1)
    monkeypatch.setattr(lab, "SIZES_QUADRATIC", (100, 200))
    monkeypatch.setattr(lab, "SIZES_QUICK", (100, 300))
    monkeypatch.setattr(lab, "SIZES_PIVOT", (100, 200))
    monkeypatch.setattr(lab, "SEED", 45)

    lab.run(seed=45, out_dir=tmp_path)
    out = capsys.readouterr().out
    assert "### Наклоны log-log" in out
    for name in ("lab03_simple_sorts_loglog.png",
                 "lab03_quicksort_loglog.png",
                 "lab03_counters_loglog.png"):
        assert (tmp_path / name).stat().st_size > 0, f"не создан {name}"
