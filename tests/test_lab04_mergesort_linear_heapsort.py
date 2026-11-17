#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Тесты решения ЛР 4 «MergeSort, Counting, Radix и HeapSort».

Проверяется то, что требует рубрика-04:

1) корректность четырёх сортировок и перенесённого QuickSort на граничных и
   сотнях случайных входов;
2) устойчивость слияния, устойчивого прохода CountingSort и поразрядного
   прохода RadixSort (пары «ключ + номер», сверка с устойчивым ``sorted``);
3) неустойчивый поразрядный проход действительно ломает результат (иначе
   демонстрация в отчёте ничего не доказывает);
4) границы области применимости: диапазон ключей CountingSort и
   отрицательные ключи RadixSort;
5) HeapSort — in-place через просейку вниз, свойство кучи после build_heap;
6) цена диапазона ключей CountingSort по времени и памяти;
7) встроенная самопроверка проходит, а полный прогон пишет графики.

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
LAB_PATH = REPO_ROOT / "lab04_mergesort_linear_heapsort.py"

BOUNDARY_INPUTS = [
    [],
    [7],
    [0],
    [1, 0],
    [5, 5, 5],
    [1, 2, 3, 4, 5],
    [5, 4, 3, 2, 1],
    [10, 0, 10, 0, 5, 5],
]


def _load_lab():
    """Загрузить решение как модуль Python, несмотря на дефис в имени файла."""
    assert LAB_PATH.is_file(), f"нет файла решения: {LAB_PATH}"
    spec = importlib.util.spec_from_file_location("lab04_solution", LAB_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


lab = _load_lab()

# Функции, возвращающие отсортированный список; ключи неотрицательны.
SORTS = {
    "merge_sort": lab.merge_sort,
    "counting_sort": lambda a: lab.counting_sort(a, key_max=999),
    "radix_sort_lsd": lab.radix_sort_lsd,
    "heap_sort": lab.heap_sorted,
    "quick_sort": lambda a: lab.quick_sort(a, rng=random.Random(0)),
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
    """Сверка с эталоном sorted() на двухстах случайных входах."""
    sort = SORTS[name]
    rng = random.Random(0)
    for _ in range(200):
        data = [rng.randint(0, 999) for _ in range(rng.randint(0, 60))]
        assert sort(list(data)) == sorted(data), f"{name}: расхождение на {data}"


@pytest.mark.parametrize("name", ["merge_sort", "counting_sort",
                                  "radix_sort_lsd", "quick_sort"])
def test_sort_does_not_mutate_input(name: str) -> None:
    """Сортировки, возвращающие новый список, не меняют вход."""
    data = [5, 0, 7, 3, 3]
    snapshot = list(data)
    SORTS[name](data)
    assert data == snapshot, f"{name}: изменил входной список"


def test_heap_sort_is_in_place() -> None:
    """HeapSort по контракту меняет переданный список и возвращает None."""
    data = [5, 0, 7, 3, 3]
    result = lab.heap_sort(data)
    assert result is None
    assert data == [0, 3, 3, 5, 7]


def test_merge_sort_is_stable() -> None:
    """MergeSort сохраняет порядок равных ключей на 200 наборах пар."""
    rng = random.Random(0)
    for _ in range(200):
        pairs = [(rng.randint(0, 5), i) for i in range(rng.randint(0, 100))]
        got = lab.merge_sort(pairs, key=lambda p: p[0])
        assert got == sorted(pairs, key=lambda p: p[0])


def test_counting_sort_is_stable() -> None:
    """Расстановка с конца массива сохраняет порядок равных ключей."""
    rng = random.Random(4)
    for _ in range(100):
        pairs = [(rng.randint(0, 9), i) for i in range(rng.randint(0, 80))]
        got = lab.counting_sort(pairs, key_max=9, key=lambda p: p[0])
        assert got == sorted(pairs, key=lambda p: p[0])


def test_radix_sort_is_stable() -> None:
    """Поразрядный проход устойчив, поэтому LSD не ломает порядок равных ключей."""
    rng = random.Random(1)
    for _ in range(100):
        pairs = [(rng.randint(0, 999), i) for i in range(rng.randint(0, 80))]
        got = lab.radix_sort_lsd(pairs, key=lambda p: p[0])
        assert got == sorted(pairs, key=lambda p: p[0])


def test_unstable_radix_pass_really_fails() -> None:
    """Намеренно неустойчивый проход обязан давать неверный результат."""
    example = [21, 1, 11, 2, 12, 3]
    assert lab.radix_sort_lsd(example) == sorted(example)
    assert lab.radix_sort_lsd_unstable(example) != sorted(example)


def test_counting_sort_rejects_keys_out_of_range() -> None:
    """CountingSort не расширяет диапазон на лету, а сообщает об ошибке."""
    with pytest.raises(ValueError):
        lab.counting_sort([1, 2, 1000], key_max=999)
    with pytest.raises(ValueError):
        lab.counting_sort([-1, 2], key_max=999)
    assert lab.counting_sort([], key_max=999) == []
    assert lab.counting_sort([0, 0, 0], key_max=0) == [0, 0, 0]


@pytest.mark.parametrize("name", ["radix_sort_lsd", "radix_sort_lsd_unstable"])
def test_radix_rejects_negative_keys(name: str) -> None:
    """Отрицательные ключи — вне области применимости LSD, это ValueError."""
    fn = getattr(lab, name)
    with pytest.raises(ValueError):
        fn([3, -1, 2])


def test_build_heap_property() -> None:
    """После build_heap родитель не меньше детей (max-куча)."""
    rng = random.Random(3)
    for _ in range(100):
        data = [rng.randint(-100, 100) for _ in range(rng.randint(0, 50))]
        lab.build_heap(data)
        for i in range(1, len(data)):
            parent = (i - 1) // 2
            assert data[parent] >= data[i], f"свойство кучи нарушено в узле {i}"


def test_measure_counting_range_reports_time_and_memory(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Опыт с k возвращает по строке на диапазон; память растёт вместе с k.

    Полный опыт считает до k = 10⁷, поэтому константы модуля подменяются
    уменьшенными значениями.
    """
    monkeypatch.setattr(lab, "RANGE_SIZES", (10, 10_000))
    monkeypatch.setattr(lab, "RANGE_N", 2_000)
    monkeypatch.setattr(lab, "REPEATS", 1)

    rows = lab.measure_counting_range(random.Random(0))
    assert len(rows) == 2
    for k, elapsed, peak in rows:
        assert k > 0 and elapsed > 0 and peak > 0
    assert rows[1][2] > rows[0][2], (
        f"память не выросла с k: {rows[0][2]} -> {rows[1][2]}")


def test_log_log_slope_matches_growth() -> None:
    """Наклон log-log: 2 у n², около 1 у n log n."""
    sizes = [100, 200, 400, 800]
    assert lab.log_log_slope([(n, n ** 2) for n in sizes]) == pytest.approx(2.0)
    slope = lab.log_log_slope([(n, n * math.log2(n)) for n in sizes])
    assert 1.0 < slope < 1.3


def test_algorithm_table_has_six_entries() -> None:
    """Бенчмарк сравнивает четыре свои сортировки, QuickSort из ЛР 3 и sorted()."""
    table = lab.algorithm_table()
    assert [name for name, _ in table] == list(lab.ALGORITHM_ORDER)
    assert len(table) == 6


def test_self_check_passes(capsys: pytest.CaptureFixture[str]) -> None:
    """Встроенная самопроверка решения не находит нарушений."""
    lab.self_check()
    assert "self_check: OK" in capsys.readouterr().out


def test_full_run_writes_plots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                               capsys: pytest.CaptureFixture[str]) -> None:
    """Полный прогон на уменьшенных размерах пишет оба графика и таблицы."""
    pytest.importorskip("matplotlib")
    monkeypatch.setattr(lab, "REPEATS", 1)
    monkeypatch.setattr(lab, "SIZES", (100, 300))
    monkeypatch.setattr(lab, "RANGE_SIZES", (10, 100, 1_000))
    monkeypatch.setattr(lab, "RANGE_N", 1_000)

    lab.run(seed=45, out_dir=tmp_path)
    out = capsys.readouterr().out
    assert "### Наклоны log-log" in out
    assert "## Сводная таблица свойств" in out
    assert "CountingSort: цена диапазона ключей" in out
    for name in ("lab04_loglog.png", "lab04_counting_range.png"):
        assert (tmp_path / name).stat().st_size > 0, f"не создан {name}"
