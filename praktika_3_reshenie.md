Отличный набор задач! Вот решения для каждой из них, с комментариями и кодом.

---

### **Библиотека Matplotlib**

#### **Задача 1. Диаграмма растворимости (Line plot)**

```python
import matplotlib.pyplot as plt

temperature = [0, 20, 40, 60, 80, 100]
solubility_NaCl = [35, 36, 37, 38, 39, 40]

plt.figure(figsize=(8, 5)) # Создаем фигуру
plt.plot(temperature, solubility_NaCl, marker='o', linestyle='-', linewidth=2) # Строим линейный график с маркерами
plt.xlabel('Температура (°C)') # Подпись оси X
plt.ylabel('Растворимость (г/100г H₂O)') # Подпись оси Y
plt.title('Зависимость растворимости NaCl от температуры') # Заголовок
plt.grid(True, linestyle='--', alpha=0.7) # Включаем сетку
plt.tight_layout() # Автоматическая настройка layout
plt.show() # Показываем график
```

#### **Задача 2. Энергия связи молекул (Scatter plot)**

```python
import matplotlib.pyplot as plt

molecules = ["H₂", "O₂", "N₂", "Cl₂", "CO₂"]
bond_energy = [436, 498, 946, 243, 799]

plt.figure(figsize=(10, 6))
plt.scatter(molecules, bond_energy, s=100, color='red', alpha=0.7) # s - размер точек

# Подписываем точки (аннотируем)
for i, mol in enumerate(molecules):
    plt.annotate(mol, (mol, bond_energy[i]), xytext=(5, 5), textcoords='offset points')

plt.xlabel('Молекулы')
plt.ylabel('Энергия связи (кДж/моль)')
plt.title('Энергия связи молекул')
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.show()
```

#### **Задача 3. Сравнение кислот (Bar chart)**

```python
import matplotlib.pyplot as plt

acids = ["HCl", "H₂SO₄", "HNO₃", "CH₃COOH", "H₂CO₃"]
pKa = [-7, -3, -1.4, 4.7, 6.4]
colors = ['red', 'orange', 'yellow', 'green', 'blue'] # Создаем список цветов

plt.figure(figsize=(10, 6))
bars = plt.bar(acids, pKa, color=colors) # Строим столбцы с разными цветами

plt.xlabel('Кислоты')
plt.ylabel('pKa')
plt.title('Кислотные константы (pKa)')
plt.tight_layout()
plt.show()
```

#### **Задача 4. Распределение масс молекул (Histogram)**

```python
import matplotlib.pyplot as plt

masses = [18, 28, 44, 98, 180, 18, 28, 44, 98, 180, 18, 28]

plt.figure(figsize=(10, 6))
plt.hist(masses, bins=5, edgecolor='black', alpha=0.7) # bins=5 - 5 интервалов
plt.xlabel('Молярная масса (г/моль)')
plt.ylabel('Частота')
plt.title('Распределение молярных масс')
plt.grid(True, axis='y', linestyle='--')
plt.tight_layout()
plt.show()
```

#### **Задача 5. Состав воздуха (Pie chart)**

```python
import matplotlib.pyplot as plt

components = ["N₂", "O₂", "Ar", "CO₂", "Прочие"]
fractions = [78.08, 20.95, 0.93, 0.04, 0.0]

plt.figure(figsize=(8, 8))
plt.pie(fractions, labels=components, autopct='%1.1f%%', startangle=90)
plt.axis('equal') # Чтобы диаграмма была круглой
plt.title('Состав воздуха')
plt.tight_layout()
plt.show()
```

---

### **Библиотека NumPy**

#### **Молярные массы (1D ndarray)**

```python
import numpy as np

# 1. Создание массива и операции
molar_masses = np.array([18.015, 44.01, 31.998, 28.014, 60.052])
print("Элемент с индексом 2:", molar_masses[2])
print("Срез последних двух элементов:", molar_masses[-2:])

# 2. Атрибуты массива
print("ndim (количество осей):", molar_masses.ndim)
print("shape (форма):", molar_masses.shape)
print("size (размер):", molar_masses.size)
print("dtype (тип данных):", molar_masses.dtype)

# 3. Преобразование типа
molar_masses_float32 = molar_masses.astype(np.float32)
print("Новый dtype:", molar_masses_float32.dtype)
```

#### **Матрица концентраций (2D ndarray)**

```python
import numpy as np

concentrations = np.array([
    [12.1,  0.5,  1.2,  5.0],
    [10.0,  0.8,  1.0,  4.6],
    [11.2,  0.6,  1.1,  5.1]
])

# 1. Элемент (строка 2, столбец 3) - помните, индексация с 0!
print("Элемент [2, 3]:", concentrations[2, 3])

# 2. Срезы строк
print("Первая строка:", concentrations[0, :])
print("Последняя строка:", concentrations[-1, :])

# 3. Срез столбца
print("Второй столбец:", concentrations[:, 1])

# 4. Суммы и средние
print("Сумма всех элементов:", np.sum(concentrations))
print("Среднее всех элементов:", np.mean(concentrations))
print("Среднее по столбцам (axis=0):", np.mean(concentrations, axis=0))
print("Среднее по строкам (axis=1):", np.mean(concentrations, axis=1))
```

#### **Нули и единицы (инициализация)**

```python
import numpy as np

# 1. Нулевая матрица 5x5
zero_matrix = np.zeros((5, 5))
print("Нулевая матрица 5x5:\n", zero_matrix)

# 2. Массив единиц 3x2 типа bool
ones_bool_matrix = np.ones((3, 2), dtype=bool)
print("\nМаска наличия пика (3x2 bool):\n", ones_bool_matrix)

# 3. Объяснение dtype
# dtype определяет тип данных, хранящихся в массиве (int, float, bool и т.д.).
# Явное указание dtype критически важно для:
# - Экономии памяти (например, int8 вместо int64).
# - Обеспечения корректности вычислений (например, для логических операций нужен bool).
# - Совместимости с другими библиотеками и аппаратным обеспечением (например, GPU часто работают с float32).
```

#### **Температурный профиль реакции (векторизация)**

```python
import numpy as np

T = np.array([20, 25, 30, 35, 40])
# 1. Векторный расчет
v = 0.1 * T + 0.5
print("Скорости реакции v:", v)

# 2. Поиск условий
indices = np.where(v > 3.5)[0] # np.where возвращает кортеж, берем первый элемент
values = T[indices]
print("Индексы, где v > 3.5:", indices)
print("Значения T, где v > 3.5:", values)
```

#### **Булева маска для pH**

```python
import numpy as np

pH = np.array([2.1, 6.9, 7.0, 7.4, 13.0])

# 1. Создание масок
acid_mask = pH < 7
neutral_mask = pH == 7
alkaline_mask = pH > 7

# 2. Индексация масками
acidic_values = pH[acid_mask]
neutral_values = pH[neutral_mask]
alkaline_values = pH[alkaline_mask]

print("Кислые значения:", acidic_values)
print("Нейтральные значения:", neutral_values)
print("Щелочные значения:", alkaline_values)

# 3. Подсчет долей (в %)
total = pH.size
acid_percentage = (np.sum(acid_mask) / total) * 100
neutral_percentage = (np.sum(neutral_mask) / total) * 100
alkaline_percentage = (np.sum(alkaline_mask) / total) * 100

print(f"Доля кислых: {acid_percentage:.1f}%")
print(f"Доля нейтральных: {neutral_percentage:.1f}%")
print(f"Доля щелочных: {alkaline_percentage:.1f}%")
```

#### **Отбраковка шумных измерений (медиана/квантили)**

```python
import numpy as np

g = np.array([2.1, 2.0, 2.2, 8.0, 2.1, 2.0, 15.0, 2.1, 2.2])

# 1. Медиана и IQR
q25, median, q75 = np.percentile(g, [25, 50, 75])
iqr = q75 - q25
print(f"Q1: {q25}, Медиана: {median}, Q3: {q75}, IQR: {iqr}")

# 2. Определение границ и создание маски
lower_bound = q25 - 1.5 * iqr
upper_bound = q75 + 1.5 * iqr
print(f"Границы: [{lower_bound:.3f}, {upper_bound:.3f}]")

# Маска для значений внутри границ
clean_mask = (g >= lower_bound) & (g <= upper_bound)
g_clean = g[clean_mask]
print("Очищенный массив:", g_clean)

# 3. Сравнение средних
mean_original = np.mean(g)
mean_clean = np.mean(g_clean)
print(f"Среднее до очистки: {mean_original:.3f}")
print(f"Среднее после очистки: {mean_clean:.3f}")
```

#### **Нормализация спектра (по каждой строке)**

```python
import numpy as np

spectra = np.array([
    [10, 20, 30, 20, 10],
    [ 5,  5, 10,  5,  5],
    [ 0,  1,  4,  1,  0]
])

# 1. Нормировка (сумма строки = 1)
row_sums = spectra.sum(axis=1, keepdims=True) # keepdims=True для правильного broadcasting
# Защита от деления на ноль: заменим 0 на 1 в суммах, чтобы не сломать код
row_sums[row_sums == 0] = 1
spectra_norm = spectra / row_sums

print("Нормированные спектры:\n", spectra_norm)

# 2. Проверка сумм (должны быть ~1.0)
sums_after_norm = spectra_norm.sum(axis=1)
print("Суммы строк после нормировки:", sums_after_norm)
print("Проверка (суммы ≈ 1.0):", np.allclose(sums_after_norm, 1.0, atol=1e-9))
```

#### **Трансляция единиц измерения (broadcasting)**

```python
import numpy as np

m_g = np.array([0.5, 1.2, 3.0])
# 1. Перевод в мг
m_mg = m_g * 1000
print("Массы в мг:", m_mg)

# 2. Расчет количества вещества
M = np.array([18.015, 58.44, 98.0])
n = m_g / M # Broadcasting: m_g (3,) и M (3,) -> поэлементное деление
print("Количество вещества n (моль):", n)
```

#### **Reshape данных времен отклика**

```python
import numpy as np

t = np.arange(10, 130, 10) # 12 измерений: 10, 20, ..., 120
print("Исходный массив:", t)

# 1. Преобразование в матрицу 3x4 (по строкам)
t_matrix = t.reshape(3, 4)
print("Матрица 3x4:\n", t_matrix)

# 2. Подматрица (последние 2 строки, последние 3 столбца)
submatrix = t_matrix[-2:, -3:]
print("Подматрица 2x3:\n", submatrix)

# 3. Разворот обратно
t_flat_ravel = submatrix.ravel() # Выстраивает по строкам (C-style)
# t_flat_reshape = submatrix.reshape(-1,) # Альтернатива
print("Развернутый массив (ravel):", t_flat_ravel)
```

#### **Склейка рядов (stack)**

```python
import numpy as np

day = np.array([18, 20, 23, 24])
night = np.array([12, 13, 14, 15])

# 1. Вертикальная склейка (2 массива по 4 элемента -> матрица 2x4)
temp_matrix_v = np.vstack((day, night))
print("Вертикальный stack (2x4):\n", temp_matrix_v)

# 2. Горизонтальная склейка
# Сначала преобразуем каждый массив в столбец (4x1)
day_col = day.reshape(-1, 1)
night_col = night.reshape(-1, 1)
# Затем склеиваем горизонтально -> матрица 4x2
temp_matrix_h = np.hstack((day_col, night_col))
print("Горизонтальный stack (4x2):\n", temp_matrix_h)

# 3. Объяснение
# vstack (vertical stack) складывает массивы вертикально (добавляет строки).
# hstack (horizontal stack) складывает массивы горизонтально (добавляет столбцы).
# Формы массивов должны быть согласованы:
#   Для vstack: (N, M) и (K, M) -> (N+K, M)
#   Для hstack: (N, M) и (N, K) -> (N, M+K)
```

#### **Максимум пика в спектре**

```python
import numpy as np

I = np.array([0, 2, 5, 9, 7, 3, 1])
x_axis = np.arange(400, 400 + len(I)) # Создаем ось X

# 1. Находим индекс и значение максимума
idx_max = np.argmax(I)
value_max = I[idx_max]
print(f"Индекс максимума: {idx_max}, Значение максимума: {value_max}")

# 2. Оцениваем положение на оси X
x_at_max = x_axis[idx_max]
print(f"Положение пика (волновое число): {x_at_max}")
```

#### **Средние по веществам vs по образцам**

```python
import numpy as np

concentrations = np.array([
    [12.1,  0.5,  1.2,  5.0],
    [10.0,  0.8,  1.0,  4.6],
    [11.2,  0.6,  1.1,  5.1]
])

# 1. Средние по веществам (по столбцам, axis=0)
mean_by_substance = np.mean(concentrations, axis=0)
print("Среднее по веществам (axis=0):", mean_by_substance)

# 2. Средние по образцам (по строкам, axis=1)
mean_by_sample = np.mean(concentrations, axis=1)
print("Среднее по образцам (axis=1):", mean_by_sample)

# 3. Сравнение вариабельности (стандартное отклонение)
std_by_substance = np.std(concentrations, axis=0)
std_by_sample = np.std(concentrations, axis=1)

print("Стандартное отклонение по веществам:", std_by_substance)
print("Стандартное отклонение по образцам:", std_by_sample)

# Где вариабельность выше? Можно сравнить среднее std или max std.
print(f"Макс. std по веществам: {np.max(std_by_substance):.3f}")
print(f"Макс. std по образцам: {np.max(std_by_sample):.3f}")
# Вывод: вариабельность концентраций между веществами значительно выше.
```

#### **Гистограмма распределения молярных масс**

```python
import numpy as np

# Дан массив с повторами (для примера)
molar_masses_array = np.array([18, 18, 18, 44, 44, 98, 98, 180, 180, 28, 28, 28, 28])

# 1. Находим уникальные значения и их частоты
unique_masses, counts = np.unique(molar_masses_array, return_counts=True)
print("Уникальные массы:", unique_masses)
print("Их частоты:", counts)

# 2. Пары "масса-частота", отсортированные по массе
# np.unique уже возвращает отсортированные уникальные значения
mass_freq_pairs = list(zip(unique_masses, counts))
print("Пары (масса, частота):", mass_freq_pairs)
```

#### **Стехиометрия: масса реагента по уравнению**

```python
import numpy as np

# Молярные массы [HCl, NaOH, NaCl, H₂O]
M = np.array([36.46, 40.00, 58.44, 18.015])

# 1. Расчет для массы HCl
m_HCl = np.array([3.646, 7.292])
n_HCl = m_HCl / M[0] # Количество молей HCl

# По реакции n_NaOH = n_HCl (коэффициент 1:1)
n_NaOH_required = n_HCl
m_NaOH_required = n_NaOH_required * M[1] # Масса чистого NaOH
print(f"Массы NaOH, требуемые для нейтрализации: {m_NaOH_required} г")

# 2. Масса образующегося NaCl
n_NaCl_produced = n_HCl # По реакции n_NaCl = n_HCl
m_NaCl_produced = n_NaCl_produced * M[2]
print(f"Массы образующегося NaCl: {m_NaCl_produced} г")
```

#### **Смешение растворов (векторно)**

```python
import numpy as np

# Исходные данные
V_A = np.array([100, 200, 500]) # мл
C_A = np.array([1, 2, 5])       # г/л
V_B = np.array([50, 50, 100])   # мл
C_B = np.array([10, 10, 10])    # г/л

# 1. Масса соли в каждом растворе (m = C * V / 1000)
m_A = C_A * V_A / 1000
m_B = C_B * V_B / 1000
print("Масса соли в A (г):", m_A)
print("Масса соли в B (г):", m_B)

# 2. Общий объем и масса при смешении
V_total = V_A + V_B
m_total = m_A + m_B
# Новая концентрация C_new = m_total / (V_total / 1000)
C_new = m_total / (V_total / 1000)
print("Общие объемы (мл):", V_total)
print("Суммарные массы соли (г):", m_total)
print("Новые концентрации (г/л):", C_new)
```

#### **Приведение температур (broadcasting)**

```python
import numpy as np

T_K = np.array([298.15, 310.15, 350.15])

# 1. Перевод в °C и °F
T_C = T_K - 273.15
T_F = T_C * 9/5 + 32

print("T (K):", T_K)
print("T (°C):", T_C)
print("T (°F):", T_F)

# 2. Сборка матрицы 3x3
# Используем np.column_stack для склейки столбцов
T_matrix = np.column_stack((T_K, T_C, T_F))
print("Матрица температур [K, °C, °F]:\n", T_matrix)
```

#### **Маскирование по порогу сигнала**

```python
import numpy as np

# Создаем случайный массив сигналов 4x5 для примера
np.random.seed(42) # Для воспроизводимости
signals = np.random.rand(4, 5)
print("Исходные сигналы:\n", signals)

# 1. Создаем маску и зануляем элементы
mask = signals < 0.1
signals[mask] = 0 # In-place операция
print("\nСигналы после зануления:\n", signals)

# 2. Считаем долю зануленных
zero_fraction = np.mean(mask)
print(f"Доля зануленных значений: {zero_fraction:.2%}")
```

#### **Условие через np.where**

```python
import numpy as np

pH = np.array([2.1, 6.9, 7.0, 7.4, 13.0])

# 1. Создаем массив строк с условиями
# Используем вложенный np.where
ph_category = np.where(
    pH < 7, 
    "acid", 
    np.where(pH == 7, "neutral", "alkaline")
)
print("Категории pH:", ph_category)

# 2. Подсчет количества каждой категории
unique, counts = np.unique(ph_category, return_counts=True)
category_counts = dict(zip(unique, counts))
print("Количество каждой категории:", category_counts)
```

#### **Квантили и обрезка хвостов (clip)**

```python
import numpy as np

# Создаем массив откликов датчика с "хвостами"
np.random.seed(123)
data = np.random.normal(loc=10, scale=2, size=100) # Нормальное распределение
data = np.append(data, [30, -5]) # Добавляем выбросы

# 1. Находим квантили
q05, q95 = np.percentile(data, [5, 95])
print(f"5-й перцентиль: {q05:.3f}, 95-й перцентиль: {q95:.3f}")

# 2. Обрезка значений
data_clipped = np.clip(data, q05, q95)

# 3. Сравнение статистик
print(f"Среднее до обрезки: {np.mean(data):.3f}, после: {np.mean(data_clipped):.3f}")
print(f"Стд. отклонение до: {np.std(data):.3f}, после: {np.std(data_clipped):.3f}")
```
