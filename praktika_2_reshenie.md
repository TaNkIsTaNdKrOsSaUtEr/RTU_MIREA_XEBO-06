### Решение задач по программированию

#### 1) Учёт реактивов в лаборатории
```python
reagents = [
    ["NaCl", 58.44, 120.0],
    ["H2O", 18.015, 5000.0],
    ["H2O2", 34.01, 250.0],
    ["C2H5OH", 46.07, 80.0],
]

# 1. Добавление нового реагента
reagents.append(["HCl", 36.46, 100.0])

# 2. Удаление "C2H5OH"
reagents.remove(["C2H5OH", 46.07, 80.0])

# 3. Список веществ с массой < 200 г
less_than_200 = [reagent for reagent in reagents if reagent[2] < 200]
print("Масса < 200 г:", less_than_200)

# 4. Сортировка по молярной массе
reagents.sort(key=lambda x: x[1])
print("Отсортировано по молярной массе:", reagents)

# 5. Список названий веществ
names = [reagent[0] for reagent in reagents]
print("Названия:", names)

# 6. Добавление нескольких элементов
reagents.extend([["HCl", 36.46, 100.0], ["KOH", 56.11, 150.0]])

# 7. Вставка "H2O" на позицию 1
reagents.insert(1, ["H2O", 18.015, 5000.0])

# 8. Извлечение последнего элемента
last_element = reagents.pop()
print("Извлечённый элемент:", last_element)
```

#### 2) Конкатенация и срезы
```python
metals = ["Na", "K", "Mg"]
halogens = ["Cl", "F"]

# Генерация комбинаций
combinations = [m + h for m in metals for h in halogens if len(m + h) in (4, 5)]
print("Все комбинации:", combinations)
print("Первые 3:", combinations[:3])
print("Последние 2:", combinations[-2:])
```

#### 3) Проверка на вхождение и индексы
```python
acids = ["HCl", "H2SO4", "HNO3", "H3PO4"]
formula = input("Введите формулу кислоты: ")

if formula in acids:
    print("Индекс:", acids.index(formula))
else:
    print("Вещества нет в списке")
```

#### 4) Таблица концентраций
```python
conc = [
    [12.1, 0.5, 1.2, 5.0],
    [10.0, 0.8, 1.0, 4.6],
    [11.2, 0.6, 1.1, 5.1]
]

# Средняя концентрация по веществам
avg_conc = [sum(col) / len(col) for col in zip(*conc)]
print("Средняя концентрация:", avg_conc)

# Максимальная концентрация и координаты
max_val = max(max(row) for row in conc)
for i, row in enumerate(conc):
    if max_val in row:
        print(f"Максимум: {max_val} на позиции ({i}, {row.index(max_val)})")

# Доля каждого вещества в образце
fractions = []
for row in conc:
    total = sum(row)
    fractions.append([val / total for val in row])

print("Доли веществ:", fractions)
```

#### 5) Таблица Менделеева
```python
atomic_mass = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "Na": 22.990, "Cl": 35.45}
element = input("Введите символ элемента: ")
mass = atomic_mass.get(element, "Неизвестный элемент")
print(f"Атомная масса: {mass}")
```

#### 6) Частотный анализ формулы
```python
formula = "C6H12O6"
freq = {}
for char in formula:
    freq[char] = freq.get(char, 0) + 1
print("Частоты символов:", freq)
```

#### 7) Обновление словаря реактивов
```python
stock = {"HCl": 2, "NaOH": 5, "H2SO4": 1}
stock.update({"NaOH": 3, "H2O": 10})

# Выдача 1 л "HCl"
if "HCl" in stock:
    stock["HCl"] -= 1
else:
    print("HCl нет в наличии")

print("Реактивы:", list(stock.keys()))
print("Суммарный объём:", sum(stock.values()))
```

#### 8) Уникальные элементы формулы
```python
formula1 = input("Введите первую формулу: ").upper()
formula2 = input("Введите вторую формулу: ").upper()

set1 = {char for char in formula1 if char.isalpha()}
set2 = {char for char in formula2 if char.isalpha()}

print("Уникальные элементы:", len(set1), len(set2))
print("Пересечение:", set1 & set2)
print("Объединение:", set1 | set2)
print("Разность:", set1 - set2)
```

#### 9) Набор постоянных
```python
constants = (6.022e23, 8.314, 1.38e-23)
print("NA:", constants[0], "R:", constants[1], "kB:", constants[2])

# Попытка изменения вызовет ошибку:
# constants[0] = 1  # TypeError: 'tuple' object does not support item assignment
```

#### 10) Подсчёт массовой доли
```python
def mass_fraction(component_mass, total_mass):
    if total_mass == 0:
        return 0
    return (component_mass / total_mass) * 100

comp = float(input("Масса компонента: "))
total = float(input("Общая масса: "))
print(f"Массовая доля: {mass_fraction(comp, total):.2f}%")
```

#### 11) Нормализация записи формулы
```python
formula = input("Введите формулу: ").strip()
print("Верхний регистр:", formula.upper())
print("Нижний регистр:", formula.lower())
index = formula.upper().find("CO")
print("Индекс 'CO':", index if index != -1 else "Не найдено")
```

#### 12) Парсер символов формулы
```python
formula = "H2O2"
digits_sum = 0
letters = ""

for char in formula:
    if char.isdigit():
        digits_sum += int(char)
    else:
        letters += char

print(f"Сумма индексов = {digits_sum}, элементы = {letters}")
```

#### 13) Класс Reagent
```python
class Reagent:
    unit = "г"
    
    def __init__(self, name: str, molar_mass: float):
        self.name = name
        self.molar_mass = molar_mass
        
    def mass(self, moles: float) -> float:
        return self.molar_mass * moles

# Создание объектов
nacl = Reagent("NaCl", 58.44)
h2o = Reagent("H2O", 18.015)

print("Масса NaCl:", nacl.mass(0.5), nacl.unit)
print("Масса H2O:", h2o.mass(0.5), h2o.unit)

# Доступ к статическому атрибуту
print("Единица измерения (через класс):", Reagent.unit)
print("Единица измерения (через экземпляр):", nacl.unit)
```
