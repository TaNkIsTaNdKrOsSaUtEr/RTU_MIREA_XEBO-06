# **Решение задач по основам Python для химических расчетов**


#**Задача 1. Инициализация переменных и вывод типов**

substance = "NaCl"
molar_mass = 58.44
moles = 2

print(substance, type(substance))
print(molar_mass, type(molar_mass))
print(moles, type(moles))

# Преобразование к float и int
print(float(substance))  # ValueError: нельзя преобразовать "NaCl" в float
# Для демонстрации преобразуем только числовые переменные
print(int(molar_mass))  # 58 (отбрасывается дробная часть)
print(float(moles))     # 2.0




#**Задача 2. Расчёт массы вещества**

name = input("Введите название вещества: ")
molar_mass = float(input("Молярная масса (г/моль): "))
moles = int(input("Количество молей: "))

mass = molar_mass * moles
print(f"Масса {name}: {mass} г")

# Обмен значений
molar_mass, mass = mass, molar_mass
print(f"После обмена: молярная масса = {molar_mass}, масса = {mass}")




#*Задача 3. Анализ строки**

formula = input("Введите формулу вещества: ")

print("Есть цифры:", any(c.isdigit() for c in formula))
first_digit_index = next((i for i, c in enumerate(formula) if c.isdigit()), -1)
print("Индекс первой цифры:", first_digit_index)
print("Только буквы:", formula.isalpha())
print("Позиция 'O':", formula.find("O"))
print("Верхний регистр:", formula.upper())
print("Нижний регистр:", formula.lower())




#**Задача 4. Суммирование и разделение строк**

name1 = "H2"  
name2 = "O"  
name3 = "NaCl"

combined = name1 + name2 + name3
print("Объединенная строка:", combined)

split_list = combined.split("C")
print("Разделение по 'C':", split_list)




#**Задача 5. Условная проверка состава**

symbol = input("Введите химический символ: ")

if symbol == "O":
    print("Кислород")
elif symbol == "H":
    print("Водород")
elif symbol == "Na":
    print("Натрий")
else:
    print("Неизвестный элемент")




#**Задача 6. Цикл while — определение pH**

while True:
    ph = float(input("Введите pH (-1 для выхода): "))
    if ph == -1:
        break
    if ph < 7:
        print("Кислая")
    elif ph == 7:
        print("Нейтральная")
    else:
        print("Щелочная")




# Задача 7. Средняя температура (цикл while)**

count = 0
total = 0
while count < 5:
    temp = float(input(f"Введите температуру {count+1}: "))
    total += temp
    count += 1

print("Средняя температура:", total / 5)




#**Задача 8. Таблица кратных масс**

for i in range(1, 11):
    print(f"{i} * 12 = {i * 12}")




#**Задача 9. Анализ цепочки реакции**

compounds = ['H2', 'O2', 'H2O', 'H2O2']

long_compounds = [c for c in compounds if len(c) > 3]
print("Соединения длиной > 3:", long_compounds)

total_count = len(compounds)
print("Общее количество соединений:", total_count)




#**Пояснения:**
#1. В Задаче 1 преобразование `"NaCl"` в число вызовет ошибку — это ожидаемо.
#2. В Задаче 3 метод `.isalpha()` вернет `False` для формул с цифрами.
#3. В Задаче 6 используется `break` для выхода из цикла по условию.
#4. В Задаче 9 список `long_compounds` создается через list comprehension.


