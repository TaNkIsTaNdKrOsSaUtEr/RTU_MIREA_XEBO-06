# 1. ОСНОВНЫЕ ОПЕРАЦИИ
stroka_one = 3
stroka_one += 2  # Увеличиваем на 2
stroka_one -= 1  # Уменьшаем на 1
print(stroka_one)  # Результат: 4

# Обмен значений
a, b = 15, 30
a, b = b, a  # Меняем значения местами
print(f'a = {a}, b = {b}')  # a = 30, b = 15

# 2. ТИПЫ ДАННЫХ
# Целочисленный
int_var = 100
# Вещественный
float_var = 3.14
# Логический
bool_var = True
# Строковый
str_var = "Hello"

# Преобразование типов
x = int("10")   # Из строки в целое
y = float("1.5") # Из строки в вещественное
z = str(10)     # Из числа в строку

# 3. ВВОД ДАННЫХ
name = input("Введите имя: ")  # Возвращает строку
age = int(input("Введите возраст: "))  # Преобразуем в число

# 4. СТРОКОВЫЕ ОПЕРАЦИИ
# Конкатенация (сложение строк)
s1 = "Текст-"
s2 = "Слово-"
s3 = "Буква"
result = s1 + s2 + s3

# Методы строк:
text = "  Hello World  "
print(text.strip())      # "Hello World" - удаляет пробелы
print(text.find("W"))    # 8 - индекс первого вхождения
print("123".isdigit())   # True - проверка на цифры
print("abc".isalpha())   # True - проверка на буквы
print("a,b,c".split(',')) # ['a', 'b', 'c'] - разделение строки
print(text.replace("H", "J"))  # "Jello World" - замена
print("hello".upper())   # "HELLO" - верхний регистр
print("HELLO".lower())   # "hello" - нижний регистр

# 5. ОПЕРАТОРЫ ВЕТВЛЕНИЯ
a = int(input("a: "))
b = int(input("b: "))

if a > b:
    print("a больше b")
elif a < b:
    print("a меньше b")
else:
    print("a равно b")

# Логические операторы
if a > 0 and b > 0:
    print("Оба положительные")
if a > 0 or b > 0:
    print("Хотя бы один положительное")
if not a > 0:
    print("a не положительное")

# 6. ЦИКЛЫ
# While
counter = 5
while counter > 0:
    print(counter)
    counter -= 1
    if counter == 2:
        break  # Прерывание цикла
    if counter == 4:
        continue  # Переход к следующей итерации

# For
for i in [1, 2, 3, 4, 5]:
    print(i)

# Функция range
for i in range(5):      # 0,1,2,3,4
    print(i)

for i in range(1, 6):   # 1,2,3,4,5
    print(i)

for i in range(0, 10, 2):  # 0,2,4,6,8
    print(i)
    