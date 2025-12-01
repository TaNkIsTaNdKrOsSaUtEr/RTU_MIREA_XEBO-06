```mermaid
flowchart TD
    A[🏁 Начало анализа данных] --> B
    
    subgraph B [📂 ШАГ 1: Загрузка данных]
        B1[pd.read_csv<br>Загрузка CSV файла] 
        B2[.shape<br>Анализ размерности] 
        B3[.head<br>Просмотр первых строк]
        B1 --> B2 --> B3
    end
    
    B --> C
    
    subgraph C [🔍 ШАГ 2: Изучение данных]
        C1[.info<br>Информация о типах данных]
        C2[.describe<br>Статистический анализ]
        C3[.isnull.sum<br>Поиск пропущенных значений]
        C4[.value_counts<br>Распределение целевой переменной]
        C1 --> C2 --> C3 --> C4
    end
    
    C --> D
    
    subgraph D [⚙️ ШАГ 3: Предобработка данных]
        D1[fillna(median)<br>Заполнение пропусков]
        D2[Проверка отсутствия пропусков]
        D1 --> D2
    end
    
    D --> E
    
    subgraph E [📊 ШАГ 4: Визуализация]
        E1[Графики распределения]
        E2[Динамика по годам]
        E3[Карта землетрясений]
        E4[Матрица корреляций]
        E5[Соотношение классов]
        E1 --> E2 --> E3 --> E4 --> E5
    end
    
    E --> F
    
    subgraph F [🎯 ШАГ 5: Подготовка к ML]
        F1[Выбор признаков]
        F2[train_test_split<br>70% train / 30% test]
        F3[StandardScaler<br>Стандартизация]
        F1 --> F2 --> F3
    end
    
    F --> G
    
    subgraph G [🤖 ШАГ 6: Обучение моделей]
        G1[Дерево решений<br>+ GridSearchCV]
        G2[Случайный лес<br>+ GridSearchCV]
        G3[SVM<br>+ GridSearchCV]
        G4[KNN<br>+ GridSearchCV]
        G5[Логистическая регрессия<br>+ GridSearchCV]
        G6[Наивный Байес]
        G1 --> G2 --> G3 --> G4 --> G5 --> G6
    end
    
    G --> H
    
    subgraph H [📈 ШАГ 7: Сравнение результатов]
        H1[Сводная таблица метрик]
        H2[Визуализация результатов]
        H3[Матрица ошибок лучшей модели]
        H1 --> H2 --> H3
    end
    
    H --> I
    
    subgraph I [💡 ШАГ 8: Анализ и выводы]
        I1[Определение лучшей модели]
        I2[Подробный отчёт]
        I3[Важность признаков]
        I4[Итоговое резюме]
        I1 --> I2 --> I3 --> I4
    end
    
    I --> J[✅ Завершение анализа]

    %% Простая стилизация без сложных классов
    style A fill:#4CAF50,color:#fff
    style B fill:#2196F3,color:#fff
    style C fill:#2196F3,color:#fff
    style D fill:#2196F3,color:#fff
    style E fill:#2196F3,color:#fff
    style F fill:#2196F3,color:#fff
    style G fill:#9C27B0,color:#fff
    style H fill:#2196F3,color:#fff
    style I fill:#2196F3,color:#fff
    style J fill:#4CAF50,color:#fff
    
    style B1 fill:#E8F5E8
    style B2 fill:#E8F5E8
    style B3 fill:#E8F5E8
    style C1 fill:#E3F2FD
    style C2 fill:#E3F2FD
    style C3 fill:#E3F2FD
    style C4 fill:#E3F2FD
    style D1 fill:#E3F2FD
    style D2 fill:#E3F2FD
    style E1 fill:#F3E5F5
    style E2 fill:#F3E5F5
    style E3 fill:#F3E5F5
    style E4 fill:#F3E5F5
    style E5 fill:#F3E5F5
    style F1 fill:#E8F5E8
    style F2 fill:#E8F5E8
    style F3 fill:#E8F5E8
    style G1 fill:#F3E5F5
    style G2 fill:#F3E5F5
    style G3 fill:#F3E5F5
    style G4 fill:#F3E5F5
    style G5 fill:#F3E5F5
    style G6 fill:#F3E5F5
    style H1 fill:#E8F5E8
    style H2 fill:#E8F5E8
    style H3 fill:#E8F5E8
    style I1 fill:#E3F2FD
    style I2 fill:#E3F2FD
    style I3 fill:#E3F2FD
    style I4 fill:#E3F2FD

```
