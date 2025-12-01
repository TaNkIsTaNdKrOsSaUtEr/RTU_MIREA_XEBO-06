```mermaid

%%{init: {'theme': 'neutral', 'themeVariables': { 'fontFamily': 'Arial, sans-serif', 'fontSize': '14px'}}}%%

flowchart TD
    Start([🏁 Начало анализа]) --> LoadData
    
    subgraph LoadData [📂 ШАГ 1: Загрузка данных]
        direction LR
        LD1["`pd.read_csv()`<br>Загрузка CSV файла"] 
        LD2["`.shape`<br>Анализ размерности"] 
        LD3["`.head()`<br>Просмотр первых строк"]
    end
    
    LoadData --> ExploreData
    
    subgraph ExploreData [🔍 ШАГ 2: Изучение данных]
        direction TB
        ED1["`.info()`<br>Информация о типах данных"] 
        ED2["`.describe()`<br>Статистический анализ"] 
        ED3["`.isnull().sum()`<br>Поиск пропущенных значений"] 
        ED4["`.value_counts()`<br>Распределение целевой переменной"]
    end
    
    ExploreData --> Preprocess
    
    subgraph Preprocess [⚙️ ШАГ 3: Предобработка данных]
        direction LR
        P1["`fillna(median)`<br>Заполнение пропусков медианой"] 
        P2["Проверка отсутствия пропусков"] 
    end
    
    Preprocess --> Visualize
    
    subgraph Visualize [📊 ШАГ 4: Визуализация]
        direction TB
        V1["Графики распределения<br>(магнитуда, глубина)"] 
        V2["Динамика по годам"] 
        V3["Карта землетрясений"] 
        V4["Матрица корреляций"] 
        V5["Соотношение классов"]
    end
    
    Visualize --> PrepareML
    
    subgraph PrepareML [🎯 ШАГ 5: Подготовка к ML]
        direction LR
        PM1["Выбор признаков<br>(исключаем Year, Month)"] 
        PM2["`train_test_split`<br>70% train, 30% test"] 
        PM3["`StandardScaler`<br>Стандартизация данных"]
    end
    
    PrepareML --> TrainModels
    
    subgraph TrainModels [🤖 ШАГ 6: Обучение моделей]
        direction TB
        TM1[Дерево решений<br>+ GridSearchCV] 
        TM2[Случайный лес<br>+ GridSearchCV] 
        TM3[SVM<br>+ GridSearchCV] 
        TM4[KNN<br>+ GridSearchCV] 
        TM5[Логистическая регрессия<br>+ GridSearchCV] 
        TM6[Наивный Байес]
    end
    
    TrainModels --> CompareResults
    
    subgraph CompareResults [📈 ШАГ 7: Сравнение результатов]
        direction LR
        CR1["Сводная таблица метрик<br>(Accuracy, Precision, Recall, F1)"] 
        CR2["Визуализация результатов<br>4 сравнительных графика"] 
        CR3["Матрица ошибок<br>лучшей модели"]
    end
    
    CompareResults --> Analyze
    
    subgraph Analyze [💡 ШАГ 8: Анализ и выводы]
        direction TB
        A1["Определение лучшей модели<br>по метрикам"] 
        A2["`classification_report`<br>Подробный отчёт"] 
        A3["Важность признаков<br>(для tree-based моделей)"] 
        A4["Итоговое резюме<br>и выводы"]
    end
    
    Analyze --> End([✅ Завершение анализа])

    %% Стилизация
    classDef startEnd fill:#4CAF50,stroke:#2E7D32,color:white,stroke-width:2px
    classDef stepHeader fill:#2196F3,stroke:#1565C0,color:white,font-weight:bold
    classDef process fill:#E8F5E8,stroke:#4CAF50,stroke-width:1.5px
    classDef mlProcess fill:#E3F2FD,stroke:#1976D2,stroke-width:1.5px
    classDef visualProcess fill:#F3E5F5,stroke:#7B1FA2,stroke-width:1.5px
    
    class Start,End startEnd
    class LoadData,ExploreData,Preprocess,Visualize,PrepareML,TrainModels,CompareResults,Analyze stepHeader
    class LD1,LD2,LD3,PM1,PM2,PM3,CR1,CR2,CR3 process
    class ED1,ED2,ED3,ED4,P1,P2,A1,A2,A3,A4 mlProcess
    class TM1,TM2,TM3,TM4,TM5,TM6 visualProcess
    class V1,V2,V3,V4,V5 visualProcess

```
