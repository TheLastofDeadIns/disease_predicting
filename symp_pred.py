# Импорт нужных библиотек
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
from colorama import Fore, Style, init

# Настройка цветного вывода
init()

# Убираем ненужные предупреждения
warnings.filterwarnings('ignore')

# Загружаем данные
print(Fore.YELLOW + "Грузим данные..." + Style.RESET_ALL)
try:
    df = pd.read_csv("dataset.csv")
except:
    print(Fore.RED + "Ошибка! Не найден файл dataset.csv")
    exit()

# Подготовка данных
print(Fore.YELLOW + "Обрабатываю симптомы..." + Style.RESET_ALL)
sickness_list = []
for i in range(len(df)):
    symptoms = []
    # Собираем все симптомы для каждой болезни
    for col in df.columns[1:]:
        if pd.notna(df.iloc[i][col]):
            symptoms.append(str(df.iloc[i][col]).strip())
    sickness_list.append((df.iloc[i]["Disease"], symptoms))

# Создаем DataFrame
sickness_df = pd.DataFrame(sickness_list, columns=["Болезнь", "Симптомы"])

# Преобразуем симптомы в числа
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(sickness_df["Симптомы"])
y = sickness_df["Болезнь"]

# Обучаем модель
print(Fore.YELLOW + "Учу модель..." + Style.RESET_ALL)
model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model.fit(X, y)


def get_symptoms():
    """Получаем симптомы от пользователя"""
    print(Fore.CYAN + "\nВводи симптомы по одному (например: itching)")
    print("Когда закончишь, просто нажми Enter\n")

    symptoms = []
    while True:
        symp = input(Fore.BLUE + "Симптом: " + Style.RESET_ALL).strip()
        if not symp:
            break
        symptoms.append(symp)
    return symptoms


def show_results(pred, probs):
    """Показываем результаты анализа"""
    print(Fore.MAGENTA + "\n" + "=" * 50)
    print(Fore.YELLOW + " РЕЗУЛЬТАТ АНАЛИЗА ".center(50))
    print(Fore.MAGENTA + "=" * 50 + Style.RESET_ALL)

    # Сортируем болезни по вероятности
    sorted_results = sorted(zip(model.classes_, probs[0]),
                            key=lambda x: x[1], reverse=True)

    # Проверяем насколько уверена модель
    if sorted_results[0][1] > 0.5:
        print(Fore.GREEN + "\nСкорее всего это: " +
              Fore.WHITE + Style.BRIGHT + f"{sorted_results[0][0]}" +
              Fore.GREEN + f" ({sorted_results[0][1]:.1%} вероятность)")
    else:
        print(Fore.YELLOW + "\nВозможные варианты:")
        for i, (name, prob) in enumerate(sorted_results[:3]):
            print(f"{i + 1}. {Fore.WHITE}{name}{Style.RESET_ALL} - {Fore.CYAN}{prob:.1%}")

    # Выводим все варианты
    print(Fore.MAGENTA + "\n" + "-" * 50)
    print(Fore.YELLOW + " Все возможные варианты:".ljust(50))
    print(Fore.MAGENTA + "-" * 50 + Style.RESET_ALL)

    for name, prob in sorted_results:
        bar = "■" * int(prob * 30)  # Более простые символы для графика
        print(f"{Fore.WHITE}{name.ljust(20)} {Fore.CYAN}{prob:.1%} {bar}")


# Основная программа
print(Fore.GREEN + "\n" + "=" * 50)
print(Fore.YELLOW + " ПРОГРАММА ДЛЯ ДИАГНОСТИКИ ".center(50))
print(Fore.GREEN + "=" * 50 + Style.RESET_ALL)

while True:
    user_symptoms = get_symptoms()

    if not user_symptoms:
        print(Fore.RED + "Ты не ввел ни одного симптома!" + Style.RESET_ALL)
        continue

    # Проверяем какие симптомы знает система
    known_symptoms = [s for s in user_symptoms if s in mlb.classes_]
    unknown_symptoms = [s for s in user_symptoms if s not in mlb.classes_]

    if unknown_symptoms:
        print(Fore.RED + "\nВнимание! " + Style.RESET_ALL +
              f"эти симптомы мне неизвестны: {', '.join(unknown_symptoms)}")

    if not known_symptoms:
        print(Fore.RED + "Нет известных симптомов для анализа!" + Style.RESET_ALL)
        continue

    # Делаем предсказание
    symptoms_vector = mlb.transform([known_symptoms])
    prediction = model.predict(symptoms_vector)
    probabilities = model.predict_proba(symptoms_vector)

    # Показываем результат
    show_results(prediction, probabilities)

    # Спрашиваем продолжить или нет
    answer = input(Fore.CYAN + "\nПродолжим? (да/нет): " + Style.RESET_ALL).lower()
    if answer not in ['да', 'д', 'y', 'yes']:
        print(Fore.YELLOW + "\nПока! Будь здоров!" + Style.RESET_ALL)
        break