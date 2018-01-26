# coding=utf-8
import pandas
import numpy as np
import datetime
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 1. Считайте таблицу с признаками из файла features.csv с помощью кода, приведенного выше.
train = pandas.read_csv('../data/features.csv', index_col='match_id')
test = pandas.read_csv('../data/features_test.csv', index_col='match_id')

train.head()
# 4. Какой столбец содержит целевую переменную? Запишите его название.
y = train['radiant_win']

# 1. Удалите признаки, связанные с итогами матча (они помечены в описании данных как отсутствующие в тестовой выборке).
train.drop(['duration',
            'radiant_win',
            'tower_status_radiant',
            'tower_status_dire',
            'barracks_status_radiant',
            'barracks_status_dire'
            ], axis=1, inplace=True)

# 2. Проверьте выборку на наличие пропусков с помощью функции count(), которая для каждого столбца показывает число
# заполненных значений. Много ли пропусков в данных? Запишите названия признаков, имеющих пропуски, и попробуйте
# для любых двух из них дать обоснование, почему их значения могут быть пропущены.

rows = len(train)
features_missing = train.count().map(lambda c: rows - c)
# print features_missing[features_missing != 0 ]


# 3. Замените пропуски на нули с помощью функции fillna(). На самом деле этот способ является предпочтительным
# для логистической регрессии, поскольку он позволит пропущенному значению не вносить никакого вклада в предсказание.
# Для деревьев часто лучшим вариантом оказывается замена пропуска на очень большое или очень маленькое значение
# — в этом случае при построении разбиения вершины можно будет отправить объекты с пропусками в отдельную ветвь дерева.
# Также есть и другие подходы — например, замена пропуска на среднее значение признака. Мы не требуем этого в задании,
# но при желании попробуйте разные подходы к обработке пропусков и сравните их между собой.

X_train = train.fillna(0)

# 4. Какой столбец содержит целевую переменную? Запишите его название.
# y = train['radiant_win']

# 5. Забудем, что в выборке есть категориальные признаки, и попробуем обучить градиентный бустинг над деревьями на
# имеющейся матрице "объекты-признаки". Зафиксируйте генератор разбиений для кросс-валидации по 5 блокам (KFold),
# не забудьте перемешать при этом выборку (shuffle=True), поскольку данные в таблице отсортированы по времени,
# и без перемешивания можно столкнуться с нежелательными эффектами при оценивании качества.

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Оцените качество градиентного бустинга (GradientBoostingClassifier) с помощью данной кросс-валидации,
# попробуйте при этом разное количество деревьев (как минимум протестируйте следующие значения для количества
# деревьев: 10, 20, 30). Долго ли настраивались классификаторы?
# Достигнут ли оптимум на испытанных значениях параметра n_estimators, или же качество, скорее всего,
# продолжит расти при дальнейшем его увеличении?

scores = []
# forest = [10, 20, 30, 60, 90]
forest = [10,20]
for trees in forest:
    model = GradientBoostingClassifier(n_estimators=trees, max_depth=3, random_state=42)
    start_time = datetime.datetime.now()
    model_scores = cross_val_score(model, X_train, y, cv=kf, scoring='roc_auc', n_jobs=3)
    print ('trees: ', str(trees),  ' time:', datetime.datetime.now() - start_time, ' data: ', model_scores)
    scores.append(np.mean(model_scores))

print (scores)


# подход 2

# 1. Оцените качество логистической регрессии (sklearn.linear_model.LogisticRegression с L2-регуляризацией)
# с помощью кросс-валидации по той же схеме, которая использовалась для градиентного бустинга.
# Подберите при этом лучший параметр регуляризации (C).
# Какое наилучшее качество у вас получилось? Как оно соотносится с качеством градиентного бустинга?
# Чем вы можете объяснить эту разницу?
# Быстрее ли работает логистическая регрессия по сравнению с градиентным бустингом?

scaler = StandardScaler()
X = scaler.fit_transform(X_train)

kf = KFold(n_splits=5, shuffle=True, random_state=42)


scores = []
CD = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
# CD = [0.00001]
for C in CD:
    model = LogisticRegression(C=C, random_state=42, n_jobs=1)
    start_time = datetime.datetime.now()
    model_scores = cross_val_score(model, X, y, cv=kf, scoring='roc_auc', n_jobs=-1)
    print ('C: ', str(C), ' time:', datetime.datetime.now() - start_time, ' data: ', model_scores)
    scores.append(np.mean(model_scores))

print (scores)


# 2. Среди признаков в выборке есть категориальные, которые мы использовали как числовые, что вряд ли является
# хорошей идеей. Категориальных признаков в этой задаче одиннадцать: lobby_type и r1_hero, r2_hero, ..., r5_hero,
# d1_hero, d2_hero, ..., d5_hero. Уберите их из выборки, и проведите кросс-валидацию для логистической регрессии на
# новой выборке с подбором лучшего параметра регуляризации. Изменилось ли качество? Чем вы можете это объяснить?

X_train.drop(['lobby_type',
        'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
        'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero'
        ], axis=1, inplace=True)


scaler = StandardScaler()
X = scaler.fit_transform(X_train)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = []
CD = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
# CD = [0.00001]
for C in CD:
    model = LogisticRegression(C=C, random_state=42, n_jobs=1)
    start_time = datetime.datetime.now()
    model_scores = cross_val_score(model, X, y, cv=kf, scoring='roc_auc', n_jobs=-1)
    print ('C: ', str(C), ' time:', datetime.datetime.now() - start_time, ' data: ', model_scores)
    scores.append(np.mean(model_scores))

print (scores)

# 3. На предыдущем шаге мы исключили из выборки признаки rM_hero и dM_hero, которые показывают, какие именно герои
# играли за каждую команду. Это важные признаки — герои имеют разные характеристики, и некоторые из них выигрывают чаще,
#  чем другие. Выясните из данных, сколько различных идентификаторов героев существует в данной игре
# (вам может пригодиться фукнция unique или value_counts).

# считываем герроев
heroes = pandas.read_csv('./final/dictionaries/heroes.csv')
print (len(heroes))

# 4. Воспользуемся подходом "мешок слов" для кодирования информации о героях. Пусть всего в игре имеет N различных
# героев. Сформируем N признаков, при этом i-й будет равен нулю, если i-й герой не участвовал в матче; единице,
# если i-й герой играл за команду Radiant; минус единице, если i-й герой играл за команду Dire.
# Ниже вы можете найти код, который выполняет данной преобразование. Добавьте полученные признаки к числовым,
# которые вы использовали во втором пункте данного этапа.
def words_bag(data):
    X_pick = np.zeros((data.shape[0], len(heroes)))
    for i, match_id in enumerate(data.index):
        for p in range(5):
            X_pick[i, data.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
            X_pick[i, data.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1

    return pandas.DataFrame(X_pick, index=data.index)

# возвращаемся к первоначальной выборке
X_train = train.fillna(0)

#
X_hero = words_bag(X_train)

scaler = StandardScaler()
X = pandas.concat([X_train, X_hero], axis=1)
X = scaler.fit_transform(X)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = []
CD = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
for C in CD:
    model = LogisticRegression(C=C, random_state=42, n_jobs=1)
    start_time = datetime.datetime.now()
    model_scores = cross_val_score(model, X, y, cv=kf, scoring='roc_auc', n_jobs=-1)
    print ('C: ', str(C), ' time:', datetime.datetime.now() - start_time, ' data: ', model_scores)
    scores.append(np.mean(model_scores))

print (scores)

model = LogisticRegression(C=0.01, random_state=42, n_jobs=-1)
model.fit(X, y)
