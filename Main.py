import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy

np.random.seed(42)
n_samples = 150

data = {
    'Имя': [f'Клиент_{i}' for i in range(n_samples)],
    'Доход': np.random.randint(40000, 250000, n_samples),
    'Долг_нагрузка': np.round(np.random.uniform(0.0, 0.7, n_samples), 2),
    'Сем_положение': np.random.choice([0, 1], n_samples),
    'Дети': np.random.choice([0, 1, 2, 3], n_samples),
    'Имущество_стоимость': np.random.randint(0, 12000000, n_samples),
    'Квартира': np.random.choice([0, 1], n_samples),
    'Машина': np.random.choice([0, 1], n_samples),
    'Возраст': np.random.randint(21, 65, n_samples),
    'Запрос_сумма': np.random.randint(100000, 3000000, n_samples),
    'Срок_кредита_мес': np.random.choice([12, 24, 36, 60], n_samples),
    'Ставка_процент': np.random.randint(10, 25, n_samples),
    'Поручитель': np.random.choice([0, 1], n_samples)
}

df = pd.DataFrame(data)

# Рассчитываем примерный платеж
r = (df['Ставка_процент'] / 100) / 12
df['Платеж'] = (df['Запрос_сумма'] * r * (1 + r)**df['Срок_кредита_мес']) / ((1 + r)**df['Срок_кредита_мес'] - 1)
df['Ratio'] = df['Платеж'] / df['Доход']

# Условие: платеж < 50% дохода ИЛИ наличие поручителя при долге < 60% ИЛИ очень дорогое имущество
df['Кредит_выдан'] = (
    ((df['Ratio'] < 0.5) & (df['Долг_нагрузка'] < 0.5)) | 
    ((df['Поручитель'] == 1) & (df['Ratio'] < 0.6)) |
    (df['Имущество_стоимость'] > 7000000)
).astype(int)

print(f"Одобрено кредитов: {df['Кредит_выдан'].sum()} из {n_samples}")

# Метод локтя
features_for_clustering = ['Доход', 'Запрос_сумма', 'Долг_нагрузка', 'Имущество_стоимость']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features_for_clustering])

distortions = []
K_range = range(1, 11)
for k in K_range:
    kmeanModel = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeanModel.fit(X_scaled)
    distortions.append(kmeanModel.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K_range, distortions, 'bo-')
plt.xlabel('Количество кластеров')
plt.ylabel('Инерция (Distortion)')
plt.axvline(x=3, color='r', linestyle='--')
plt.title('Метод локтя для определения оптимального K')
plt.show()

# Кластеризация и энтропия
n_clusters = 3 
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df['Группа'] = kmeans.fit_predict(X_scaled)

def calculate_entropy(data):
    counts = data.value_counts(normalize=True)
    return entropy(counts, base=2)

total_entropy = calculate_entropy(df['Кредит_выдан'])
print(f"\nОбщая энтропия системы: {total_entropy:.4f}")

for g in range(n_clusters):
    group_entropy = calculate_entropy(df[df['Группа'] == g]['Кредит_выдан'])
    print(f"Энтропия Группы {g}: {group_entropy:.4f}")

# Дерево принятия решений
features_tree = ['Доход', 'Долг_нагрузка', 'Запрос_сумма', 'Имущество_стоимость', 'Поручитель']
X = df[features_tree]
y = df['Кредит_выдан']

tree_clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
tree_clf.fit(X, y)

plt.figure(figsize=(15, 8))
plot_tree(tree_clf, feature_names=features_tree, class_names=['Отказ', 'Выдача'], filled=True)
plt.title("Древо принятия решения о выдаче кредита")
plt.show()

# Визуализация поверхности
# Создаем сетку (Доход vs Запрос_сумма)
x_range = np.linspace(df['Доход'].min(), df['Доход'].max(), 100)
y_range = np.linspace(df['Запрос_сумма'].min(), df['Запрос_сумма'].max(), 100)
xx, yy = np.meshgrid(x_range, y_range)

# Для предсказания фиксируем остальные признаки средними значениями
grid_data = pd.DataFrame({
    'Доход': xx.ravel(),
    'Долг_нагрузка': df['Долг_нагрузка'].median(),
    'Запрос_сумма': yy.ravel(),
    'Имущество_стоимость': df['Имущество_стоимость'].median(),
    'Поручитель': 1 # Предположим, есть поручитель
})

Z = tree_clf.predict(grid_data[features_tree]).reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlGn')
plt.scatter(df['Доход'], df['Запрос_сумма'], c=df['Кредит_выдан'], edgecolors='k', cmap='RdYlGn')
plt.xlabel('Доход')
plt.ylabel('Сумма кредита')
plt.title('Поверхность принятия решений (Доход vs Сумма)')
plt.show()

# Классификация новых клиентов
new_clients = pd.DataFrame({
    'Имя': ['Иван', 'Мария', 'Петр', 'Анна', 'Олег'],
    'Доход': [150000, 45000, 200000, 60000, 90000],
    'Долг_нагрузка': [0.1, 0.6, 0.4, 0.2, 0.8],
    'Запрос_сумма': [500000, 1000000, 5000000, 300000, 1500000],
    'Имущество_стоимость': [2000000, 0, 10000000, 500000, 0],
    'Поручитель': [1, 0, 1, 1, 0]
})

new_clients['Прогноз_выдачи'] = tree_clf.predict(new_clients[features_tree])
print("\nПрогноз для новых клиентов")
print(new_clients[['Имя', 'Доход', 'Запрос_сумма', 'Прогноз_выдачи']])