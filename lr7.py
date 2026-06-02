import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

df = sns.load_dataset('titanic')
print("Размер:", df.shape)
print("Пропуски:\n", df.isnull().sum())

df.dropna(subset=['age'], inplace=True)
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df.drop(columns=['deck', 'embark_town', 'alive'], inplace=True)

df['pclass'] = df['pclass'].astype(str)
df['survived'] = df['survived'].astype(int)
print("\nПосле обработки:\n", df.isnull().sum())

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('Анализ Titanic — Seaborn', fontsize=14, fontweight='bold')

sns.histplot(data=df, x='age', hue='sex', kde=True, bins=30, ax=axes[0, 0])
axes[0, 0].set_title('Распределение возраста')
axes[0, 0].set_xlabel('Возраст')
axes[0, 0].set_ylabel('Количество')

sns.scatterplot(data=df, x='age', y='fare', hue='sex', alpha=0.6, ax=axes[0, 1])
axes[0, 1].set_title('Возраст vs Стоимость билета')
axes[0, 1].set_xlabel('Возраст')
axes[0, 1].set_ylabel('Стоимость (£)')

sns.boxplot(data=df, x='pclass', y='fare', order=['1', '2', '3'],
            palette='muted', ax=axes[1, 0])
axes[1, 0].set_title('Стоимость билета по классу')
axes[1, 0].set_xlabel('Класс')
axes[1, 0].set_ylabel('Стоимость (£)')
axes[1, 0].set_ylim(0, 300)

corr = df[['age', 'fare', 'sibsp', 'parch', 'survived']].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1, 1])
axes[1, 1].set_title('Корреляции числовых признаков')

plt.tight_layout()
plt.savefig('seaborn_plots.png', dpi=150, bbox_inches='tight')
plt.show()


fig1 = px.scatter(df, x='age', y='fare', color='sex',
                  title='Возраст vs Стоимость билета',
                  labels={'age': 'Возраст', 'fare': 'Стоимость (£)', 'sex': 'Пол'})
fig1.write_html('plot1_scatter.html')
fig1.show()

line_data = df.groupby(['pclass', 'sex'])['age'].mean().reset_index()
fig2 = px.line(line_data, x='pclass', y='age', color='sex', markers=True,
               title='Средний возраст по классу каюты',
               labels={'pclass': 'Класс', 'age': 'Средний возраст', 'sex': 'Пол'})
fig2.write_html('plot2_line.html')
fig2.show()

bar_data = df.groupby('pclass')['survived'].sum().reset_index()
fig3 = px.bar(bar_data, x='pclass', y='survived', text='survived',
              title='Количество выживших по классу',
              labels={'pclass': 'Класс', 'survived': 'Выжило человек'})
fig3.update_traces(textposition='outside')
fig3.write_html('plot3_bar.html')
fig3.show()

pivot = df.pivot_table(values='survived', index='embarked',
                       columns='pclass', aggfunc='mean').round(2)
fig4 = px.imshow(pivot, text_auto=True, color_continuous_scale='RdYlGn',
                 title='Доля выживших: Порт × Класс',
                 labels=dict(x='Класс', y='Порт', color='Доля'))
fig4.write_html('plot4_heatmap.html')
fig4.show()

features = ['age', 'fare', 'sibsp', 'parch']
labels   = ['Возраст', 'Стоимость билета', 'Братья/сёстры', 'Родители/дети']

traces = [go.Histogram(x=df[f], name=lbl, visible=(i == 0), nbinsx=30)
          for i, (f, lbl) in enumerate(zip(features, labels))]

buttons = [dict(label=lbl, method='update',
                args=[{'visible': [j == i for j in range(len(features))]},
                      {'title': f'Распределение: {lbl}', 'xaxis': {'title': lbl}}])
           for i, lbl in enumerate(labels)]

fig5 = go.Figure(data=traces)
fig5.update_layout(
    title='Распределение: Возраст',
    xaxis_title='Значение',
    yaxis_title='Количество',
    updatemenus=[dict(buttons=buttons, direction='down',
                      x=0.01, y=1.15, showactive=True)]
)
fig5.write_html('plot5_dropdown.html')
fig5.show()
