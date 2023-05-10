import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

sns.set(context='talk', style='ticks')

st.set_page_config(
     page_title="Análise Exploratória - Previsão de Renda",
     page_icon="./input/icon.png",
     layout="wide",
)


#======================================================================================================

st.write('# Análise exploratória da previsão de renda')

df = pd.read_csv('./input/previsao_de_renda.csv')

#======================================================================================================

st.markdown("#### Escolha o período a ser análisado:")

df['data_ref'] = pd.to_datetime(df['data_ref'])

data_minima = data_minima = df['data_ref'].min()
data_maxima = data_maxima = df['data_ref'].max()

min_selecionado = st.date_input(label = "Data Inicial", min_value=data_minima, max_value=data_maxima, value=data_minima)
max_selecionado = st.date_input(label = "Data Final", min_value=data_minima, max_value=data_maxima, value=data_maxima)

st.write(min_selecionado, max_selecionado)

df = df[ (df['data_ref'] >= pd.to_datetime(min_selecionado)) & (df['data_ref'] <= pd.to_datetime(max_selecionado)) ].copy()


#======================================================================================================


st.markdown("---------------")

st.markdown("## Análise Univariada - Variáveis Binárias")

figura, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(15,10))

sns.countplot(data = df, x = 'sexo', ax = ax1)
sns.countplot(data = df, x = 'posse_de_veiculo', ax = ax2)
sns.countplot(data = df, x = 'posse_de_imovel', ax = ax3)

ax1.set_title('Distribuição da variável "sexo"')
ax2.set_title('Distribuição da variável "posse_de_veiculo"')
ax3.set_title('Distribuição da variável "posse_de_imovel"')

ax1.set_xlabel('')
ax2.set_xlabel('')
ax3.set_xlabel('')

ax1.set_ylabel('Total')
ax2.set_ylabel('Total')
ax3.set_ylabel('Total')

ax4.remove()

st.pyplot(plt)


#======================================================================================================


st.markdown("---------------")
st.markdown("## Análise Univariada - Variáveis Numéricas")

figura, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 15))

sns.boxplot(data = df, x = 'idade', ax = ax1)
sns.boxplot(data = df, x = 'tempo_emprego', ax = ax2)
sns.boxplot(data = df, x = 'qtd_filhos', ax = ax3)
sns.boxplot(data = df, x = 'qt_pessoas_residencia', ax = ax4)

ax1.set_xlabel('')
ax2.set_xlabel('')
ax3.set_xlabel('')
ax4.set_xlabel('')

ax1.set_title('Idade')
ax2.set_title('Tempo de emprego')
ax3.set_title('Quantidade de filhos')
ax4.set_title('Quantidade de pessoas na residência')

figura.subplots_adjust(hspace=0.4, wspace=0.5)

st.pyplot(plt)

st.write(df[ ['idade', 'tempo_emprego', 'qtd_filhos', 'qt_pessoas_residencia'] ].describe().transpose())


#======================================================================================================


st.markdown("---------------")
st.markdown("## Análise Univariada - Variáveis Categóricas")

st.markdown("#### Tipo de renda:")

figura = plt.figure(figsize=(15,5))

ax = sns.countplot(data = df, x = 'tipo_renda')
ax.set_title("Tipo de renda")
ax.set_xlabel("")
ax.set_ylabel("Total")

st.pyplot(plt)

st.write(df['tipo_renda'].value_counts())


#======================================================================================================


st.markdown("---------------")
st.markdown("#### Educação")

figura = plt.figure(figsize=(15,5))

ax = sns.countplot(data = df, x = 'educacao')
ax.set_title("Nível de educação")
ax.set_xlabel("")
ax.set_ylabel("Total")

st.pyplot(plt)

st.write(df['educacao'].value_counts())


#======================================================================================================


st.markdown("---------------")
st.markdown("#### Estado Civil")

figura = plt.figure(figsize=(15,5))

ax = sns.countplot(data = df, x = 'estado_civil')
ax.set_title("Estado civil")
ax.set_xlabel("")
ax.set_ylabel("Total")

st.pyplot(plt)

st.write(df['estado_civil'].value_counts())


#======================================================================================================


st.markdown("---------------")
st.markdown("#### Tipo de Residência")

figura = plt.figure(figsize=(15,5))

ax = sns.countplot(data = df, x = 'tipo_residencia')
ax.set_title("Tipo de residência")
ax.set_xlabel("")
ax.set_ylabel("Total")

st.pyplot(plt)

st.write(df['tipo_residencia'].value_counts())


#======================================================================================================


st.markdown("---------------")
st.markdown("## Análise Bivariada - Variáveis Binárias")

figura, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(15,15))

sns.pointplot(data = df, x = 'data_ref', y = 'renda', hue = 'sexo', ax = ax1)
sns.pointplot(data = df, x = 'data_ref', y = 'renda', hue = 'posse_de_veiculo', ax = ax2)
sns.pointplot(data = df, x = 'data_ref', y = 'renda', hue = 'posse_de_imovel', ax = ax3)

valores_x = pd.to_datetime(df['data_ref']).dt.strftime("%m/%y").unique()

ax1.set_xlabel("")
ax2.set_xlabel("")
ax3.set_xlabel("")

ax1.set_title("Média da renda ao longo dos meses baseado no sexo")
ax2.set_title("Média da renda ao longo dos meses baseado se possuí veículo ou não")
ax3.set_title("Média da renda ao longo dos meses baseado se possuí imóvel ou não")

ax1.set_xticklabels(valores_x)
ax2.set_xticklabels(valores_x)
ax3.set_xticklabels(valores_x)

figura.subplots_adjust(hspace=0.3)

st.pyplot(plt)


#======================================================================================================


st.markdown("---------------")
st.markdown("## Análise Bivariada - Variáveis Numéricas")
st.markdown("#### Matriz de Correlação")

figura = plt.figure(figsize=(15,10))

ax = sns.heatmap(df.select_dtypes('number').corr(),
                annot=True, 
                cmap='coolwarm')

figura.subplots_adjust(hspace=0.5)

st.pyplot(plt)


#======================================================================================================


st.markdown("---------------")
st.markdown("## Análise Bivariada - Variáveis Categóricas")

figura, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(15,10))

sns.barplot(data = df, x = 'tipo_renda', y = 'renda', errorbar=('ci', 95), ax = ax1)
sns.barplot(data = df, x = 'educacao', y = 'renda', errorbar=('ci', 95), ax = ax2)
sns.barplot(data = df, x = 'estado_civil', y = 'renda', errorbar=('ci', 95), ax = ax3)
sns.barplot(data = df, x = 'tipo_residencia', y = 'renda', errorbar=('ci', 95), ax = ax4)


ax1.set_xticklabels(ax1.get_xticklabels(), rotation=20)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30)
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=20)

ax1.set_xlabel("")
ax2.set_xlabel("")
ax3.set_xlabel("")
ax4.set_xlabel("")

ax1.set_title("Renda média por tipo de renda")
ax2.set_title("Renda média por nível de educação")
ax3.set_title("Renda média por estado civil")
ax4.set_title("Renda média por tipo de residência")

figura.subplots_adjust(hspace=1)

st.pyplot(plt)