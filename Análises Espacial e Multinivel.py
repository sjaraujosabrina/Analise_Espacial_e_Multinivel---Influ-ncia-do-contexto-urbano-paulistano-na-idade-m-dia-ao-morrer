# -*- coding: utf-8 -*-

# Análise espacial e multinível da influência do contexto urbano paulistano na idade média ao morrer 
# MBA em Data Science e Analytics USP ESALQ 2025

# Aluna Sabrina Jesus de Araujo

#%% Pacotes

!pip install pandas
!pip install numpy
!pip install matplotlib
!pip install seaborn
!pip install plotly
!pip install scipy
!pip install scikit-learn
!pip install pingouin
!pip install pyshp
!pip install geopandas
!pip install unidecode
!pip install openpyxl

#%% Importando os pacotes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'
import scipy.stats as stats
from scipy.stats import zscore
import pingouin as pg 
import scipy.cluster.hierarchy as sch  # Hierarchical clustering
from scipy.spatial.distance import pdist  # Distâncias entre pontos
from sklearn.cluster import AgglomerativeClustering  # Cluster hierárquico
from sklearn.cluster import KMeans  # K-means clustering
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import shapefile as shp
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import geopandas as gpd
from scipy.stats import mode # Para calcular o cluster dominante
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import unidecode
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt 

#%% Importando os dados
#%% CSV

# Importando os dados

distritos = pd.read_csv("Distritos.csv", sep=";", decimal=".", encoding="latin1")
domicilios_fav = pd.read_csv("Domicílios em Favelas.csv", sep=";", decimal=".", encoding="latin1") 
pop_total = pd.read_csv("População total.csv", sep=";", decimal=".", encoding="latin1") 
pop_preta_parda = pd.read_csv("População Preta e Parda.csv", sep=";", decimal=".", encoding="latin1") 
idade_media = pd.read_csv("Idade média ao morrer.csv", sep=";", decimal=".", encoding="latin1") 
dengue = pd.read_csv("Incidência de Dengue.csv", sep=";", decimal=".", encoding="latin1")
emprego_2014 = pd.read_csv("Emprego e Renda 2014.csv", sep=";", decimal=".", encoding="latin1") 
emprego_2015 = pd.read_csv("Emprego e Renda 2015.csv", sep=";", decimal=".", encoding="latin1") 
emprego_2016 = pd.read_csv("Emprego e Renda 2016.csv", sep=";", decimal=".", encoding="latin1") 
emprego_2017 = pd.read_csv("Emprego e Renda 2017.csv", sep=";", decimal=".", encoding="latin1") 
emprego_2018 = pd.read_csv("Emprego e Renda 2018.csv", sep=";", decimal=".", encoding="latin1") 
emprego_2019 = pd.read_csv("Emprego e Renda 2019.csv", sep=";", decimal=".", encoding="latin1") 
emprego_2020_2023 = pd.read_csv("Emprego e Renda 2020 a 2023.csv", sep=";", decimal=".", encoding="latin1")
oferta_emprego_formal = pd.read_csv("Oferta de emprego Formal.csv", sep=";", decimal=".", encoding="latin1")

############################################### DATA WRANGLING ############################################### 
####################################TRATANDO IDADE MÉDIA AO MORRER ##########################################

print(idade_media)

# Remove colunas sem nome
idade_media = idade_media.loc[:, ~idade_media.columns.str.contains('^Unnamed')]

# Remove linhas totalmente vazias
idade_media = idade_media.dropna(how='all')

# Corrige vírgulas e converte para float nas colunas de ano
# Lista de colunas dos anos
colunas_anos = idade_media.columns[2:-1] 

print(idade_media.columns)

# Lista das colunas dos anos
colunas_anos = ['2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']

# Loop para corrigir vírgulas e transformar em número
for coluna in colunas_anos: idade_media[coluna] = idade_media[coluna].astype(str).str.replace(',', '.').astype(float)

# Visualizar
print(idade_media.head())

# Seleciona apenas as colunas de anos
#anos = list(range(2014, 2024))  # 2014 até 2023
#anos = [str(ano) for ano in anos]  # transformar em string, pois as colunas estão como '2014', '2015', etc.

# Calcula a média linha a linha (cada distrito)
#idade_media["Media_2014_2023"] = idade_media[anos].mean(axis=1)

# Exibe o resultado
#print(idade_media[["Distrito", "Media_2014_2023"]].head())

# Média da média (todos os distritos juntos)
# media_da_media = idade_media["Media_2014_2023"].mean()

# print(media_da_media)

### TRATANDO DOMICÍLIOS EM FAVELAS

print(domicilios_fav)

# Definindo as colunas dos anos
colunas_anos = ['2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']

# Corrigindo dados: substitui '-' por 0, troca vírgula por ponto, remove pontos de milhar e converte
for coluna in colunas_anos:
    domicilios_fav[coluna] = (
        domicilios_fav[coluna]
        .astype(str)
        .str.replace('-', '0')
        .str.replace(',', '.')  # Só troca decimal
        .astype(float)  # Mantém número com decimal correto
    )


print(domicilios_fav)

print(domicilios_fav.dtypes)

print(domicilios_fav.describe())

### TRATANDO EMPREGOS 2014

print (emprego_2014)

print(emprego_2014.columns)


# Remove a coluna desnecessária
emprego_2014 = emprego_2014.drop(columns=['Faixa de Rendimento em Salários Mínimos'])

# Renomeia as colunas
emprego_2014.columns = [
    'Distrito',
    'Até 0,5 SM',
    '0,51 a 1 SM',
    '1,01 a 1,5 SM',
    '1,51 a 2 SM',
    '2,01 a 3 SM',
    '3,01 a 4 SM',
    '4,01 a 5 SM',
    '5,01 a 7 SM',
    '7,01 a 10 SM',
    '10,01 a 15 SM',
    '15,01 a 20 SM',
    'Mais de 20 SM',
    'Ignorado'
]


# Remover a linha com NaN na coluna 'Distrito'
emprego_2014 = emprego_2014[emprego_2014['Distrito'].notna()].reset_index(drop=True)

# Lista de colunas numéricas
colunas_numericas = emprego_2014.columns[1:]  # Todas, exceto 'Distrito'

# Tratamento dos dados numéricos (remove pontos dos milhares e converte)
for coluna in colunas_numericas:
    emprego_2014[coluna] = (
        emprego_2014[coluna]
        .astype(str)
        .str.replace('.', '', regex=False)  # Remove pontos dos milhares
        .str.replace(',', '.', regex=False)  # Caso tenha vírgulas como separador decimal (por segurança)
        .astype(float)  # Converte para número float
    )

# Visualização final
print(emprego_2014.head())
print(emprego_2014.dtypes)

### TRATANDO EMPREGOS 2015

print (emprego_2015)

print(emprego_2015.columns.tolist())

# Remove a linha onde a coluna 'Unidades Territoriais' é NaN
emprego_2015 = emprego_2015[~emprego_2015['Unidades Territoriais'].isna()]

# Remove a coluna desnecessária
emprego_2015 = emprego_2015.drop(columns=['Faixa de Rendimento em Salários Mínimos 2015'])


# Renomeia as colunas
emprego_2015.columns = [
    'Distrito',
    'Até 0,5 SM',
    '0,51 a 1 SM',
    '1,01 a 1,5 SM',
    '1,51 a 2 SM',
    '2,01 a 3 SM',
    '3,01 a 4 SM',
    '4,01 a 5 SM',
    '5,01 a 7 SM',
    '7,01 a 10 SM',
    '10,01 a 15 SM',
    '15,01 a 20 SM',
    'Mais de 20 SM',
    'Ignorado'
]

# Tratamento 
colunas_numericas = emprego_2015.columns.drop('Distrito')

for coluna in colunas_numericas:
    emprego_2015[coluna] = (
        emprego_2015[coluna]
        .astype(str)
        .astype(float)  # Converte para float
    )

print(emprego_2015.head())
print(emprego_2015.dtypes)

### TRATANDO EMPREGOS 2016

print(emprego_2016)

print(emprego_2016.columns.tolist())

# Remove a coluna desnecessária
emprego_2016 = emprego_2016.drop(columns=['Faixa de Rendimento em Salários Mínimos'])

# Remove a linha onde a coluna 'Unidades Territoriais' é NaN
emprego_2016 = emprego_2016[~emprego_2016['Unidades Territoriais'].isna()]

# Renomeando as colunas
emprego_2016.columns = [
    'Distrito',
    'Até 0,5 SM',
    '0,51 a 1 SM',
    '1,01 a 1,5 SM',
    '1,51 a 2 SM',
    '2,01 a 3 SM',
    '3,01 a 4 SM',
    '4,01 a 5 SM',
    '5,01 a 7 SM',
    '7,01 a 10 SM',
    '10,01 a 15 SM',
    '15,01 a 20 SM',
    'Mais de 20 SM',
    'Ignorado',
]

colunas_renda = emprego_2016.columns[1:]  # Todas as colunas menos 'Distrito'

for coluna in colunas_renda:
    emprego_2016[coluna] = (
        emprego_2016[coluna]
        .astype(str)
        .str.replace('.', '', regex=False)  # Remove pontos dos milhares
        .str.replace(',', '.', regex=False) # Caso tenha vírgulas
        .astype(float)  # Converte para float
    )

# Verificar o resultado
print(emprego_2016.head())
print(emprego_2016.dtypes)

### TRATANDO EMPREGOS 2017

print(emprego_2017)

print(emprego_2017.columns.tolist())

# Remove a coluna desnecessária
emprego_2017 = emprego_2017.drop(columns=['Faixa de Rendimento em Salários Mínimos 2017'])

# Remove a linha onde a coluna 'Unidades Territoriais' é NaN
emprego_2017 = emprego_2017[~emprego_2017['Unidades Territoriais'].isna()]

# Renomeando as colunas
emprego_2017.columns = [
    'Distrito',
    'Até 0,5 SM',
    '0,51 a 1 SM',
    '1,01 a 1,5 SM',
    '1,51 a 2 SM',
    '2,01 a 3 SM',
    '3,01 a 4 SM',
    '4,01 a 5 SM',
    '5,01 a 7 SM',
    '7,01 a 10 SM',
    '10,01 a 15 SM',
    '15,01 a 20 SM',
    'Mais de 20 SM',
    'Ignorado',
]

colunas_renda = emprego_2017.columns[1:]  # Todas as colunas menos 'Distrito'

for coluna in colunas_renda:
    emprego_2017[coluna] = (
        emprego_2017[coluna]
        .astype(str)
        .str.replace('.', '', regex=False)  # Remove pontos dos milhares
        .str.replace(',', '.', regex=False) # Caso tenha vírgulas
        .astype(float)  # Converte para float
    )

# Verificar o resultado
print(emprego_2017.head())
print(emprego_2017.dtypes)

### TRATANDO EMPREGOS 2018

print(emprego_2018)

print(emprego_2018.columns.tolist())

# Remove a coluna desnecessária
emprego_2018 = emprego_2018.drop(columns=['Faixa de Rendimento em Salários Mínimos'])

# Remove a linha onde a coluna 'Unidades Territoriais' é NaN
emprego_2018 = emprego_2018[~emprego_2018['Unidades Territoriais'].isna()]

# Renomeando as colunas
emprego_2018.columns = [
    'Distrito',
    'Até 0,5 SM',
    '0,51 a 1 SM',
    '1,01 a 1,5 SM',
    '1,51 a 2 SM',
    '2,01 a 3 SM',
    '3,01 a 4 SM',
    '4,01 a 5 SM',
    '5,01 a 7 SM',
    '7,01 a 10 SM',
    '10,01 a 15 SM',
    '15,01 a 20 SM',
    'Mais de 20 SM',
    'Ignorado',
]

colunas_renda = emprego_2018.columns[1:]  # Todas as colunas menos 'Distrito'

for coluna in colunas_renda:
    emprego_2018[coluna] = (
        emprego_2018[coluna]  # <- Aqui estava errado, você puxou emprego_2017
        .astype(str)
        .str.replace('.', '', regex=False)  # Remove pontos dos milhares
        .str.replace(',', '.', regex=False) # Troca vírgula por ponto decimal
        .astype(float)  # Converte para float
    )


# Verificar o resultado
print(emprego_2018.head())
print(emprego_2018.dtypes)

### TRATANDO EMPREGOS 2019

print(emprego_2019)

print(emprego_2019.columns.tolist())

# Remove a coluna desnecessária
emprego_2019 = emprego_2019.drop(columns=['Faixa de Rendimento em Salários Mínimos'])

# Remove a linha onde a coluna 'Unidades Territoriais' é NaN
emprego_2019 = emprego_2019[~emprego_2019['Unidades Territoriais'].isna()]

# Renomeando as colunas
emprego_2019.columns = [
    'Distrito',
    'Até 0,5 SM',
    '0,51 a 1 SM',
    '1,01 a 1,5 SM',
    '1,51 a 2 SM',
    '2,01 a 3 SM',
    '3,01 a 4 SM',
    '4,01 a 5 SM',
    '5,01 a 7 SM',
    '7,01 a 10 SM',
    '10,01 a 15 SM',
    '15,01 a 20 SM',
    'Mais de 20 SM',
    'Ignorado',
]

# Tratamento dos dados numéricos
colunas_renda = emprego_2019.columns[1:]  # Todas as colunas menos 'Distrito'

for coluna in colunas_renda:
    emprego_2019[coluna] = (
        emprego_2019[coluna]
        .astype(str)
        .str.replace('.', '', regex=False)  # Remove pontos dos milhares
        .str.replace(',', '.', regex=False) # Troca vírgula por ponto decimal
        .astype(float)  # Converte para float
    )

# Verificar o resultado
print(emprego_2019.head())
print(emprego_2019.dtypes)

### TRATANDO EMPREGOS 2020 a 2023

print(emprego_2020_2023)

print(emprego_2020_2023.columns.tolist())

# Remove a coluna desnecessária
emprego_2020_2023 = emprego_2020_2023.drop(columns=['Faixa de Rendimento em Salários Mínimos'])

# Remove a linha onde a coluna 'Unidades Territoriais' é NaN
emprego_2020_2023 = emprego_2020_2023[~emprego_2020_2023['Unidades Territoriais'].isna()]

# Renomeando as colunas
emprego_2020_2023.columns = [
    'Distrito',
    'Até 0,5 SM',
    '0,51 a 1 SM',
    '1,01 a 1,5 SM',
    '1,51 a 2 SM',
    '2,01 a 3 SM',
    '3,01 a 4 SM',
    '4,01 a 5 SM',
    '5,01 a 7 SM',
    '7,01 a 10 SM',
    '10,01 a 15 SM',
    '15,01 a 20 SM',
    'Mais de 20 SM',
    'Ignorado',
]

# Tratamento dos dados numéricos
colunas_renda = emprego_2020_2023.columns[1:]  # Todas as colunas menos 'Distrito'

for coluna in colunas_renda:
    emprego_2020_2023[coluna] = (
        emprego_2020_2023[coluna]
        .astype(str)
        .str.replace('.', '', regex=False)  # Remove pontos dos milhares
        .str.replace(',', '.', regex=False) # Troca vírgula por ponto decimal
        .astype(float)  # Converte para float
    )

# Verificar o resultado
print(emprego_2020_2023.head())
print(emprego_2020_2023.dtypes)

### TRATANDO POPULAÇÃO TOTAL

print(pop_total)
print(pop_total.head())
print(pop_total.dtypes)

# Remove colunas desnecessárias
pop_total = pop_total.drop(columns=['Dado', 'Unnamed: 12'])

# Remove linhas onde o distrito está vazio (NaN)
pop_total = pop_total[~pop_total['DISTRITO'].isna()]

# Reseta o índice para manter organizado
pop_total = pop_total.reset_index(drop=True)


### TRATANDO POPULAÇÃO PRETA E PARDA

print(pop_preta_parda)

# Corrigir vírgula para ponto e transformar em float nas colunas de ano
colunas_ano = pop_preta_parda.columns[1:]  # Todas menos 'DISTRITO'

for coluna in colunas_ano:
    pop_preta_parda[coluna] = (
        pop_preta_parda[coluna]
        .astype(str)
        .str.replace(',', '.', regex=False)
        .astype(float)
    )

# Verificar se os tipos estão corretos
print(pop_preta_parda.dtypes)
print(pop_preta_parda.head())

### TRATANDO OFERTA DE EMPREGO FORMAL

print(oferta_emprego_formal)
print(oferta_emprego_formal.dtypes)
print(oferta_emprego_formal.head())

# Sai linha com NaN
oferta_emprego_formal = oferta_emprego_formal.dropna(how='all')

# Remove coluna 'Dado'
oferta_emprego_formal = oferta_emprego_formal.drop(columns=['Dado'])

# Altera coluna 'Distritos' para 'Distrito'
oferta_emprego_formal = oferta_emprego_formal.rename(columns={'Distritos': 'Distrito'})

# Garante que os dados dos anos estejam no tipo numérico
anos = [str(ano) for ano in range(2014, 2024)]
oferta_emprego_formal[anos] = oferta_emprego_formal[anos].apply(pd.to_numeric, errors='coerce')

# Confere se há valores nulos
print(oferta_emprego_formal.isnull().sum())

#Visualização geral
print(oferta_emprego_formal.dtypes)
print(oferta_emprego_formal.head())


### VISUALIZANDO DADOS SOBRE DENGUE

print(dengue)
print(dengue.dtypes)
print(dengue.head())


### Obtendo número total de pessoas pretas e pardas nos distritos, para não haver ponderação arbitrária com a porcentagem

print(pop_total)
print(pop_preta_parda)

print(pop_total.columns)
print(pop_preta_parda.columns)

# Lista dos anos
anos = [str(ano) for ano in range(2014, 2024)]

# Merge dos dois dataframes, adicionando sufixos para diferenciar
df_merge = pop_preta_parda.merge(pop_total, on='DISTRITO', suffixes=('_perc', '_total'))

# Verificando se o merge foi correto
print(df_merge.columns)

# Criando as colunas de número absoluto de pessoas pretas e pardas
for ano in anos:
    perc_col = f'{ano}_perc'
    total_col = f'{ano}_total'
    df_merge[f'n_preta_parda_{ano}'] = (df_merge[perc_col] / 100) * df_merge[total_col]

# Selecionano apenas as colunas de interesse para o resultado
colunas_resultado = ['DISTRITO'] + [f'n_preta_parda_{ano}' for ano in anos]
resultado = df_merge[colunas_resultado]

# Verificando o resultado
print(resultado.head())

# Exportar Excel
resultado.to_excel('pop_preta_parda_absoluto.xlsx', index=False)

# Abrindo planilha (converti para CSV)

pop_total_preta_parda = pd.read_csv("População total preta parda.csv", sep=";", decimal=".", encoding="latin1")

# Visualizando 
print(pop_total_preta_parda)

# Tratando os dados

pop_total_preta_parda = pd.read_csv('População total preta parda.csv', sep=';', decimal=',', encoding='latin1')


print(pop_total_preta_parda.dtypes)
print(pop_total_preta_parda.head())


##### DATA WRANGLING ##### RENOMEANDO

print(pop_total_preta_parda) #Moóca
print(oferta_emprego_formal)
print(domicilios_fav)
print(dengue) #Moóca
print(idade_media) #Moóca


# Renomeando colunas
pop_total_preta_parda = pop_total_preta_parda.rename(columns={'Distrito': 'Distrito'})
oferta_emprego_formal = oferta_emprego_formal.rename(columns={'Distrito': 'Distrito'})
domicilios_fav = domicilios_fav.rename(columns={'Distritos': 'Distrito'})
dengue = dengue.rename(columns={'Distrito': 'Distrito'})
idade_media = idade_media.rename(columns={'DISTRITO': 'Distrito'})

pop_long = pop_total_preta_parda.melt(
    id_vars='Distrito',
    var_name='Ano',
    value_name='Pop_preta_parda'
)

# Mudando nome de Moóca para Mooca
pop_total_preta_parda['Distrito'] = pop_total_preta_parda['Distrito'].replace('Moóca', 'Mooca')

dengue['Distrito'] = dengue['Distrito'].replace('Moóca', 'Mooca')

idade_media['Distrito'] = idade_media['Distrito'].replace('Moóca', 'Mooca')

# Ajustar coluna Ano, removendo prefixo 'n_preta_parda_'
pop_long['Ano'] = pop_long['Ano'].str.replace('n_preta_parda_', '').astype(int)


# Oferta de emprego formal
oferta_long = oferta_emprego_formal.melt(id_vars='Distrito', var_name='Ano', value_name='Oferta_emprego')
oferta_long['Ano'] = oferta_long['Ano'].astype(int)

# Domicílios em favelas
domicilios_long = domicilios_fav.melt(id_vars='Distrito', var_name='Ano', value_name='Domicilios_fav')
domicilios_long['Ano'] = domicilios_long['Ano'].astype(int)

# Dengue
dengue_long = dengue.melt(id_vars='Distrito', var_name='Ano', value_name='Casos_dengue')
dengue_long['Ano'] = dengue_long['Ano'].astype(int)

# Idade média
idade_long = idade_media.melt(id_vars=['Distrito', 'Dado'], var_name='Ano', value_name='Idade_media')
idade_long['Ano'] = idade_long['Ano'].astype(int)

# Como só tem um tipo de dado ("Idade média ao morrer") na coluna 'Dado', pode remover essa coluna depois:
idade_long = idade_long.drop(columns='Dado')

# Unir todos usando merges 
dados_quanti_ = pop_long.merge(oferta_long, on=['Distrito', 'Ano'], how='outer')
dados_quanti_ = dados_quanti_.merge(domicilios_long, on=['Distrito', 'Ano'], how='outer')
dados_quanti_ = dados_quanti_.merge(dengue_long, on=['Distrito', 'Ano'], how='outer')
dados_quanti_ = dados_quanti_.merge(idade_long, on=['Distrito', 'Ano'], how='outer')

print(dados_quanti_.head())
print(dados_quanti_.info())

# Verificar quais distritos e anos estão presentes
print(dados_quanti_['Ano'].unique())
print(dados_quanti_['Distrito'].nunique())  # Deve ser 96
print(dados_quanti_['Distrito'].unique())   # Ver nomes e se há erros de digitação


# Remover espaços antes ou depois dos nomes
dados_quanti_['Distrito'] = dados_quanti_['Distrito'].str.strip()

# Verificar onde estão os nulos
print(dados_quanti_[dados_quanti_.isnull().any(axis=1)])

dados_quanti_['Distrito'] = dados_quanti_['Distrito'].apply(
    lambda x: unidecode.unidecode(x.strip()).title()
)

# Conferir se agora está 96
print(dados_quanti_['Distrito'].nunique())

print(dados_quanti_.columns)

colunas = [c for c in dados_quanti_.columns if c not in ['Distrito', 'Ano']]

from sklearn.preprocessing import StandardScaler
import pandas as pd

scaler = StandardScaler()

variaveis_norm = pd.DataFrame(
    scaler.fit_transform(dados_quanti_[colunas]),
    columns=colunas,
    index=dados_quanti_.index  # mantém o mesmo index do DataFrame original
)

variaveis_norm = pd.DataFrame(
    scaler.fit_transform(dados_quanti_[colunas]),
    columns=colunas,
    index=dados_quanti_['Distrito']
)

# seleciona as variáveis
variaveis_cluster = dados_quanti_[['Pop_preta_parda', 'Oferta_emprego', 'Domicilios_fav', 'Casos_dengue', 'Idade_media']]

# imputação com ZERO
imputer = SimpleImputer(strategy='constant', fill_value=0)
variaveis_imputadas = imputer.fit_transform(variaveis_cluster)

# normalização
scaler = StandardScaler()
variaveis_norm = scaler.fit_transform(variaveis_imputadas)

# método do cotovelo (WCSS)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(variaveis_norm)
    wcss.append(kmeans.inertia_)


#############################################################################################################
############################################## CLUSTERIZAÇÃO ################################################
#############################################################################################################

# Plotando o gráfico do cotovelo
plt.figure(figsize=(8,5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# Aplicando K-Means com 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
dados_quanti_['Cluster'] = kmeans.fit_predict(variaveis_norm)

# Estatísticas médias por cluster
agrupado = dados_quanti_.groupby('Cluster').mean(numeric_only=True)
print(agrupado)


plt.figure(figsize=(8,6))
sns.scatterplot(
    x=dados_quanti_['Idade_media'],
    y=dados_quanti_['Pop_preta_parda'],
    hue=dados_quanti_['Cluster'],
    palette='viridis'
)
plt.title('Distribuição dos clusters')
plt.xlabel('Idade Média')
plt.ylabel('População Preta e Parda (%)')
plt.grid(True)
plt.show()


# Re-executando o K-Means com 5 clusters
kmeans_5 = KMeans(n_clusters=5, random_state=42, n_init=10) # n_init=10 para garantir boa inicialização
dados_quanti_['Cluster_5'] = kmeans_5.fit_predict(variaveis_norm)

# Estatísticas médias por cluster para 5 clusters
agrupado_5_clusters = dados_quanti_.groupby('Cluster_5').mean(numeric_only=True)
print("\n--- Estatísticas Médias por Cluster (5 Clusters) ---")
print(agrupado_5_clusters)

# Estatísticas de desvio padrão por cluster para ter uma noção da variabilidade
agrupado_5_std = dados_quanti_.groupby('Cluster_5').std(numeric_only=True)
print("\n--- Desvio Padrão por Cluster (5 Clusters) ---")
print(agrupado_5_std)

# Visualizando a distribuição dos clusters com 5 grupos
# Podemos manter o mesmo gráfico de dispersão inicial para comparação
plt.figure(figsize=(10, 7))
sns.scatterplot(
    x=dados_quanti_['Idade_media'],
    y=dados_quanti_['Pop_preta_parda'],
    hue=dados_quanti_['Cluster_5'], # Usando a nova coluna de clusters
    palette='flare', 
    s=70, # Tamanho dos pontos
    alpha=0.7 # Transparência
)
plt.title('Distribuição dos Clusters (5 Grupos)')
plt.xlabel('Idade Média ao Morrer')
plt.ylabel('População Preta e Parda (Absoluto)') # Ajustei o label, pois você calculou o N absoluto
plt.grid(True)
plt.show()


# PairPlot

sns.pairplot(dados_quanti_, vars=variaveis_cluster.columns, hue='Cluster_5', palette='viridis', plot_kws={'alpha': 0.6})
plt.suptitle('Pair Plot das Variáveis por Cluster (5 Grupos)', y=1.02) 
plt.show()

# Aplicando K-Means com 4 clusters (re-executando para garantir a coluna correta)
kmeans_4 = KMeans(n_clusters=4, random_state=42, n_init=10)
dados_quanti_['Cluster'] = kmeans_4.fit_predict(variaveis_norm)

# Gerando o Pair Plot para os 4 clusters
sns.pairplot(dados_quanti_, vars=variaveis_cluster.columns, hue='Cluster', palette='viridis', plot_kws={'alpha': 0.6})
plt.suptitle('Pair Plot das Variáveis por Cluster (4 Grupos)', y=1.02)
plt.show()

# Opcional: Para relembrar as médias dos 4 clusters, você pode rodar novamente:
print("\n--- Estatísticas Médias por Cluster (4 Clusters) ---")
agrupado_4_clusters = dados_quanti_.groupby('Cluster').mean(numeric_only=True)
print(agrupado_4_clusters)

# EXPLORANDO A CLUSTERIZAÇÃO HIERÁRQUICA

print("\n--- Iniciando Clusterização Hierárquica ---")

# Calculando as distâncias
# Usamos a distância euclidiana, que é o padrão para 'pdist'
distancias = pdist(variaveis_norm, metric='euclidean')

# Realizando o linkage (ligação)
# O método 'ward' minimiza a variância dos clusters que estão sendo unidos.
linkage_matrix = sch.linkage(distancias, method='ward')

# Gerando o Dendrograma
plt.figure(figsize=(15, 8))
sch.dendrogram(
    linkage_matrix,
    truncate_mode='lastp',  # Mostra apenas os últimos p clusters formados
    p=50,                   # Número de folhas a serem exibidas (ajuste conforme necessário)
    leaf_rotation=90.,      # Rotação dos rótulos das folhas
    leaf_font_size=8.,      # Tamanho da fonte dos rótulos das folhas
    show_contracted=True    # Mostra linhas tracejadas para nós contraídos
)
plt.title('Dendrograma para Clusterização Hierárquica')
plt.xlabel('Índice da Amostra ou (Tamanho do Cluster)')
plt.ylabel('Distância Euclidiana')
plt.axhline(y=20, color='r', linestyle='--', label='Corte Exemplo (Distância)') # Linha para ajudar a visualizar o corte
plt.legend()
plt.show()

# Análise fatorial

pca = PCA()
pca.fit(variaveis_norm)

# Analisando a variância explicada
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

print("Variância explicada por cada componente principal:")
for i, var in enumerate(explained_variance_ratio):
    print(f"Componente {i+1}: {var:.4f}")

print("\nVariância acumulada explicada:")
for i, cum_var in enumerate(cumulative_explained_variance):
    print(f"Até Componente {i+1}: {cum_var:.4f}")



# Aplicando PCA com 4 componentes
n_components_to_retain = 4
pca = PCA(n_components=n_components_to_retain)
principal_components = pca.fit_transform(variaveis_norm)

# Criando um DataFrame com os componentes principais para facilitar a visualização e manipulação
df_pca = pd.DataFrame(data=principal_components,
                      columns=[f'PC{i+1}' for i in range(n_components_to_retain)])


dados_quanti_pca = dados_quanti_.copy() # Criando uma cópia para não alterar o original
for i in range(n_components_to_retain):
    dados_quanti_pca[f'PC{i+1}'] = principal_components[:, i]


# Reaplicando o Método do Cotovelo nos Componentes Principais
wcss_pca = []
for i in range(1, 11):
    kmeans_pca = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans_pca.fit(principal_components)
    wcss_pca.append(kmeans_pca.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss_pca, marker='o')
plt.title('Método do Cotovelo (com PCA - 4 Componentes)')
plt.xlabel('Número de clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# Aplicando K-Means nos Componentes Principais
# Com base no novo gráfico do cotovelo, você pode ajustar o número de clusters.
# Por enquanto, vamos manter 4 clusters como um ponto de partida para comparação.
n_clusters_final = 4 # Ou o número que você decidir após ver o novo cotovelo

kmeans_final = KMeans(n_clusters=n_clusters_final, random_state=42, n_init=10)
dados_quanti_pca['Cluster_PCA'] = kmeans_final.fit_predict(principal_components)

# Estatísticas médias por cluster (usando as variáveis ORIGINAIS, mas agrupado pelos clusters da PCA)
# Isso é crucial para interpretar os clusters no contexto dos seus dados originais
print(f"\n--- Estatísticas Médias das Variáveis Originais por Cluster (K-Means em {n_components_to_retain} PCs, {n_clusters_final} Clusters) ---")
agrupado_pca_originais = dados_quanti_pca.groupby('Cluster_PCA')[variaveis_cluster.columns].mean()
print(agrupado_pca_originais)

# Estatísticas médias dos Componentes Principais por cluster
print(f"\n--- Estatísticas Médias dos Componentes Principais por Cluster (K-Means em {n_components_to_retain} PCs, {n_clusters_final} Clusters) ---")
agrupado_pca_pcs = dados_quanti_pca.groupby('Cluster_PCA')[[f'PC{i+1}' for i in range(n_components_to_retain)]].mean()
print(agrupado_pca_pcs)

# Visualização da distribuição dos clusters nos primeiros 2 PCs (se n_components_to_retain >= 2)
if n_components_to_retain >= 2:
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x=dados_quanti_pca['PC1'],
        y=dados_quanti_pca['PC2'],
        hue=dados_quanti_pca['Cluster_PCA'],
        palette='viridis',
        s=70,
        alpha=0.7
    )
    plt.title(f'Distribuição dos Clusters (K-Means nos PCs - {n_clusters_final} Grupos)')
    plt.xlabel('Primeiro Componente Principal (PC1)')
    plt.ylabel('Segundo Componente Principal (PC2)')
    plt.grid(True)
    plt.show()
    
###### APLICANDO DADOS EM UM MAPA ######


import geopandas as gpd
import matplotlib.pyplot as plt

# Caminho para o shapefile
caminho_shp = r"C:\Users\sabri\OneDrive\Documentos\Análise espacial e multinível\Projeto Spyder - Analise multinivel\DEINFO_DISTRITO.shp"

# Carregar o shapefile
gdf_distritos = gpd.read_file(caminho_shp)

# Exibir primeiras linhas e colunas
print("Primeiras linhas do GeoDataFrame dos distritos:")
print(gdf_distritos.head())

print("\nColunas disponíveis no GeoDataFrame dos distritos:")
print(gdf_distritos.columns.tolist())

# Função para limpar e padronizar nomes de distritos
def limpar_nome_distrito(df_col):
    return (
        df_col.astype(str)
        .str.normalize('NFKD')   # Remove acentos
        .str.encode('ascii', errors='ignore')
        .str.decode('utf-8')
        .str.upper()             # Letras maiúsculas
        .str.strip()             # Remove espaços no início/fim
    )

# Aplicar limpeza nos nomes do shapefile
gdf_distritos['NOME_DIST_LIMPO'] = limpar_nome_distrito(gdf_distritos['NOME_DIST'])

# Correções de nomes inconsistentes
correcoes_distritos = {
    'JD SAO LUIS': 'JARDIM SAO LUIS',
    'JD ANGELA': 'JARDIM ANGELA',
    'JD PAULISTA': 'JARDIM PAULISTA',
    'JD HELENA': 'JARDIM HELENA',
    'JD CURUCA': 'VILA CURUCA',
    'SANTA CECILIA': 'SANTA CECILIA',
    'REPUBLICA': 'REPUBLICA',
    'CID ADEMAR': 'CIDADE ADEMAR',
    'CID DUTRA': 'CIDADE DUTRA',
    'CID LIDER': 'CIDADE LIDER',
    'CID TIRADENTES': 'CIDADE TIRADENTES'
}

# Aplicar correção
gdf_distritos['NOME_DIST_CORRIGIDO'] = gdf_distritos['NOME_DIST_LIMPO'].replace(correcoes_distritos)

# Verificando resultado
print(gdf_distritos[['NOME_DIST', 'NOME_DIST_LIMPO', 'NOME_DIST_CORRIGIDO']].drop_duplicates())

# --- Preparar os dados quantitativos ---
# Supondo que o DataFrame seja 'dados_quanti_pca'
# Limpar nomes de distritos nos dados
dados_quanti_pca['Distrito_LIMPO'] = limpar_nome_distrito(dados_quanti_pca['Distrito'])

# Conferir nomes ausentes em cada lado
print("\nDistritos no shapefile, mas não nos dados quantitativos:")
print(set(gdf_distritos['NOME_DIST_CORRIGIDO']) - set(dados_quanti_pca['Distrito_LIMPO']))

print("\nDistritos nos dados quantitativos, mas não no shapefile:")
print(set(dados_quanti_pca['Distrito_LIMPO']) - set(gdf_distritos['NOME_DIST_CORRIGIDO']))


# Exemplo de agregação por distrito (média dos valores numéricos)
dados_distritos_agregados = dados_quanti_pca.groupby('Distrito_LIMPO').agg({
    'Pop_preta_parda': 'mean',
    'Oferta_emprego': 'mean',
    'Domicilios_fav': 'mean',
    'Casos_dengue': 'sum',       # soma, pois faz sentido somar casos
    'Idade_media': 'mean',
}).reset_index()

# Conferir resultado
print(dados_distritos_agregados.head())

mapa_dados = gdf_distritos.merge(
    dados_distritos_agregados,
    left_on='NOME_DIST_CORRIGIDO',
    right_on='Distrito_LIMPO',
    how='left'
)

print(mapa_dados[['NOME_DIST_CORRIGIDO', 'Pop_preta_parda', 'Casos_dengue']].head())

fig, ax = plt.subplots(1, 1, figsize=(15, 12))
mapa_dados.plot(
    column='Pop_preta_parda',
    cmap='viridis',
    linewidth=0.8,
    ax=ax,
    edgecolor='0.8',
    legend=True,
    missing_kwds={
        "color": "lightgrey",
        "edgecolor": "red",
        "hatch": "///",
        "label": "Sem dados",
    }
)
ax.set_axis_off()
plt.title('Distribuição de População Preta/Parda por Distrito em São Paulo', fontsize=16)
plt.show()

########## MAPA DE CLUSTERS

#Selecionar variáveis para clustering
variaveis_cluster = dados_distritos_agregados[['Pop_preta_parda', 'Oferta_emprego', 'Domicilios_fav', 'Casos_dengue', 'Idade_media']]

#Normalizar
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
variaveis_norm = scaler.fit_transform(variaveis_cluster)

#Definir número de clusters
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
dados_distritos_agregados['Cluster_Dominante_PCA'] = kmeans.fit_predict(variaveis_norm)

#Merge com shapefile
mapa_dados = gdf_distritos.merge(
    dados_distritos_agregados,
    left_on='NOME_DIST_CORRIGIDO',
    right_on='Distrito_LIMPO',
    how='left'
)

#Plot do mapa
mapa_dados['Cluster_Dominante_PCA'] = mapa_dados['Cluster_Dominante_PCA'].astype(int)
clusters_unicos = sorted(mapa_dados['Cluster_Dominante_PCA'].dropna().unique())

cmap = plt.cm.viridis
norm = mpl.colors.Normalize(vmin=min(clusters_unicos), vmax=max(clusters_unicos))

fig, ax = plt.subplots(1, 1, figsize=(15, 12))
mapa_dados.plot(
    column='Cluster_Dominante_PCA',
    cmap=cmap,
    linewidth=0.8,
    ax=ax,
    edgecolor='0.8',
    legend=False,
    missing_kwds={
        "color": "lightgrey",
        "edgecolor": "red",
        "hatch": "///",
        "label": "Sem dados",
    }
)

# Barra de cores
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.025, pad=0.02)
cbar.set_label('Clusters (Códigos)', fontsize=12)

ax.set_axis_off()
plt.title('Clusters Dominantes dos Distritos de São Paulo', fontsize=16)
plt.show()


###### MAPA DA IDADE MÉDIA AO MORRER POR DISTRITO ######

def plot_mapa(mapa, coluna, titulo, cmap="viridis"):
    """
    Função para plotar mapas temáticos.
    
    mapa   -> GeoDataFrame já com merge
    coluna -> variável numérica para plotar
    titulo -> título do mapa
    cmap   -> colormap (ex: 'viridis', 'plasma', 'coolwarm')
    """
    fig, ax = plt.subplots(1, 1, figsize=(15, 12))
    
    mapa.plot(
        column=coluna,
        cmap=cmap,
        linewidth=0.8,
        ax=ax,
        edgecolor="0.8",
        legend=True,
        missing_kwds={
            "color": "lightgrey",
            "edgecolor": "red",
            "hatch": "///",
            "label": "Sem dados",
        }
    )
    
    ax.set_axis_off()
    plt.title(titulo, fontsize=16)
    plt.show()

###### MAPA DA IDADE MÉDIA AO MORRER POR DISTRITO ######
plot_mapa(
    mapa=mapa_dados,
    coluna="Idade_media",
    titulo="Distribuição da Idade Média ao Morrer por Distrito em São Paulo"
)


############

import matplotlib.pyplot as plt

###### MAPA DA IDADE MÉDIA AO MORRER POR DISTRITO ######

# Calcula a média geral
media_geral = mapa_dados["Idade_media"].mean()

# Cria coluna categórica
mapa_dados["Situacao"] = mapa_dados["Idade_media"].apply(
    lambda x: "Acima da média" if x >= media_geral else "Abaixo da média"
)

# Definir duas cores do viridis (um tom claro e um mais escuro)
cores = {
    "Abaixo da média": plt.cm.viridis(0.2),  # tom mais claro
    "Acima da média": plt.cm.viridis(0.8)    # tom mais escuro
}

# Função modificada para categorias
def plot_mapa_categorico(mapa, coluna, titulo):
    fig, ax = plt.subplots(1, 1, figsize=(15, 12))

    mapa.plot(
        column=coluna,
        color=mapa[coluna].map(cores),  # <- corrigi aqui (era "colors")
        linewidth=0.8,
        ax=ax,
        edgecolor="0.8",
    )

    # Cria legenda manual
    for categoria, cor in cores.items():
        ax.scatter([], [], color=cor, label=categoria)

    ax.legend(title="Situação", fontsize=12)
    ax.set_axis_off()
    plt.title(titulo, fontsize=16)  # agora funciona porque "titulo" vem do parâmetro
    plt.show()

###### MAPA COLORIDO ######
plot_mapa_categorico(
    mapa=mapa_dados,
    coluna="Situacao",
    titulo="Distritos Acima ou Abaixo da Média da Idade Média ao Morrer"
)

############

###### MAPA DA OFERTA DE EMPREGO FORMAL POR DISTRITO ######
plot_mapa(
    mapa=mapa_dados,
    coluna="Oferta_emprego",
    titulo="Distribuição da Oferta de Emprego Formal por Distrito em São Paulo"
)

###### MAPA DA INCIDÊNCIA DE DENGUE POR DISTRITO ######


plot_mapa(
    mapa=mapa_dados,
    coluna="Casos_dengue",
    titulo="Distribuição da Incidência de Casos de Dengue por Distrito em São Paulo"
)

##### GRÁFICO FAIXAS DE SALÁRIOS MÍNIMOS #####

print(emprego_2020_2023.columns.tolist())

renomear = {
    'Até 0,5 SM': 'Até 0,50',
    '0,51 a 1 SM': '0,51 a 1,00',
    '1,01 a 1,5 SM': '1,01 a 1,50',
    '1,51 a 2 SM': '1,51 a 2,00',
    '2,01 a 3 SM': '2,01 a 3,00',
    '3,01 a 4 SM': '3,01 a 4,00',
    '4,01 a 5 SM': '4,01 a 5,00',
    '5,01 a 7 SM': '5,0 a 7,00',
    '7,01 a 10 SM': '7,01 a 10,00',
    '10,01 a 15 SM': '10,01 a 15,00',
    '15,01 a 20 SM': '15,01 a 20,00',
    'Mais de 20 SM': 'Mais de 20,00'
}

emprego_2020_2023 = emprego_2020_2023.rename(columns=renomear)


# faixas salariais
faixas = [
    'Até 0,50', '0,51 a 1,00', '1,01 a 1,50', '1,51 a 2,00', 
    '2,01 a 3,00', '3,01 a 4,00', '4,01 a 5,00', '5,0 a 7,00',
    '7,01 a 10,00', '10,01 a 15,00', '15,01 a 20,00', 
    'Mais de 20,00', 'Ignorado'
]

# Criando gráfico
emprego_2020_2023.set_index("Distrito")[faixas].plot(
    kind="bar",
    stacked=True,
    figsize=(16, 8),
    colormap="viridis"
)

plt.ylabel("População empregada")
plt.title("Distribuição da população por faixa de salários mínimos (2020-2023)", fontsize=14)
plt.legend(title="Faixas de salários mínimos", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

##### Mapa coroplético por faixa (Choropleth Map) #####

print(mapa_dados.columns.tolist())

mapa_emprego = gdf_distritos.merge(
    emprego_2020_2023,
    left_on="NOME_DIST_CORRIGIDO",
    right_on="Distrito",
    how="left"
)

print(mapa_emprego.head())
print(mapa_emprego.geometry.head())

print(mapa_dados["NOME_DIST_CORRIGIDO"].unique()[:20])
print(emprego_2020_2023["Distrito"].unique()[:20])

##### Adaptações necessárias ######

def limpar_nome_distrito(df_col):
    return (
        df_col.astype(str)
        .str.normalize('NFKD')   # Remove acentos
        .str.encode('ascii', errors='ignore')
        .str.decode('utf-8')
        .str.upper()             # Letras maiúsculas
        .str.strip()             # Remove espaços no início/fim
    )

# Aplicando limpeza
mapa_dados['Distrito_merge'] = mapa_dados['NOME_DIST_CORRIGIDO']
emprego_2020_2023['Distrito_merge'] = limpar_nome_distrito(emprego_2020_2023['Distrito'])

# Merge
mapa_emprego = mapa_dados.merge(
    emprego_2020_2023,
    on='Distrito_merge',
    how='left'
)

# Conferir se os dados de emprego agora estão preenchidos
print(mapa_emprego[["Distrito_merge"] + emprego_2020_2023.columns[1:].tolist()].head())


# Plot do mapa para a faixa "Até 5 salários mínimos"
fig, ax = plt.subplots(1, 1, figsize=(15, 12))

mapa_emprego.plot(
    column='4,01 a 5,00',   
    cmap='viridis',
    linewidth=0.8,
    ax=ax,
    edgecolor='0.8',
    legend=True,
    legend_kwds={'label': "População", 'orientation': "horizontal"},
    missing_kwds={
        "color": "lightgrey",
        "edgecolor": "red",
        "hatch": "///",
        "label": "Sem dados",
    }
)

ax.set_axis_off()
plt.title('População que ganha até 5 salários mínimos por distrito', fontsize=16)
plt.show()


# Plot do mapa para a faixa "Até 4 salários mínimos"
fig, ax = plt.subplots(1, 1, figsize=(15, 12))

mapa_emprego.plot(
    column='3,01 a 4,00',   
    cmap='viridis',
    linewidth=0.8,
    ax=ax,
    edgecolor='0.8',
    legend=True,
    legend_kwds={'label': "População", 'orientation': "horizontal"},
    missing_kwds={
        "color": "lightgrey",
        "edgecolor": "red",
        "hatch": "///",
        "label": "Sem dados",
    }
)

ax.set_axis_off()
plt.title('População que ganha até 4 salários mínimos por distrito', fontsize=16)
plt.show()

##########

# Plot do mapa para a faixa "Até X salários mínimos - mudar column"
fig, ax = plt.subplots(1, 1, figsize=(15, 12))

mapa_emprego.plot(
    column='Até 0,50',   
    cmap='viridis',
    linewidth=0.8,
    ax=ax,
    edgecolor='0.8',
    legend=True,
    legend_kwds={'label': "População", 'orientation': "horizontal"},
    missing_kwds={
        "color": "lightgrey",
        "edgecolor": "red",
        "hatch": "///",
        "label": "Sem dados",
    }
)

ax.set_axis_off()
plt.title('População que ganha até 0,5 salário mínimo por distrito', fontsize=16)
plt.show()

print(mapa_emprego.columns.tolist())

###################################### MODELAGEM MULTINÍVEL ######################################
##################################################################################################

# --- Preparação para a Modelagem Multinível
import pandas as pd
import numpy as np
from statistics import mode   # Mais simples que scipy.stats.mode
import statsmodels.formula.api as smf

# --- Função utilitária ---
def limpar_nome_distrito(serie: pd.Series) -> pd.Series:
    """Padroniza nomes de distritos: remove acentos, deixa maiúsculo e sem espaços extras."""
    return (
        serie.astype(str)
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
        .str.upper()
        .str.strip()
    )

# --- 1. Padronizar distritos ---
for df in [dados_quanti_pca, dados_quanti_]:
    df["Distrito_padronizado"] = limpar_nome_distrito(df["Distrito"])

# --- 2. Cluster dominante e médias por distrito ---
df_cluster_dominante = (
    dados_quanti_pca
    .groupby("Distrito_padronizado", as_index=False)
    .agg(
        Cluster_dominante=("Cluster_PCA", lambda x: mode(x)),
        Pop_preta_parda_media=("Pop_preta_parda", "mean"),
        Oferta_emprego_media=("Oferta_emprego", "mean"),
        Domicilios_fav_media=("Domicilios_fav", "mean"),
        Casos_dengue_media=("Casos_dengue", "mean"),
    )
)

# --- 3. Merge com base principal ---
dados_final = (
    dados_quanti_
    .merge(df_cluster_dominante, on="Distrito_padronizado", how="left")
    .rename(columns={"Distrito_padronizado": "Distrito_ID"})
    .drop(columns=[c for c in ["Cluster", "Cluster_5"] if c in dados_quanti_.columns])  # só remove se existir
)

# --- 4. Verificação ---
print(dados_final.head())
print(dados_final.info())


# Média e desvio padrão das variáveis por cluster
tabela_clusters = (
    dados_final.groupby("Cluster_dominante")
    .agg({
        "Idade_media": ["mean", "std"],
        "Pop_preta_parda": "mean",
        "Oferta_emprego": "mean",
        "Domicilios_fav": "mean",
        "Casos_dengue": "mean"
    })
    .round(2)
)

print(tabela_clusters)

tabela_ano_cluster = (
    dados_final.groupby(["Ano", "Cluster_dominante"])
    .agg({"Idade_media": "mean"})
    .reset_index()
)

print(tabela_ano_cluster.head())

### Distribuição da Idade Média por Cluster

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
sns.boxplot(x="Cluster_dominante", y="Idade_media", data=dados_final)
plt.title("Distribuição da Idade Média por Cluster Dominante")
plt.show()

### Evolução Temporal da Idade Média por Cluster

plt.figure(figsize=(10,6))
sns.lineplot(
    data=dados_final, 
    x="Ano", y="Idade_media", 
    hue="Cluster_dominante", 
    estimator="mean", ci=None
)
plt.title("Evolução da Idade Média por Cluster Dominante")
plt.show()

### Relação entre População Preta/Parda e Idade Média

plt.figure(figsize=(7,5))
sns.scatterplot(
    data=dados_final, 
    x="Pop_preta_parda", 
    y="Idade_media", 
    hue="Cluster_dominante", 
    alpha=0.7
)
plt.title("Idade Média vs. População Preta/Parda (%)")
plt.show()


### Favelas × Ofertas de Emprego Formal

plt.figure(figsize=(7,5))
sns.scatterplot(
    data=dados_final,
    x="Domicilios_fav",
    y="Oferta_emprego",
    hue="Cluster_dominante",
    alpha=0.7
)
sns.regplot(
    data=dados_final,
    x="Domicilios_fav",
    y="Oferta_emprego",
    scatter=False,
    color="red"
)
plt.title("Domicílios em Favelas vs. Oferta de Emprego Formal")
plt.xlabel("Proporção de Domicílios em Favelas")
plt.ylabel("Oferta de Emprego Formal")
plt.show()

### Favelas × Casos de Dengue

plt.figure(figsize=(7,5))
sns.scatterplot(
    data=dados_final,
    x="Domicilios_fav",
    y="Casos_dengue",
    hue="Cluster_dominante",
    alpha=0.7
)
sns.regplot(
    data=dados_final,
    x="Domicilios_fav",
    y="Casos_dengue",
    scatter=False,
    color="blue"
)
plt.title("Domicílios em Favelas vs. Casos de Dengue")
plt.xlabel("Proporção de Domicílios em Favelas")
plt.ylabel("Casos de Dengue")
plt.show()

### Oferta de emprego × Idade Média ao Morrer

plt.figure(figsize=(7,5))
sns.scatterplot(
    data=dados_final,
    x="Domicilios_fav",       # favelas
    y="Oferta_emprego",       # emprego formal
    alpha=0.7
)
sns.regplot(
    data=dados_final,
    x="Domicilios_fav",
    y="Oferta_emprego",
    scatter=False,
    color="red"
)
plt.title("Favelas (domicílios em favelas) vs. Ofertas de Emprego Formal")
plt.show()

####

corr = dados_final[["Domicilios_fav", "Oferta_emprego"]].corr()

plt.figure(figsize=(5,4))
sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f")
plt.title("Correlação entre Favelas e Empregos Formais")
plt.show()

# --- Modelagem Hierárquica Linear (HLM)

dados_final = dados_final.dropna(subset=["Idade_media", "Ano", "Pop_preta_parda_media", "Domicilios_fav_media", "Cluster_dominante"])

import statsmodels.formula.api as smf

# Modelo 1: Apenas com Intercepto Aleatório para o Distrito
print("\n--- Rodando o Modelo 1 (Intercepto Aleatório) ---")
modelo_1 = smf.mixedlm(
    "Idade_media ~ 1",
    data=dados_final,
    groups=dados_final["Distrito_ID"]
)
resultado_1 = modelo_1.fit(reml=False)
print(resultado_1.summary())

# ---

# Modelo 2: Adicionando Efeitos Fixos e o Cluster Dominante
print("\n--- Rodando o Modelo 2 (Efeitos Fixos e Cluster) ---")
modelo_2 = smf.mixedlm(
    "Idade_media ~ Ano + Pop_preta_parda_media + Domicilios_fav_media + C(Cluster_dominante)",
    data=dados_final,
    groups=dados_final["Distrito_ID"]
)
resultado_2 = modelo_2.fit(reml=False)
print(resultado_2.summary())

# ---

# Modelo 3 (Mais Avançado): Adicionando a inclinação aleatória para a variável tempo (Ano)

modelo_final = smf.mixedlm(
    "Idade_media ~ Ano_s + Pop_preta_parda_media_s + Domicilios_fav_media_s + C(Cluster_dominante)",
    data=dados_final,
    groups=dados_final["Distrito_ID"]
)
resultado_final = modelo_final.fit(reml=False)
print(resultado_final.summary())


### Efeito do cluster dominante

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Previsão média por cluster (usando coeficientes do modelo)
clusters = [0, 1, 3]  # Cluster de referência = 0
coef = resultado_final.params

idade_media_prevista = [
    coef['Intercept'],                    # Cluster 0
    coef['Intercept'] + coef.get('C(Cluster_dominante)[T.1]', 0), 
    coef['Intercept'] + coef.get('C(Cluster_dominante)[T.3]', 0)
]

plt.figure(figsize=(6,4))
sns.barplot(x=clusters, y=idade_media_prevista, palette="Set2")
plt.xlabel("Cluster Dominante")
plt.ylabel("Idade Média ao Morrer Prevista")
plt.title("Efeito do Cluster Dominante na Idade Média")
plt.show()


### Relação com proporção de população preta/parda

pop_vals = np.linspace(dados_final['Pop_preta_parda_media_s'].min(),
                       dados_final['Pop_preta_parda_media_s'].max(), 100)

idade_prev = coef['Intercept'] + coef['Pop_preta_parda_media_s']*pop_vals

plt.figure(figsize=(6,4))
plt.plot(pop_vals, idade_prev, color='red')
plt.xlabel("População Preta/Parda (padronizada)")
plt.ylabel("Idade Média ao Morrer Prevista")
plt.title("Efeito da População Preta/Parda na Idade Média")
plt.show()

### Linha do tempo por distrito

# Seleciona 5 distritos aleatórios
distritos_ex = np.random.choice(dados_final['Distrito_ID'].unique(), 5, replace=False)

plt.figure(figsize=(8,5))
for d in distritos_ex:
    df_dist = dados_final[dados_final['Distrito_ID']==d]
    plt.plot(df_dist['Ano'], df_dist['Idade_media'], marker='o', label=d)

plt.xlabel("Ano")
plt.ylabel("Idade Média ao Morrer")
plt.title("Idade Média ao Morrer por Distrito")
plt.legend()
plt.show()

### Gráfico de dispersão 3D

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

x = dados_final['Pop_preta_parda_media']
y = dados_final['Oferta_emprego_media']
z = dados_final['Idade_media']

# Cor pelo cluster dominante
clusters = dados_final['Cluster_dominante']
colors = sns.color_palette("Set2", n_colors=len(clusters.unique()))

for c in clusters.unique():
    mask = clusters == c
    ax.scatter(x[mask], y[mask], z[mask], label=f'Cluster {c}', s=50, alpha=0.7)

ax.set_xlabel('População Preta/Parda (%)')
ax.set_ylabel('Oferta de Emprego')
ax.set_zlabel('Idade Média ao Morrer')
ax.set_title('Relação entre População Preta/Parda, Emprego e Idade Média')
ax.legend()
plt.show()

### Heatmap / tamanho do ponto (2D)

plt.figure(figsize=(7,5))
sns.scatterplot(
    data=dados_final,
    x='Pop_preta_parda_media',
    y='Oferta_emprego_media',
    size='Idade_media',
    hue='Cluster_dominante',
    palette='Set2',
    alpha=0.7,
    sizes=(20,200)
)
plt.xlabel('População Preta/Parda (%)')
plt.ylabel('Oferta de Emprego')
plt.title('População Preta/Parda vs. Oferta de Emprego com Idade Média ao Morrer')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.show()

### Scatter 3D: População + emprego + domicílio

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

x = dados_final['Pop_preta_parda_media']
y = dados_final['Oferta_emprego_media']
z = dados_final['Domicilios_fav_media']
clusters = dados_final['Cluster_dominante']

# Paleta de cores para os clusters
colors = sns.color_palette("Set2", n_colors=len(clusters.unique()))

for c in clusters.unique():
    mask = clusters == c
    ax.scatter(x[mask], y[mask], z[mask], label=f'Cluster {c}', s=50, alpha=0.7)

ax.set_xlabel('População Preta/Parda (%)')
ax.set_ylabel('Oferta de Emprego')
ax.set_zlabel('Domicílios em Favelas')
ax.set_title('População Preta/Parda x Oferta de Emprego x Domicílios em Favelas')
ax.legend()
plt.show()

### Scatter 2D com tamanho

plt.figure(figsize=(7,5))
sns.scatterplot(
    data=dados_final,
    x='Pop_preta_parda_media',
    y='Oferta_emprego_media',
    size='Domicilios_fav_media',
    hue='Cluster_dominante',
    palette='Set2',
    alpha=0.7,
    sizes=(20,200)
)
plt.xlabel('População Preta/Parda (%)')
plt.ylabel('Oferta de Emprego')
plt.title('População Preta/Parda vs. Oferta de Emprego (tamanho = Domicílios em Favelas)')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.show()

### Scatter 3D: Idade + emprego + domicílio

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

x = dados_final['Idade_media']
y = dados_final['Oferta_emprego_media']
z = dados_final['Domicilios_fav_media']
clusters = dados_final['Cluster_dominante']

colors = sns.color_palette("Set2", n_colors=len(clusters.unique()))

for c in clusters.unique():
    mask = clusters == c
    ax.scatter(x[mask], y[mask], z[mask], label=f'Cluster {c}', s=50, alpha=0.7)

ax.set_xlabel('Idade Média ao Morrer')
ax.set_ylabel('Oferta de Emprego')
ax.set_zlabel('Domicílios em Favelas')
ax.set_title('Idade Média ao Morrer x Oferta de Emprego x Domicílios em Favelas')
ax.legend()
plt.show()

### Scatter 2D com tamanho

plt.figure(figsize=(7,5))
sns.scatterplot(
    data=dados_final,
    x='Idade_media',
    y='Oferta_emprego_media',
    size='Domicilios_fav_media',
    hue='Cluster_dominante',
    palette='Set2',
    alpha=0.7,
    sizes=(20,200)
)
plt.xlabel('Idade Média ao Morrer')
plt.ylabel('Oferta de Emprego')
plt.title('Idade Média ao Morrer vs. Oferta de Emprego (tamanho = Domicílios em Favelas)')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.show()


### Scatter 3D Idade média ao morrer + oferta de empregos + dengue

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

x = dados_final['Idade_media']
y = dados_final['Oferta_emprego_media']
z = dados_final['Casos_dengue_media']
clusters = dados_final['Cluster_dominante']

colors = sns.color_palette("Set2", n_colors=len(clusters.unique()))

for c in clusters.unique():
    mask = clusters == c
    ax.scatter(x[mask], y[mask], z[mask], label=f'Cluster {c}', s=50, alpha=0.7)

ax.set_xlabel('Idade Média ao Morrer')
ax.set_ylabel('Oferta de Emprego')
ax.set_zlabel('Casos de Dengue')
ax.set_title('Idade Média ao Morrer x Oferta de Emprego x Casos de Dengue')
ax.legend()
plt.show()

### Scatter 2D

#V1
plt.figure(figsize=(7,5))
sns.scatterplot(
    data=dados_final,
    x='Idade_media',
    y='Oferta_emprego_media',
    size='Casos_dengue_media',
    hue='Cluster_dominante',
    palette='Set2',
    alpha=0.7,
    sizes=(20,200)
)
plt.xlabel('Idade Média ao Morrer')
plt.ylabel('Oferta de Emprego')
plt.title('Idade Média ao Morrer vs. Oferta de Emprego (tamanho = Casos de Dengue)')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.show()

### Gráfico de cores contínuas (colormap)

plt.figure(figsize=(8,6))
scatter = plt.scatter(
    x=dados_final['Oferta_emprego_media'],
    y=dados_final['Idade_media'],
    s=dados_final['Idade_media']*5,
    c=dados_final['Casos_dengue_media'],
    cmap='Reds',
    alpha=0.7,
    edgecolor='k'
)
plt.xlabel('Oferta de Emprego')
plt.ylabel('Idade Média ao Morrer')
plt.title('Oferta de Emprego x Idade Média (cor = Casos de Dengue)')
cbar = plt.colorbar(scatter)
cbar.set_label('Casos de Dengue')
plt.show()

### Pairplot para explorar relações múltiplas

sns.pairplot(
    dados_final[['Idade_media','Oferta_emprego_media','Casos_dengue_media']],
    kind='scatter',
    plot_kws={'alpha':0.6, 's':50}
)
plt.suptitle('Relação entre Idade Média, Oferta de Emprego e Dengue', y=1.02)
plt.show()

### Heatmap de correlação
corr = dados_final[['Idade_media','Oferta_emprego_media','Casos_dengue_media']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlação entre Idade Média, Oferta de Emprego e Dengue')
plt.show()

### Linha do tempo oferta de emprego

plt.figure(figsize=(8,5))
for d in distritos_ex:
    df_dist = dados_final[dados_final['Distrito_ID'] == d]
    plt.plot(df_dist['Ano'], df_dist['Oferta_emprego'], marker='o', label=d)

plt.xlabel("Ano")
plt.ylabel("Oferta de Emprego")
plt.title("Oferta de Emprego Formal por Distrito ao Longo do Tempo")
plt.legend()
plt.show()

### Linha do tempo de Domicílios em Favelas por distrito (valores anuais)

# Seleciona 5 distritos aleatórios
distritos_ex = np.random.choice(dados_final['Distrito_ID'].unique(), 5, replace=False)

plt.figure(figsize=(8,5))
for d in distritos_ex:
    df_dist = dados_final[dados_final['Distrito_ID'] == d]
    plt.plot(df_dist['Ano'], df_dist['Domicilios_fav'], marker='o', label=d)

plt.xlabel("Ano")
plt.ylabel("Domicílios em Favelas")
plt.title("Domicílios em Favelas por Distrito ao Longo do Tempo")
plt.legend()
plt.show()

### Linha do tempo de Casos de Dengue por distrito (valores anuais)

# Seleciona 5 distritos aleatórios
distritos_ex = np.random.choice(dados_final['Distrito_ID'].unique(), 5, replace=False)

plt.figure(figsize=(8,5))
for d in distritos_ex:
    df_dist = dados_final[dados_final['Distrito_ID'] == d]
    plt.plot(df_dist['Ano'], df_dist['Casos_dengue'], marker='o', label=d)

plt.xlabel("Ano")
plt.ylabel("Casos de Dengue")
plt.title("Incidência de Dengue por Distrito ao Longo do Tempo")
plt.legend()
plt.show()





