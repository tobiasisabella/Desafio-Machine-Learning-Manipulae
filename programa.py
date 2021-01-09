#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 20:58:22 2020

@author: isabella tobias
"""


#Importando as bibliotecas necessárias
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#Importando o arquivo com dados
df = pd.read_csv('dados.csv')


#Começando a fase de pré-processamento e exploração dos dados
#Saber o nome das colunas
print("Tamanho do arquivo:")
print(df.shape)
print("--------------------")




print("Título das colunas")
print(df.columns.values)
print("--------------------")

print("Retirando as informações do arquivo:")
print(df.info())
print("--------------------")


#Verificando se exitem de dados vazio nas colunas
print(df.isnull().sum())
print("--------------------")


#Função para utilizar apenas o número dado pela quantidade de cápsulas
def separate(line):
    a = line.split(" ")
    b = a[0]
    return b

df['descricao'] = df['descricao'].map(separate)


#Gráficos
drop_elements = ['criado']
train = df.drop(drop_elements, axis = 1)
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Correlação entre as Variáveis', y=1.05, size=15)
sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
plt.show()


#O quadro de correlação mostrou que apenas a variável "calculado" tem uma 
#correlação significativa com a varíavel correto.



# Separando o array em componentes de input e output
atributos = ['calculado']
# Variável a ser prevista
atrib_prev = ['correto']
# Criando objetos
X = df[atributos].values
Y = df[atrib_prev].values

#Normalizando os dados entre 0 a 1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))
rescaledX = scaler.fit_transform(X)
rescaledY = scaler.fit_transform(Y)


#Dividindo os dados transformados em treino (~70%) e teste (~30%)
from sklearn.model_selection import train_test_split
split_test_size = 0.280859
# Criando dados de treino e de teste
X_treino, X_teste, Y_treino, Y_teste = train_test_split(rescaledX, rescaledY, test_size = split_test_size, random_state = 42)
# Imprimindo os resultados
print("{0:0.2f}% nos dados de treino".format((len(X_treino)/len(df.index)) * 100))
print("{0:0.2f} nos dados de treino".format((len(X_treino))))
print("{0:0.2f}% nos dados de teste".format((len(X_teste)/len(df.index)) * 100))
print("{0:0.2f} nos dados de teste".format((len(X_teste))))



#Testando os algoritmos de regressão para encontrar aquele com o menor erro
print('------------------------Regressão Linear------------------------------')
#Regressão Linear
from sklearn.linear_model import LinearRegression
modelo = LinearRegression()
modelo.fit(X_treino, Y_treino.ravel())
predict_train = modelo.predict(X_treino)
predict_test = modelo.predict(X_teste)
print("Exatidão Treino (Accuracy): {0:.4f}%".format((modelo.score(Y_treino, predict_train))*100))
print("Exatidão Teste (Accuracy): {0:.4f}%".format((modelo.score(Y_teste, predict_test))*100))
from sklearn import metrics
import numpy as np
print("Mean absolute error (MAE):", metrics.mean_absolute_error(Y_teste, predict_test))
print("Mean square error (MSE):", metrics.mean_squared_error(Y_teste, predict_test))
print("Root mean square error (RMSE):", np.sqrt(metrics.mean_squared_error(Y_teste, predict_test)))
print("R-squared value of predictions:",round(metrics.r2_score(Y_teste, predict_test),3))



# Import dos módulos
print('---------------------------Ridge Regression---------------------------')
#Ridge Regression
from sklearn.linear_model import Ridge
# Criando o modelo
modelo1 = Ridge()
modelo1.fit(X_treino, Y_treino.ravel())
predict_train = modelo1.predict(X_treino)
predict_test = modelo1.predict(X_teste)
print("Exatidão Treino (Accuracy): {0:.4f}%".format((modelo1.score(Y_treino, predict_train))*100))
print("Exatidão Teste (Accuracy): {0:.4f}%".format((modelo1.score(Y_teste, predict_test))*100))
print("Mean absolute error (MAE):", metrics.mean_absolute_error(Y_teste, predict_test))
print("Mean square error (MSE):", metrics.mean_squared_error(Y_teste, predict_test))
print("Root mean square error (RMSE):", np.sqrt(metrics.mean_squared_error(Y_teste, predict_test)))
print("R-squared value of predictions:",round(metrics.r2_score(Y_teste, predict_test),3))

print('-------------------------Lasso Regression-----------------------------')
#Lasso Regression
from sklearn.linear_model import Lasso
modelo2= Lasso()
modelo2.fit(X_treino, Y_treino.ravel())
predict_train = modelo2.predict(X_treino)
predict_test = modelo2.predict(X_teste)
print("Exatidão Treino (Accuracy): {0:.4f}%".format((modelo2.score(Y_treino, predict_train))*100))
print("Exatidão Teste (Accuracy): {0:.4f}%".format((modelo2.score(Y_teste, predict_test))*100))
print("Mean absolute error (MAE):", metrics.mean_absolute_error(Y_teste, predict_test))
print("Mean square error (MSE):", metrics.mean_squared_error(Y_teste, predict_test))
print("Root mean square error (RMSE):", np.sqrt(metrics.mean_squared_error(Y_teste, predict_test)))
print("R-squared value of predictions:",round(metrics.r2_score(Y_teste, predict_test),3))

print('-------------------------------KNN------------------------------------')
#Regressão Linear
from sklearn.neighbors import KNeighborsRegressor
modelo3 = KNeighborsRegressor()
modelo3.fit(X_treino, Y_treino.ravel())
predict_train = modelo3.predict(X_treino)
predict_test = modelo3.predict(X_teste)
print("Exatidão Treino (Accuracy): {0:.4f}%".format((modelo3.score(Y_treino, predict_train))*100))
print("Exatidão Teste (Accuracy): {0:.4f}%".format((modelo3.score(Y_teste, predict_test))*100))
print("Mean absolute error (MAE):", metrics.mean_absolute_error(Y_teste, predict_test))
print("Mean square error (MSE):", metrics.mean_squared_error(Y_teste, predict_test))
print("Root mean square error (RMSE):", np.sqrt(metrics.mean_squared_error(Y_teste, predict_test)))
print("R-squared value of predictions:",round(metrics.r2_score(Y_teste, predict_test),3))


print('-------------------------ElasticNet Regression------------------------')
#Regressão Linear
from sklearn.linear_model import ElasticNet
modelo4 = ElasticNet()
modelo4.fit(X_treino, Y_treino.ravel())
predict_train = modelo4.predict(X_treino)
predict_test = modelo4.predict(X_teste)
print("Exatidão Treino (Accuracy): {0:.4f}%".format((modelo4.score(Y_treino, predict_train))*100))
print("Exatidão Teste (Accuracy): {0:.4f}%".format((modelo4.score(Y_teste, predict_test))*100))
print("Mean absolute error (MAE):", metrics.mean_absolute_error(Y_teste, predict_test))
print("Mean square error (MSE):", metrics.mean_squared_error(Y_teste, predict_test))
print("Root mean square error (RMSE):", np.sqrt(metrics.mean_squared_error(Y_teste, predict_test)))
print("R-squared value of predictions:",round(metrics.r2_score(Y_teste, predict_test),3))

print('-------------------------------CART-----------------------------------')
#Regressão Linear
from sklearn.tree import DecisionTreeRegressor
modelo5 = DecisionTreeRegressor()
modelo5.fit(X_treino, Y_treino.ravel())
predict_train = modelo5.predict(X_treino)
predict_test = modelo5.predict(X_teste)
print("Exatidão Treino (Accuracy): {0:.4f}%".format((modelo5.score(Y_treino, predict_train))*100))
print("Exatidão Teste (Accuracy): {0:.4f}%".format((modelo5.score(Y_teste, predict_test))*100))
print("Mean absolute error (MAE):", metrics.mean_absolute_error(Y_teste, predict_test))
print("Mean square error (MSE):", metrics.mean_squared_error(Y_teste, predict_test))
print("Root mean square error (RMSE):", np.sqrt(metrics.mean_squared_error(Y_teste, predict_test)))
print("R-squared value of predictions:",round(metrics.r2_score(Y_teste, predict_test),3))


print('-------------------------------SVM------------------------------------')
#Regressão Linear
from sklearn.svm import SVR
modelo6 = SVR()
modelo6.fit(X_treino, Y_treino.ravel())
predict_train = modelo6.predict(X_treino)
predict_test = modelo6.predict(X_teste)
print("Exatidão Treino (Accuracy): {0:.4f}%".format((modelo6.score(Y_treino, predict_train))*100))
print("Exatidão Teste (Accuracy): {0:.4f}%".format((modelo6.score(Y_teste, predict_test))*100))
print("Mean absolute error (MAE):", metrics.mean_absolute_error(Y_teste, predict_test))
print("Mean square error (MSE):", metrics.mean_squared_error(Y_teste, predict_test))
print("Root mean square error (RMSE):", np.sqrt(metrics.mean_squared_error(Y_teste, predict_test)))
print("R-squared value of predictions:",round(metrics.r2_score(Y_teste, predict_test),3))


print('--------------------------Random Forest-------------------------------')
#Regressão Linear
from sklearn.ensemble import RandomForestRegressor
modelo = RandomForestRegressor()
modelo.fit(X_treino, Y_treino.ravel())
predict_train = modelo.predict(X_treino)
predict_test = modelo.predict(X_teste)
print("Exatidão Treino (Accuracy): {0:.4f}%".format((modelo.score(Y_treino, predict_train))*100))
print("Exatidão Teste (Accuracy): {0:.4f}%".format((modelo.score(Y_teste, predict_test))*100))
from sklearn import metrics
import numpy as np
print("Mean absolute error (MAE):", metrics.mean_absolute_error(Y_teste, predict_test))
print("Mean square error (MSE):", metrics.mean_squared_error(Y_teste, predict_test))
print("Root mean square error (RMSE):", np.sqrt(metrics.mean_squared_error(Y_teste, predict_test)))
print("R-squared value of predictions:",round(metrics.r2_score(Y_teste, predict_test),3))

#Todos os algoritmos rodaram. Alguns foram melhores do que outros e alguns tiveram erros.
#Esses erros são explicados pela estrutura dos dados do arquivo.

#O algoritmo que apresentou os melhores resultados foi o de "Regressão Linear"
