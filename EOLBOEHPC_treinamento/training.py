import pandas as pd

####################################################################################################################################################
dados = pd.read_csv('EOLBOEHPC_treinamento\\ObesityDataSet_raw_and_data_sinthetic.csv', sep = ',')
print(dados.head(5))
####################################################################################################################################################
dados_numericos = dados.drop(columns = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad'])
dados_categoricos = dados[['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad']]
####################################################################################################################################################



#normalizar os dados categoricos
dados_categoricos_normalizados = pd.get_dummies(data=dados_categoricos, dtype=int)


print(dados_categoricos_normalizados)

####################################################################################################################################################

#Treinar o modelo normalizador para os dados numericos
from pickle import dump
from sklearn import preprocessing
normalizador = preprocessing.MinMaxScaler() #?
modelo_normalizador = normalizador.fit(dados_numericos)
dump(modelo_normalizador, open('modelo_normalizador.pkl', 'wb'))
####################################################################################################################################################








#Normalizar base de dados de entrada
dados_numericos_normalizados = modelo_normalizador.fit_transform(dados_numericos)
#print(dados_numericos)
#print(dados_numericos_normalizados)

#criar um dataframe com os dados normalizados tanto categoricos e numericos
#converter os dados numericos normalizados em dataframe

dados_numericos_normalizados = pd.DataFrame(data = dados_numericos_normalizados, columns = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH20', 'FAF', 'TUE'])

#juntar com os dados categoricos normalizados
dados_normalizados_final = dados_numericos_normalizados.join(dados_categoricos_normalizados, how = 'left')
#print(dados_normalizados_final.head(1))

#transforma dados em algo legivel

dados_normalizados_final_legiveis = modelo_normalizador.inverse_transform(dados_numericos_normalizados)
#print(dados_normalizados_final_legiveis)

dados_normalizados_final_legiveis = pd.DataFrame(data = dados_normalizados_final_legiveis, columns = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH20', 'FAF', 'TUE']).join(dados_categoricos_normalizados)
pd.set_option('display.max_columns', None)
print(dados_normalizados_final_legiveis)
####################################################################################################################################################




from sklearn.cluster import KMeans #clusterizador
import matplotlib.pyplot as plt #para graficos
import math #matematica
from scipy.spatial.distance import cdist #para calcular as distancias e distorções
import numpy as np # para procedimentos numericos
distortions = []
K = range (1, 101)
#Treinar iterativamente conforme n_clusters = K[i]
for i in K:
  EOLBOEHPC_kmeans_model =  KMeans(n_clusters = i).fit(dados_normalizados_final_legiveis)
  distortions.append(sum(np.min(cdist(dados_normalizados_final_legiveis, EOLBOEHPC_kmeans_model.cluster_centers_, 'euclidean'), axis = 1)/dados_normalizados_final_legiveis.shape[0]))

print(distortions)




####################################################################################################################################################

fig, ax = plt.subplots()
ax.plot(K, distortions)
ax.set(xlabel = 'n Clusters', ylabel = 'Distorção', title = 'Elbow pela distorção')
ax.grid()
fig.savefig('EOLBOEHPC_treinamento\\elbow_distorcao.png')
plt.show()

####################################################################################################################################################


#Calcular o numero otimo de clusters
from math import sqrt
x0 = K[0]
y0 = distortions[0]
xn = K[len(K)-1]
yn = distortions[len(distortions)- 1]
#iterar nos pontos gerados durante os treinamentos preliminares
distancias = []

for i in range(len(distortions)):
  x = K[i]
  y = distortions[i]
  numerador = abs((yn - y0) * x- (xn-y0)*y + xn*y0 - yn*x0)
  denominador = sqrt((yn-y0)**2 + (xn-x0)**2)
  distancias.append(numerador/denominador)

#maior distancia
n_clusters_otimos = K[distancias.index(np.max(distancias))]
print(n_clusters_otimos)



#treinar modelo definitivo
EOLBOEHPC_kmeans_model =  KMeans(n_clusters = n_clusters_otimos, random_state=42).fit(dados_normalizados_final)

#salvar o modelo definitivo
from pickle import dump
dump(EOLBOEHPC_kmeans_model, open('EOLBOEHPC_treinamento\\EOLBOEHPC_clusters_2024.pkl', 'wb')) # salva modelo
print(EOLBOEHPC_kmeans_model.cluster_centers_)
#salvar colunas

# Convertendo o Index de nomes de colunas para uma lista
nomes_colunas = dados_normalizados_final.columns.tolist()

# Salvar os nomes das colunas em um arquivo CSV
with open('EOLBOEHPC_treinamento\\EOLBOEHPC.csv', 'w') as file:
    file.write(','.join(nomes_colunas))
    
    
