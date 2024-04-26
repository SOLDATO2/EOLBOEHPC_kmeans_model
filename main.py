import pandas as pd
from pickle import load
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Carregar o modelo de normalização e colunas categóricas
EOLBOEHPC_kmeans_model = load(open('EOLBOEHPC_treinamento\\EOLBOEHPC_clusters_2024.pkl', 'rb'))
modelo_normalizador = load(open('modelo_normalizador.pkl', 'rb'))
with open('EOLBOEHPC_treinamento\\EOLBOEHPC.csv', 'r') as file:
    # Ler a primeira linha do arquivo, que contém os nomes das colunas separados por vírgulas
    columns = file.readline().strip().split(',')

# Criar DataFrame vazio com as colunas obtidas
data_frame = pd.DataFrame(columns=columns)

# Nova instância
nova_instancia = ["Male", 22, 1.75, 95, "yes", "no", 2, 3, "Sometimes", "no", 3, "no", 3, 2, "no", "Walking", "Obesity_Type_I"]
nova_instancia_dict = {
    'Gender': nova_instancia[0],
    'Age': nova_instancia[1],
    'Height': nova_instancia[2],
    'Weight': nova_instancia[3],
    'family_history_with_overweight': nova_instancia[4],
    'FAVC': nova_instancia[5],
    'FCVC': nova_instancia[6],
    'NCP': nova_instancia[7],
    'CAEC': nova_instancia[8],
    'SMOKE': nova_instancia[9],
    'CH2O': nova_instancia[10],
    'SCC': nova_instancia[11],
    'FAF': nova_instancia[12],
    'TUE': nova_instancia[13],
    'CALC': nova_instancia[14],
    'MTRANS': nova_instancia[15],
    'NObeyesdad': nova_instancia[16]
}

# Convertendo para DataFrame
nova_instancia_df = pd.DataFrame([nova_instancia_dict])

# Separar dados numéricos e categóricos
nova_instancia_numericos = nova_instancia_df.drop(columns=['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad'])
nova_instancia_categoricos = nova_instancia_df[['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad']]


############################
############################
# Normalizar dados numéricos
nova_instancia_numericos_normalizados = modelo_normalizador.transform(nova_instancia_numericos)
# Aplicar one-hot encoding aos dados categóricos
nova_instancia_categoricos_normalizados = pd.get_dummies(nova_instancia_categoricos, dtype=int)
############################
############################

# Juntar dados normalizados e codificados
# Gera um unico dataframe com todas as tabelas, primeiro existem no pd df as tabelas numericas e entao sao inseridas as tabelas categoricas
nova_instancia_final = pd.DataFrame(data=nova_instancia_numericos_normalizados, columns=nova_instancia_numericos.columns).join(nova_instancia_categoricos_normalizados)


# Troca colunas nulas por 0
for column in nova_instancia_final.columns:
    if column in data_frame.columns:
        data_frame.loc[0, column] = nova_instancia_final.loc[0, column]
data_frame = data_frame.fillna(0)
pd.set_option('display.max_columns', None)



#pega a primeira instancia da tabela
nova_instancia_final_df = data_frame.iloc[0]

#pega os valores da instancia (esquerda para direita)
nova_instancia_final_df = nova_instancia_final_df.values


# nova instancia final em teoria deve estar com PRIMEIRO todos os valores numericos e então todos os valores categoricos


print("Indice do grupo do novo entrevistado:",EOLBOEHPC_kmeans_model.predict([nova_instancia_final_df]))
print("Centroide do entrevistado: ", EOLBOEHPC_kmeans_model.cluster_centers_[EOLBOEHPC_kmeans_model.predict([nova_instancia_final_df])])

centroid = pd.DataFrame(EOLBOEHPC_kmeans_model.cluster_centers_[EOLBOEHPC_kmeans_model.predict([nova_instancia_final_df])])











print(centroid) #Possui 38 colunas, bate com a quantidade no EOLBOEHPC.csv
print("Quantidade de colunas em centroid:", len(centroid.columns))

#1. Atribuir os rótulos do arquivo de treinamento ao centroid
#2. Segmentar o centroid em numéricos e categóricos
#3. Centroid_numericos = aplicar o inverse transform
#4. Centroid_categoricos = aplicar o pd.from_dummies()


# Atribuir os rótulos do arquivo de treinamento ao centroid
centroid.columns = columns 
print(centroid)
print("Quantidade de colunas em centroid:", len(centroid.columns))


# Segmentar o centroid em numéricos e categóricos
print("##############################################")
centroid_colunas_numericas = centroid.drop(columns=nova_instancia_categoricos_normalizados)
print(centroid_colunas_numericas)

print("----------------------------------------------")

centroid_colunas_categoricas = data_frame.drop(columns=nova_instancia_numericos.columns)
print(centroid_colunas_categoricas)
print("##############################################")

#3. Centroid_numericos = aplicar o inverse transform


#esta dando errado
centroid_colunas_numericas_desnormalizadas = modelo_normalizador.inverse_transform(centroid_colunas_numericas)



#4. Centroid_categoricos = aplicar o pd.from_dummies()


centroid_colunas_categoricas_desnormalizadas = pd.from_dummies(centroid_colunas_categoricas, sep='_', default_category=None)
print(centroid_colunas_categoricas_desnormalizadas)