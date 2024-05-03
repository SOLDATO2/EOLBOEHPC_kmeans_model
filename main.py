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
nova_instancia_df = pd.DataFrame([nova_instancia_dict]) #Até aqui ele está inserindo os dados corretamente


# Separar dados numéricos e categóricos
nova_instancia_numericos = nova_instancia_df.drop(columns=['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad'])
nova_instancia_categoricos = nova_instancia_df[['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad']]

#print(nova_instancia_df)
#print(nova_instancia_numericos) ###################dados categoricos possuem 8 colunas##############################
#print(nova_instancia_categoricos)


############################
############################
# Normalizar dados numéricos
nova_instancia_numericos_normalizados = modelo_normalizador.transform(nova_instancia_numericos) ##############continua tendo 8 elementos#############
print(len(nova_instancia_numericos_normalizados[0]))
# Aplicar one-hot encoding aos dados categóricos
nova_instancia_categoricos_normalizados = pd.get_dummies(nova_instancia_categoricos, dtype=int) #Esta aplicando o get dummies corretamente
print(len(nova_instancia_categoricos_normalizados.columns))
############################
############################
pd.set_option('display.max_columns', None)
# Juntar dados normalizados e codificados
# Gera um unico dataframe com todas as tabelas, primeiro existem no pd df as tabelas numericas e entao sao inseridas as tabelas categoricas
nova_instancia_final_normalizada_df = pd.DataFrame(data=nova_instancia_numericos_normalizados, columns=nova_instancia_numericos.columns).join(nova_instancia_categoricos_normalizados)
print(len(nova_instancia_final_normalizada_df.columns)) #Possui 17 colunas, ou seja, está juntando corretamente os numericos com os categoricos
print(nova_instancia_final_normalizada_df)


################################################

#O PROBLEMA ESTÁ AQUI, apos fazer o join em 'nova_instancia_final_normalizada_df', ainda falta as outras colunas dos dados
#categricos que não foram incluidas apos o get dummies

################################################


# Copiar o DataFrame original para manter a ordem das colunas
# Alimentar as colunas existentes
nova_instancia_final_normalizada_ORGANIZADA_df = data_frame.copy()
nova_instancia_final_normalizada_ORGANIZADA_df[nova_instancia_final_normalizada_df.columns] = nova_instancia_final_normalizada_df
# Preencher as colunas faltantes com zeros
print(nova_instancia_final_normalizada_ORGANIZADA_df)
nova_instancia_final_normalizada_ORGANIZADA_df = nova_instancia_final_normalizada_ORGANIZADA_df.fillna(0)
print(nova_instancia_final_normalizada_ORGANIZADA_df)

# Troca colunas nulas por 0



#pega a primeira instancia da tabela
instancia_normalizada_do_df_normalizado_organizado = nova_instancia_final_normalizada_ORGANIZADA_df.iloc[0]

#pega os valores da instancia (esquerda para direita)
valores_instancia_normalizada_do_df_normalizado_organizado = instancia_normalizada_do_df_normalizado_organizado.values

#print(len(valores_instancia_normalizada_do_df_normalizado_organizado)) #Possui 38 elementos, ou seja, 38 valores para 38 colunas
#Contei manualmente as colunas normalizadas em "EOLBOEHPC.csv" e de fato existem 38 colunas após a normalização


# nova instancia final em teoria deve estar com PRIMEIRO todos os valores numericos e então todos os valores categoricos


print("Indice do grupo do novo entrevistado:",EOLBOEHPC_kmeans_model.predict([valores_instancia_normalizada_do_df_normalizado_organizado]))
print("Centroide do entrevistado: ", EOLBOEHPC_kmeans_model.cluster_centers_[EOLBOEHPC_kmeans_model.predict([valores_instancia_normalizada_do_df_normalizado_organizado])])

centroid = pd.DataFrame(EOLBOEHPC_kmeans_model.cluster_centers_[EOLBOEHPC_kmeans_model.predict([valores_instancia_normalizada_do_df_normalizado_organizado])])











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


# Segmentar o centroid em numéricos e categóricos <---------------------------------------------- problema pode estar aqui
print("##############################################")
lista_colunas_categoricas_normalizadas_para_drop_centroid = []

# Iterar sobre as colunas do DataFrame categorico normalizado
for coluna in nova_instancia_categoricos_normalizados.columns:
    # Extrair o prefixo da coluna atual
    prefixo = coluna.split('_')[0]
    # Filtrar as colunas do DataFrame final normalizado que têm o mesmo prefixo
    colunas_prefixo = list(filter(lambda x: x.startswith(prefixo), nova_instancia_final_normalizada_ORGANIZADA_df.columns))
    # Adicionar as colunas encontradas à lista
    lista_colunas_categoricas_normalizadas_para_drop_centroid.extend(colunas_prefixo)
    
print(len(lista_colunas_categoricas_normalizadas_para_drop_centroid))
# Remover duplicatas da lista, se houver
colunas_categoricas_normalizadas_para_drop_centroid = list(set(lista_colunas_categoricas_normalizadas_para_drop_centroid))
print(len(colunas_categoricas_normalizadas_para_drop_centroid))



centroid_colunas_numericas_normalizadas = centroid.drop(columns=colunas_categoricas_normalizadas_para_drop_centroid)
print(centroid_colunas_numericas_normalizadas)


centroid_colunas_categoricas_normalizadas = centroid[colunas_categoricas_normalizadas_para_drop_centroid]
print(centroid_colunas_categoricas_normalizadas)

print("----------------------------------------------")
print("##############################################")

#3. Centroid_numericos = aplicar o inverse transform

#esta dando errado
centroid_colunas_numericas_desnormalizadas = modelo_normalizador.inverse_transform(centroid_colunas_numericas_normalizadas)

print(centroid_colunas_numericas_desnormalizadas)

#4. Centroid_categoricos = aplicar o pd.from_dummies()


centroid_colunas_categoricas_desnormalizadas = pd.from_dummies(centroid_colunas_categoricas, sep='_', default_category=None)
print(centroid_colunas_categoricas_desnormalizadas)