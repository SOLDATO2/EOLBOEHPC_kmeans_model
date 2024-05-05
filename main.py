import pandas as pd
from pickle import load
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import pandas as pd

def undummify(df):
    # Create an empty dictionary to store the undummified values
    undummified_dict = {}
    assigned_categories = set()  # Keep track of assigned categories
    
    for col in df.columns:
        # Split the column name by underscores
        parts = col.split("_")
        
        # Extract the original category name (all parts except the last one)
        category_name = "_".join(parts[:-1])
        
        # Get the value (1 or 0) from the dummy-encoded column
        value = df[col]
        
        # Check if the category name has already been assigned a value
        if category_name not in assigned_categories:
            if value[0] == 1:
                undummified_value = parts[-1]
                assigned_categories.add(category_name)  # Mark as assigned
                undummified_dict[category_name] = undummified_value
            else:
                undummified_value = None  # Handle cases where value is 0
                undummified_dict[category_name] = undummified_value
        
        # Store the undummified value in the dictionary

    
    # Create a new DataFrame from the dictionary
    undummified_df = pd.DataFrame(undummified_dict, index=[0])
    
    return undummified_df


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


# Iterando pela lista e modificando os elementos
for i in range(len(nova_instancia)):
    if isinstance(nova_instancia[i], str) and "_" in nova_instancia[i]:
        nova_instancia[i] = nova_instancia[i].replace("_", "")

print(nova_instancia)



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
print(nova_instancia_df)

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
print(nova_instancia_categoricos)
nova_instancia_categoricos_normalizados = pd.get_dummies(nova_instancia_categoricos, dtype=int) #Esta aplicando o get dummies corretamente
print(nova_instancia_categoricos_normalizados)
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
print(instancia_normalizada_do_df_normalizado_organizado)
#pega os valores da instancia (esquerda para direita)
valores_instancia_normalizada_do_df_normalizado_organizado = instancia_normalizada_do_df_normalizado_organizado.values
print(valores_instancia_normalizada_do_df_normalizado_organizado)

#armazena colunas para dar ao centroid futuramente
ordem_colunas = nova_instancia_final_normalizada_ORGANIZADA_df.columns
print(ordem_colunas)

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
centroid.columns = ordem_colunas
print(centroid.columns)
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
    
print(lista_colunas_categoricas_normalizadas_para_drop_centroid)
# Remover duplicatas da lista, se houver
#colunas_categoricas_normalizadas_para_drop_centroid = list(set(lista_colunas_categoricas_normalizadas_para_drop_centroid))
#print(len(colunas_categoricas_normalizadas_para_drop_centroid))



centroid_colunas_numericas_normalizadas = centroid.drop(columns=lista_colunas_categoricas_normalizadas_para_drop_centroid)
print(centroid_colunas_numericas_normalizadas)
print("----------------------------------------------")

centroid_colunas_categoricas_normalizadas = centroid[lista_colunas_categoricas_normalizadas_para_drop_centroid]
print(centroid_colunas_categoricas_normalizadas)

print("----------------------------------------------")
print("##############################################")


centroid_colunas_numericas_desnormalizadas = modelo_normalizador.inverse_transform(centroid_colunas_numericas_normalizadas)
print(centroid_colunas_numericas_desnormalizadas)

#4. Centroid_categoricos = aplicar o pd.from_dummies()

print(centroid_colunas_categoricas_normalizadas)
print("----------------------------------------------")
#centroid_colunas_categoricas_normalizadas = centroid_colunas_categoricas_normalizadas.round()
centroid_colunas_categoricas_normalizadas = centroid_colunas_categoricas_normalizadas.applymap(lambda x: 1 if x >= 0.45 else 0)
centroid_colunas_categoricas_normalizadas = centroid_colunas_categoricas_normalizadas.astype(int)
print(centroid_colunas_categoricas_normalizadas.iloc[0])
#centroid_colunas_categoricas_desnormalizadas = pd.from_dummies(centroid_colunas_categoricas_normalizadas)

centroid_colunas_categoricas_desnormalizadas = undummify(centroid_colunas_categoricas_normalizadas)
print("----------------------------------------------")

print(centroid_colunas_categoricas_desnormalizadas)



nova_instancia_final_normalizada_df = pd.DataFrame(data=centroid_colunas_numericas_desnormalizadas.round(), columns=centroid_colunas_numericas_normalizadas.columns).join(centroid_colunas_categoricas_desnormalizadas)


print("----------------------------------------------")


print(nova_instancia_final_normalizada_df)
#















