import pandas as pd
from pickle import load
from sklearn.preprocessing import MinMaxScaler

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

# Normalizar dados numéricos
nova_instancia_numericos_normalizados = modelo_normalizador.transform(nova_instancia_numericos)


# Aplicar one-hot encoding aos dados categóricos
nova_instancia_categoricos_normalizados = pd.get_dummies(nova_instancia_categoricos, dtype=int)

# Juntar dados normalizados e codificados

nova_instancia_final = pd.DataFrame(data=nova_instancia_numericos_normalizados, columns=nova_instancia_numericos.columns).join(nova_instancia_categoricos_normalizados)

for column in nova_instancia_final.columns:
    if column in data_frame.columns:
        data_frame.loc[0, column] = nova_instancia_final.loc[0, column]
data_frame = data_frame.fillna(0)
pd.set_option('display.max_columns', None)




nova_instancia_final = data_frame.iloc[0]
nova_instancia_final = nova_instancia_final.values



print("Indice do grupo do novo entrevistado:",EOLBOEHPC_kmeans_model.predict([nova_instancia_final]))
print("Centroide do entrevistado: ", EOLBOEHPC_kmeans_model.cluster_centers_[EOLBOEHPC_kmeans_model.predict([nova_instancia_final])])