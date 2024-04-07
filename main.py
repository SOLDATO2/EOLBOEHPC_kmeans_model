import pandas as pd

# Caminho para o arquivo CSV contendo os nomes das colunas
caminho_csv = 'EOLBOEHPC_treinamento\\EOLBOEHPC.csv'

# Abrir o arquivo CSV e extrair os nomes das colunas
with open(caminho_csv, 'r') as file:
    colunas = file.readline().strip().split(',')

# Criar um DataFrame vazio com as colunas extraídas do CSV
novo_df = pd.DataFrame(columns=colunas)

# Nova instância
nova_instancia = [20, 1.59, 60, 2.0, 3.0, 2.0, 2.0, 0.6, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]


# Converter a lista em uma única linha de DataFrame
nova_instancia_df = pd.DataFrame([nova_instancia], columns=colunas)

# Adicionar a nova instância ao DataFrame existente
novo_df = novo_df._append(nova_instancia_df, ignore_index=True)

# Exibir o DataFrame resultante
print(novo_df)


from pickle import load
EOLBOEHPC_kmeans_model = load(open('EOLBOEHPC_treinamento\\EOLBOEHPC_clusters_2024.pkl', 'rb'))


print("Indice do grupo do novo entrevistado:",EOLBOEHPC_kmeans_model.predict([nova_instancia]))
print("Centroide do entrevistado: ", EOLBOEHPC_kmeans_model.cluster_centers_[EOLBOEHPC_kmeans_model.predict([nova_instancia])])








