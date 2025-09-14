# SISTEMA DE RACIOCÍNIO BASEADO EM CASOS (RBC)
# Implementação para a Atividade Avaliativa 1

# Importando as bibliotecas necessárias
# pandas é usado para gerenciar os dados em formato de DataFrame (tabela)
# scikit-learn fornece o dataset Iris e as ferramentas para o modelo e avaliação
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# 1. Carregamento e Preparação do Dataset Iris
# O Iris Dataset será nossa base de casos passados.
# Ele tem 4 atributos numéricos (características da flor) e a classe (espécie) como solução.
print("1. Carregando e preparando o Dataset Iris...")
iris = load_iris()
# Cria um DataFrame com os atributos da flor
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# Cria uma Series com as classes (solução)
y = pd.Series(iris.target)

# 2. Divisão da Base de Dados em Treino e Teste
# Dividimos os dados em 80% para treino (nossa 'memória' de casos)
# e 20% para teste (os 'novos problemas' a serem resolvidos).
print("2. Dividindo o dataset em conjuntos de treino (80%) e teste (20%)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Exibe o número de casos em cada conjunto para comprovação
print(f"   - Casos para treinamento (memória): {len(X_train)}")
print(f"   - Casos para teste (novos problemas): {len(X_test)}\n")

# 3. Implementação do Sistema RBC com K-Nearest Neighbors (KNN)
# O KNN com n_neighbors=1 simula o processo do RBC.
# Ele encontra o vizinho (caso) mais próximo e usa a solução dele.
# A métrica 'euclidean' é a nossa medida de similaridade (Distância Euclidiana).
print("3. Implementando o sistema RBC com KNN e Distância Euclidiana...")
rbc_model = KNeighborsClassifier(n_neighbors=1, metric='euclidean')

# 4. Treinamento do Sistema
# Para o KNN, o "treino" é simplesmente armazenar os dados de treino.
# O modelo não aprende um padrão complexo, ele apenas memoriza os casos.
print("4. Treinando o sistema com a base de casos passados...")
rbc_model.fit(X_train, y_train)

# 5. Execução do Sistema nos Novos Casos
# O sistema agora usa a sua "memória" para prever a solução dos casos de teste.
print("5. Executando o sistema para prever a classe dos casos de teste...")
y_pred = rbc_model.predict(X_test)

# 6. Avaliação e Apresentação dos Resultados
# Calculamos a acurácia para medir o desempenho do sistema.
acuracia = accuracy_score(y_test, y_pred)
# O classification_report dá um resumo detalhado de outras métricas
relatorio = classification_report(y_test, y_pred, target_names=iris.target_names)

print("\n--- RESULTADOS DO SISTEMA DE RBC ---")
print(f"Acurácia do Sistema: {acuracia:.2f}")
print("-------------------------------------")
print("Relatório de Classificação Detalhado:\n", relatorio)
print("-------------------------------------")