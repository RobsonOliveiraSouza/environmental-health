# -*- coding: utf-8 -*-
"""
Projeto de Aprendizado de Máquina
Objetivo: Classificar a qualidade do ar usando SVM com Scikit-learn.
Dataset: openweather_weather_airpollution_top3cities_per_country_J.csv
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix,
    ConfusionMatrixDisplay
)

def main():
    # Criar diretórios de saída caso não existam
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # 1. Carregar o dataset completo.
    print("--- Etapa 1: Carregando o Dataset Completo ---")
    dataset_path = 'data/openweather_weather_airpollution_top3cities_per_country_J.csv'
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"O arquivo {dataset_path} não foi encontrado no diretório atual.")
    
    df_raw = pd.read_csv(dataset_path)
    print(f"Dataset bruto carregado. Formato: {df_raw.shape[0]} linhas e {df_raw.shape[1]} colunas.")

    # 2. Selecionar apenas as variáveis preditoras escolhidas e a variável alvo.
    print("\n--- Etapa 2: Seleção de Variáveis Preditoras e Alvo ---")
    target_col = 'target_aqi_class'
    features = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3', 'temp', 'humidity']
    
    # Seleção inicial das colunas
    df_selected = df_raw[features + [target_col]].copy()
    print(f"Variáveis selecionadas: {features} e alvo: {target_col}")

    # 3. Remover valores nulos das variáveis selecionadas.
    print("\n--- Etapa 3: Tratamento de Valores Nulos ---")
    null_counts = df_selected.isnull().sum()
    print("Valores nulos identificados por variável:")
    print(null_counts)
    
    df_clean = df_selected.dropna()
    print(f"Tamanho do dataset filtrado (sem nulos): {df_clean.shape[0]} linhas.")

    # 4. Realizar análise estatística descritiva usando o dataset filtrado completo.
    print("\n--- Etapa 4: Análise Estatística Descritiva (Dataset Filtrado Completo) ---")
    print("\nTipos das variáveis selecionadas:")
    print(df_clean.dtypes)
    
    print("\nResumo Estatístico:")
    print(df_clean.describe())
    
    print("\nDistribuição das Classes de Qualidade do Ar (target_aqi_class):")
    class_counts = df_clean[target_col].value_counts().sort_index()
    print(class_counts)

    # 5. Gerar a matriz de correlação usando o dataset filtrado completo.
    print("\n--- Etapa 5: Geração da Matriz de Correlação (Dataset Filtrado Completo) ---")
    correlation_matrix = df_clean[features].corr()
    print("Matriz de Correlação (Valores):")
    print(correlation_matrix)

    # 6. Apenas após concluir a análise exploratória, criar a amostra estratificada para treinamento do modelo.
    # Aumentamos o tamanho da amostra para 10.000 registros estratificados.
    # Justificativa técnica: 10.000 registros fornecem uma maior representatividade estatística das classes menos frequentes
    # (como classes 4 e 5), permitindo que o classificador SVM estabeleça fronteiras de decisão mais robustas e precisas, 
    # ao mesmo tempo em que mantém o custo computacional do treinamento do SVC sob limites razoáveis de tempo e CPU.
    print("\n--- Etapa 6: Criação da Amostra Estratificada (10.000 registros) ---")
    sample_size = 10000
    _, df_sample = train_test_split(
        df_clean,
        test_size=sample_size,
        stratify=df_clean[target_col],
        random_state=42
    )
    print(f"Tamanho da amostra gerada: {df_sample.shape[0]} registros.")
    print("Proporções das classes na amostra (estratificação mantida):")
    print(df_sample[target_col].value_counts(normalize=True))

    # 7. Separar dados de treino e teste.
    print("\n--- Etapa 7: Divisão Treino e Teste (80% treino, 20% teste) ---")
    X = df_sample[features]
    y = df_sample[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Conjunto de Treino: X={X_train.shape}, y={y_train.shape}")
    print(f"Conjunto de Teste: X={X_test.shape}, y={y_test.shape}")

    # 8. Executar o Pipeline (StandardScaler + SVC).
    print("\n--- Etapa 8: Criação do Pipeline (StandardScaler + SVC) ---")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(random_state=42))
    ])

    # 9. Realizar StratifiedKFold (10 folds).
    print("\n--- Etapa 9: Configurando Validação Cruzada (StratifiedKFold, 10 folds) ---")
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # 10. Executar GridSearchCV.
    print("\n--- Etapa 10: Executando GridSearchCV para Otimização de Hiperparâmetros ---")
    param_grid = {
        'svc__C': [0.1, 1, 10],
        'svc__kernel': ['linear', 'rbf'],
        'svc__gamma': ['scale', 'auto']
    }
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    print("Ajustando GridSearchCV no conjunto de treino...")
    grid_search.fit(X_train, y_train)

    # 11. Avaliar o modelo.
    print("\n--- Etapa 11: Avaliação do Modelo no Conjunto de Teste ---")
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Melhores parâmetros: {grid_search.best_params_}")
    print(f"Acurácia Média na Validação Cruzada (CV): {grid_search.best_score_:.4f}")
    print(f"Acurácia no Teste: {acc:.4f}")
    print(f"F1-Score Ponderado no Teste: {f1_weighted:.4f}")
    print("\nRelatório de Classificação Detalhado:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 12. Gerar gráficos.
    # Geramos os gráficos exploratórios usando o dataset filtrado COMPLETO (Etapas 4 e 5),
    # e os gráficos de modelagem usando os dados de teste (Etapa 11).
    print("\n--- Etapa 12: Gerando e Salvando Gráficos em outputs/ ---")
    
    label_mapping = {
        1: 'good',
        2: 'fair',
        3: 'moderate',
        4: 'poor',
        5: 'very_poor'
    }
    
    # Gráfico 1: Distribuição das classes da variável alvo (dataset completo filtrado)
    plt.figure(figsize=(8, 5))
    class_labels = [label_mapping[i] for i in class_counts.index]
    sns.barplot(x=class_labels, y=class_counts.values, hue=class_labels, legend=False, palette='viridis')
    plt.title('Distribuição das Classes de Qualidade do Ar (Dataset Completo Filtrado)')
    plt.xlabel('Classe de AQI')
    plt.ylabel('Frequência')
    plt.tight_layout()
    plt.savefig('outputs/class_distribution.png')
    plt.close()
    
    # Gráfico 2: Heatmap de correlação (dataset completo filtrado)
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Matriz de Correlação das Variáveis Preditoras (Dataset Completo Filtrado)')
    plt.tight_layout()
    plt.savefig('outputs/correlation_heatmap.png')
    plt.close()
    
    # Gráfico 3: Matriz de Confusão
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[label_mapping[i] for i in sorted(y_test.unique())])
    disp.plot(cmap='Blues', ax=plt.gca(), xticks_rotation=45)
    plt.title('Matriz de Confusão - Classificador SVM')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png')
    plt.close()
    
    # Salvar métricas textuais em outputs/metrics.txt
    metrics_path = 'outputs/metrics.txt'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write("=== METRICAS DE AVALIACAO DO MODELO SVM (10.000 Registros) ===\n\n")
        f.write(f"Melhores parametros encontrados: {grid_search.best_params_}\n")
        f.write(f"Melhor acuracia obtida na Validacao Cruzada (CV): {grid_search.best_score_:.4f}\n\n")
        f.write("--- Desempenho no Conjunto de Teste (2.000 Registros) ---\n")
        f.write(f"Acuracia Global: {acc:.4f}\n\n")
        f.write("Medias Ponderadas (Weighted Averages):\n")
        f.write(f"Precisao: {prec_weighted:.4f}\n")
        f.write(f"Recall: {rec_weighted:.4f}\n")
        f.write(f"F1-Score: {f1_weighted:.4f}\n\n")
        f.write("Medias Simples (Macro Averages):\n")
        f.write(f"Precisao: {prec_macro:.4f}\n")
        f.write(f"Recall: {rec_macro:.4f}\n")
        f.write(f"F1-Score: {f1_macro:.4f}\n\n")
        f.write("--- Relatorio de Classificacao Detalhado ---\n")
        f.write(classification_report(y_test, y_pred, zero_division=0))
        f.write("\n--- Matriz de Confusao ---\n")
        f.write(np.array2string(cm))
    print(f"Métricas salvas em '{metrics_path}'.")

    # 13. Salvar o modelo treinado.
    # O modelo final para produção será treinado sobre todo o conjunto da amostra de 10.000 registros,
    # utilizando os melhores parâmetros hiperotimizados pelo GridSearchCV.
    print("\n--- Etapa 13: Salvando o Modelo Treinado final ---")
    final_model = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(
            C=grid_search.best_params_['svc__C'],
            kernel=grid_search.best_params_['svc__kernel'],
            gamma=grid_search.best_params_['svc__gamma'],
            random_state=42
        ))
    ])
    print("Treinando o modelo de produção final com os melhores hiperparâmetros nas 10.000 amostras...")
    final_model.fit(X, y)
    
    model_filename = 'models/modelo_svm.pkl'
    joblib.dump(final_model, model_filename)
    print(f"Modelo exportado com sucesso para '{model_filename}'.")

if __name__ == '__main__':
    main()
