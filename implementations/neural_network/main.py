# -*- coding: utf-8 -*-
"""
Projeto de Aprendizado de Maquina
Algoritmo: Neural Network (Rede Neural)
Objetivo: Classificar a qualidade do ar usando redes neurais com Scikit-learn.
"""

import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Adiciona o diretorio raiz do projeto ao sys.path para permitir importacoes de common/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from common.evaluation import evaluate_predictions
from common.feature_selection import get_features_and_target
from common.preprocessing import (
    get_stratified_sample,
    load_data,
    select_and_clean,
    split_train_test,
)
from common.visualization import (
    save_class_distribution_plot,
    save_correlation_heatmap,
)


def save_neural_network_metrics_to_file(filepath, metrics_dict, best_params, cv_score):
    """
    Salva as metricas calculadas para a Rede Neural.
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=== METRICAS DE AVALIACAO DO MODELO REDE NEURAL (10.000 Registros) ===\n\n")
        f.write(f"Melhores parametros encontrados: {best_params}\n")
        f.write(f"Melhor acuracia obtida na Validacao Cruzada (CV): {cv_score:.4f}\n\n")
        f.write("--- Desempenho no Conjunto de Teste (2.000 Registros) ---\n")
        f.write(f"Acuracia Global: {metrics_dict['accuracy']:.4f}\n\n")
        f.write("Medias Ponderadas (Weighted Averages):\n")
        f.write(f"Precisao: {metrics_dict['precision_weighted']:.4f}\n")
        f.write(f"Recall: {metrics_dict['recall_weighted']:.4f}\n")
        f.write(f"F1-Score: {metrics_dict['f1_weighted']:.4f}\n\n")
        f.write("Medias Simples (Macro Averages):\n")
        f.write(f"Precisao: {metrics_dict['precision_macro']:.4f}\n")
        f.write(f"Recall: {metrics_dict['recall_macro']:.4f}\n")
        f.write(f"F1-Score: {metrics_dict['f1_macro']:.4f}\n\n")
        f.write("--- Relatorio de Classificacao Detalhado ---\n")
        f.write(metrics_dict['classification_report'])
        f.write("\n--- Matriz de Confusao ---\n")
        f.write(np.array2string(metrics_dict['confusion_matrix']))


def save_neural_network_confusion_matrix_plot(filepath, cm, unique_labels):
    """
    Gera e salva a matriz de confusao da Rede Neural.
    """
    label_mapping = {
        1: 'good',
        2: 'fair',
        3: 'moderate',
        4: 'poor',
        5: 'very_poor',
    }
    plt.figure(figsize=(8, 6))
    display_labels = [label_mapping[i] for i in sorted(unique_labels)]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap='Blues', ax=plt.gca(), xticks_rotation=45)
    plt.title('Matriz de Confusao - Classificador Rede Neural')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def main():
    os.makedirs('outputs/neural_network', exist_ok=True)
    os.makedirs('models/neural_network', exist_ok=True)

    print("--- Etapa 1: Carregando o Dataset Completo ---")
    dataset_path = 'data/openweather_weather_airpollution_top3cities_per_country_J.csv'
    df_raw = load_data(dataset_path)
    print(f"Dataset bruto carregado. Formato: {df_raw.shape[0]} linhas e {df_raw.shape[1]} colunas.")

    print("\n--- Etapa 2: Selecao de Variaveis Preditoras e Alvo ---")
    features, target_col = get_features_and_target()
    print(f"Variaveis selecionadas: {features} e alvo: {target_col}")

    print("\n--- Etapa 3: Tratamento de Valores Nulos ---")
    df_clean = select_and_clean(df_raw, features, target_col)
    print(f"Tamanho do dataset filtrado (sem nulos): {df_clean.shape[0]} linhas.")

    print("\n--- Etapa 4: Analise Estatistica Descritiva (Dataset Filtrado Completo) ---")
    print("\nTipos das variaveis selecionadas:")
    print(df_clean.dtypes)

    print("\nResumo Estatistico:")
    print(df_clean.describe())

    print("\nDistribuicao das Classes de Qualidade do Ar (target_aqi_class):")
    class_counts = df_clean[target_col].value_counts().sort_index()
    print(class_counts)

    print("\n--- Etapa 5: Geracao da Matriz de Correlacao (Dataset Filtrado Completo) ---")
    correlation_matrix = df_clean[features].corr()
    print("Matriz de Correlacao (Valores):")
    print(correlation_matrix)

    print("\n--- Etapa 6: Criacao da Amostra Estratificada (10.000 registros) ---")
    df_sample = get_stratified_sample(df_clean, target_col, sample_size=10000)
    print(f"Tamanho da amostra gerada: {df_sample.shape[0]} registros.")
    print("Proporcoes das classes na amostra (estratificacao mantida):")
    print(df_sample[target_col].value_counts(normalize=True))

    print("\n--- Etapa 7: Divisao Treino e Teste (80% treino, 20% teste) ---")
    X_train, X_test, y_train, y_test = split_train_test(df_sample, features, target_col)
    print(f"Conjunto de Treino: X={X_train.shape}, y={y_train.shape}")
    print(f"Conjunto de Teste: X={X_test.shape}, y={y_test.shape}")

    print("\n--- Etapa 8: Criacao do Pipeline (StandardScaler + MLPClassifier) ---")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=42,
        )),
    ])

    print("\n--- Etapa 9: Configurando Validacao Cruzada (StratifiedKFold, 10 folds) ---")
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    print("\n--- Etapa 10: Executando GridSearchCV para Otimizacao de Hiperparametros ---")
    param_grid = {
        'mlp__hidden_layer_sizes': [(32,), (64,), (64, 32)],
        'mlp__activation': ['relu', 'tanh'],
        'mlp__alpha': [0.0001, 0.001],
        'mlp__learning_rate_init': [0.001, 0.01],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
    )

    print("Ajustando GridSearchCV no conjunto de treino...")
    grid_search.fit(X_train, y_train)

    print("\n--- Etapa 11: Avaliacao do Modelo no Conjunto de Teste ---")
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    metrics = evaluate_predictions(y_test, y_pred)

    print(f"Melhores parametros: {grid_search.best_params_}")
    print(f"Acuracia Media na Validacao Cruzada (CV): {grid_search.best_score_:.4f}")
    print(f"Acuracia no Teste: {metrics['accuracy']:.4f}")
    print(f"F1-Score Ponderado no Teste: {metrics['f1_weighted']:.4f}")
    print("\nRelatorio de Classificacao Detalhado:")
    print(metrics['classification_report'])

    print("\n--- Etapa 12: Gerando e Salvando Graficos em outputs/neural_network/ ---")
    save_class_distribution_plot('outputs/neural_network/class_distribution.png', class_counts)
    save_correlation_heatmap('outputs/neural_network/correlation_heatmap.png', correlation_matrix)
    save_neural_network_confusion_matrix_plot(
        'outputs/neural_network/confusion_matrix.png',
        metrics['confusion_matrix'],
        y_test.unique()
    )

    save_neural_network_metrics_to_file(
        'outputs/neural_network/metrics.txt',
        metrics,
        grid_search.best_params_,
        grid_search.best_score_
    )
    print("Metricas e graficos salvos com sucesso.")

    print("\n--- Etapa 13: Salvando o Modelo Treinado final ---")
    final_model = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=grid_search.best_params_['mlp__hidden_layer_sizes'],
            activation=grid_search.best_params_['mlp__activation'],
            alpha=grid_search.best_params_['mlp__alpha'],
            learning_rate_init=grid_search.best_params_['mlp__learning_rate_init'],
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=42,
        )),
    ])
    X_full = df_sample[features]
    y_full = df_sample[target_col]
    print("Treinando o modelo de producao final com os melhores hiperparametros nas 10.000 amostras...")
    final_model.fit(X_full, y_full)

    model_filename = 'models/neural_network/modelo_neural_network.pkl'
    joblib.dump(final_model, model_filename)
    print(f"Modelo exportado com sucesso para '{model_filename}'.")


if __name__ == '__main__':
    main()
