# -*- coding: utf-8 -*-
"""
Projeto de Aprendizado de Máquina
Algoritmo: Decision Tree (Árvore de Decisão)
Objetivo: Classificar a qualidade do ar usando DecisionTreeClassifier com Scikit-learn.
"""

import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

# Adiciona o diretório raiz do projeto ao sys.path para permitir importações de common/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATASET_PATH = os.path.join(PROJECT_ROOT, 'data', 'openweather_weather_airpollution_top3cities_per_country_J.csv')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'decision_tree')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'decision_tree')

from common.evaluation import evaluate_predictions
from common.feature_selection import get_features_and_target
from common.preprocessing import get_stratified_sample, load_data, select_and_clean, split_train_test
from common.visualization import save_class_distribution_plot, save_correlation_heatmap


def save_decision_tree_metrics_to_file(filepath, metrics_dict, best_params, cv_score):
    """
    Salva as metricas calculadas para a Arvore de Decisao.
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=== METRICAS DE AVALIACAO DO MODELO ARVORE DE DECISAO (10.000 Registros) ===\n\n")
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


def save_decision_tree_confusion_matrix_plot(filepath, cm, unique_labels):
    """
    Gera e salva a matriz de confusao da Arvore de Decisao.
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
    plt.title('Matriz de Confusao - Classificador Arvore de Decisao')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("--- Etapa 1: Carregando o Dataset Completo ---")
    df_raw = load_data(DATASET_PATH)
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

    print("\n--- Etapa 8: Criacao do Modelo Base (DecisionTreeClassifier) ---")
    model = DecisionTreeClassifier(random_state=42)

    print("\n--- Etapa 9: Configurando Validacao Cruzada (StratifiedKFold, 10 folds) ---")
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    print("\n--- Etapa 10: Executando GridSearchCV para Otimizacao de Hiperparametros ---")
    param_grid = {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': [4, 5, 6],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [4, 8],
        'splitter': ['best', 'random'],
    }

    grid_search = GridSearchCV(
        estimator=model,
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

    print("\n--- Etapa 12: Gerando e Salvando Graficos em outputs/decision_tree/ ---")
    save_class_distribution_plot(os.path.join(OUTPUT_DIR, 'class_distribution.png'), class_counts)
    save_correlation_heatmap(os.path.join(OUTPUT_DIR, 'correlation_heatmap.png'), correlation_matrix)
    save_decision_tree_confusion_matrix_plot(
        os.path.join(OUTPUT_DIR, 'confusion_matrix.png'),
        metrics['confusion_matrix'],
        y_test.unique(),
    )

    save_decision_tree_metrics_to_file(
        os.path.join(OUTPUT_DIR, 'metrics.txt'),
        metrics,
        grid_search.best_params_,
        grid_search.best_score_,
    )
    print("Metricas e graficos salvos com sucesso.")

    print("\n--- Etapa 13: Salvando o Modelo Treinado final ---")
    final_model = DecisionTreeClassifier(
        criterion=grid_search.best_params_['criterion'],
        max_depth=grid_search.best_params_['max_depth'],
        min_samples_split=grid_search.best_params_['min_samples_split'],
        min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
        splitter=grid_search.best_params_['splitter'],
        random_state=42,
    )
    X_full = df_sample[features]
    y_full = df_sample[target_col]
    print("Treinando o modelo de producao final com os melhores hiperparametros nas 10.000 amostras...")
    final_model.fit(X_full, y_full)

    model_filename = os.path.join(MODEL_DIR, 'modelo_decision_tree.pkl')
    joblib.dump(final_model, model_filename)
    print(f"Modelo exportado com sucesso para '{model_filename}'.")

if __name__ == '__main__':
    main()
