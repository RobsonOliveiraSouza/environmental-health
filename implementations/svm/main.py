# -*- coding: utf-8 -*-
"""
Projeto de Aprendizado de Máquina
Algoritmo: SVM (Support Vector Machine)
Objetivo: Classificar a qualidade do ar usando SVM com Scikit-learn.
"""

import sys
import os
import joblib

# Adiciona o diretório raiz do projeto ao sys.path para permitir importações de common/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from common.feature_selection import get_features_and_target
from common.preprocessing import load_data, select_and_clean, get_stratified_sample, split_train_test
from common.evaluation import evaluate_predictions, save_metrics_to_file
from common.visualization import save_class_distribution_plot, save_correlation_heatmap, save_confusion_matrix_plot

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold

def main():
    # Criar diretórios de saída caso não existam
    os.makedirs('outputs/svm', exist_ok=True)
    os.makedirs('models/svm', exist_ok=True)
    
    # 1. Carregar o dataset completo.
    print("--- Etapa 1: Carregando o Dataset Completo ---")
    dataset_path = 'data/openweather_weather_airpollution_top3cities_per_country_J.csv'
    df_raw = load_data(dataset_path)
    print(f"Dataset bruto carregado. Formato: {df_raw.shape[0]} linhas e {df_raw.shape[1]} colunas.")

    # 2. Selecionar apenas as variáveis preditoras escolhidas e a variável alvo.
    print("\n--- Etapa 2: Seleção de Variáveis Preditoras e Alvo ---")
    features, target_col = get_features_and_target()
    print(f"Variáveis selecionadas: {features} e alvo: {target_col}")

    # 3. Remover valores nulos das variáveis selecionadas.
    print("\n--- Etapa 3: Tratamento de Valores Nulos ---")
    df_clean = select_and_clean(df_raw, features, target_col)
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
    df_sample = get_stratified_sample(df_clean, target_col, sample_size=10000)
    print(f"Tamanho da amostra gerada: {df_sample.shape[0]} registros.")
    print("Proporções das classes na amostra (estratificação mantida):")
    print(df_sample[target_col].value_counts(normalize=True))

    # 7. Separar dados de treino e teste.
    print("\n--- Etapa 7: Divisão Treino e Teste (80% treino, 20% teste) ---")
    X_train, X_test, y_train, y_test = split_train_test(df_sample, features, target_col)
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
    
    metrics = evaluate_predictions(y_test, y_pred)
    
    print(f"Melhores parâmetros: {grid_search.best_params_}")
    print(f"Acurácia Média na Validação Cruzada (CV): {grid_search.best_score_:.4f}")
    print(f"Acurácia no Teste: {metrics['accuracy']:.4f}")
    print(f"F1-Score Ponderado no Teste: {metrics['f1_weighted']:.4f}")
    print("\nRelatório de Classificação Detalhado:")
    print(metrics['classification_report'])

    # 12. Gerar gráficos.
    print("\n--- Etapa 12: Gerando e Salvando Gráficos em outputs/svm/ ---")
    save_class_distribution_plot('outputs/svm/class_distribution.png', class_counts)
    save_correlation_heatmap('outputs/svm/correlation_heatmap.png', correlation_matrix)
    save_confusion_matrix_plot('outputs/svm/confusion_matrix.png', metrics['confusion_matrix'], y_test.unique())
    
    # Salvar métricas textuais em outputs/svm/metrics.txt
    save_metrics_to_file('outputs/svm/metrics.txt', metrics, grid_search.best_params_, grid_search.best_score_)
    print("Métricas e gráficos salvos com sucesso.")

    # 13. Salvar o modelo treinado.
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
    X_full = df_sample[features]
    y_full = df_sample[target_col]
    print("Treinando o modelo de produção final com os melhores hiperparâmetros nas 10.000 amostras...")
    final_model.fit(X_full, y_full)
    
    model_filename = 'models/svm/modelo_svm.pkl'
    joblib.dump(final_model, model_filename)
    print(f"Modelo exportado com sucesso para '{model_filename}'.")

if __name__ == '__main__':
    main()
