# -*- coding: utf-8 -*-
"""
Módulo de Avaliação de Modelos
Calcula métricas de classificação e gera o relatório detalhado de desempenho.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def evaluate_predictions(y_true, y_pred):
    """
    Calcula acurácia, precisão, recall, f1-score (ponderados e macro) e matriz de confusão.
    """
    acc = accuracy_score(y_true, y_pred)
    prec_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)
    
    return {
        'accuracy': acc,
        'precision_weighted': prec_weighted,
        'recall_weighted': rec_weighted,
        'f1_weighted': f1_weighted,
        'precision_macro': prec_macro,
        'recall_macro': rec_macro,
        'f1_macro': f1_macro,
        'confusion_matrix': cm,
        'classification_report': report
    }

def save_metrics_to_file(filepath, metrics_dict, best_params, cv_score):
    """
    Salva as métricas calculadas em um arquivo de texto.
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=== METRICAS DE AVALIACAO DO MODELO SVM (10.000 Registros) ===\n\n")
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
