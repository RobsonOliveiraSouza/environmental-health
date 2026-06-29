# -*- coding: utf-8 -*-
"""
Módulo de Visualização de Dados
Gera gráficos de análise exploratória e resultados de modelos de classificação.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

def save_class_distribution_plot(filepath, class_counts):
    """
    Gera e salva o gráfico de barras da distribuição das classes de qualidade do ar.
    """
    label_mapping = {
        1: 'good',
        2: 'fair',
        3: 'moderate',
        4: 'poor',
        5: 'very_poor'
    }
    plt.figure(figsize=(8, 5))
    class_labels = [label_mapping[i] for i in class_counts.index]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(class_labels)))
    plt.bar(class_labels, class_counts.values, color=colors)
    plt.title('Distribuição das Classes de Qualidade do Ar (Dataset Completo Filtrado)')
    plt.xlabel('Classe de AQI')
    plt.ylabel('Frequência')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def save_correlation_heatmap(filepath, correlation_matrix):
    """
    Gera e salva a matriz de correlação em formato de heatmap.
    """
    plt.figure(figsize=(10, 8))
    matrix = correlation_matrix.values
    image = plt.imshow(matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar(image, fraction=0.046, pad=0.04)
    tick_labels = list(correlation_matrix.columns)
    plt.xticks(range(len(tick_labels)), tick_labels, rotation=45, ha='right')
    plt.yticks(range(len(tick_labels)), tick_labels)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            plt.text(col, row, f'{matrix[row, col]:.2f}', ha='center', va='center', color='black', fontsize=8)
    plt.title('Matriz de Correlação das Variáveis Preditoras (Dataset Completo Filtrado)')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def save_confusion_matrix_plot(filepath, cm, unique_labels):
    """
    Gera e salva a visualização da matriz de confusão.
    """
    label_mapping = {
        1: 'good',
        2: 'fair',
        3: 'moderate',
        4: 'poor',
        5: 'very_poor'
    }
    plt.figure(figsize=(8, 6))
    display_labels = [label_mapping[i] for i in sorted(unique_labels)]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap='Blues', ax=plt.gca(), xticks_rotation=45)
    plt.title('Matriz de Confusão - Classificador SVM')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
