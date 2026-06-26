# -*- coding: utf-8 -*-
"""
Módulo de Seleção de Features
Define as variáveis preditoras (features) e a variável alvo (target) do projeto.
"""

def get_features_and_target():
    """
    Retorna a lista de variáveis preditoras selecionadas e o nome da variável alvo.
    """
    target_col = 'target_aqi_class'
    features = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3', 'temp', 'humidity']
    return features, target_col
