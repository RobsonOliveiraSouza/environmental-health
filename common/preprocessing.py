# -*- coding: utf-8 -*-
"""
Módulo de Preprocessamento
Responsável por carregar o dataset, tratar valores nulos, realizar amostragem e dividir os dados.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """
    Carrega o dataset a partir do caminho do arquivo CSV.
    """
    return pd.read_csv(filepath)

def select_and_clean(df, features, target_col):
    """
    Seleciona as colunas de interesse e remove linhas com valores nulos.
    """
    df_selected = df[features + [target_col]].copy()
    df_clean = df_selected.dropna()
    return df_clean

def get_stratified_sample(df, target_col, sample_size=10000, random_state=42):
    """
    Gera uma amostra estratificada mantendo as proporções das classes do target.
    """
    _, df_sample = train_test_split(
        df,
        test_size=sample_size,
        stratify=df[target_col],
        random_state=random_state
    )
    return df_sample

def split_train_test(df_sample, features, target_col, test_size=0.2, random_state=42):
    """
    Divide os dados da amostra em conjuntos de treino e teste de forma estratificada.
    """
    X = df_sample[features]
    y = df_sample[target_col]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
