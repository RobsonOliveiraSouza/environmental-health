# -*- coding: utf-8 -*-
"""
Projeto de Aprendizado de Máquina
Algoritmo: Neural Network (Rede Neural)
"""

import sys
import os

# Adiciona o diretório raiz do projeto ao sys.path para permitir importações de common/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from common.feature_selection import get_features_and_target
from common.preprocessing import load_data, select_and_clean

def main():
    print("Implementação de Rede Neural - Em Breve")
    # Exemplo de uso das funções comuns:
    features, target_col = get_features_and_target()
    print(f"Features comuns disponíveis: {features}")

if __name__ == '__main__':
    main()
