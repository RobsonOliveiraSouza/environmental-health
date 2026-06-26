# Environmental Health - Air Quality Classification

## Project Overview

Este projeto implementa e avalia modelos de Aprendizado de Máquina para classificar a qualidade do ar de diversas cidades globais com base em fatores climáticos e poluentes atmosféricos. 

O repositório foi estruturado para suportar três implementações independentes de Machine Learning desenvolvidas por diferentes membros da equipe, garantindo que todas compartilhem exatamente o mesmo pipeline de preprocessamento, seleção de features e metodologia de avaliação.

---

## Repository Structure

O repositório é organizado da seguinte forma:

```
environmental-health/
│
├── common/                          # Módulos compartilhados e reutilizáveis
│   ├── __init__.py
│   ├── preprocessing.py             # Carga, limpeza, amostragem e split dos dados
│   ├── feature_selection.py         # Definição das features preditoras e variável alvo
│   ├── evaluation.py                # Métricas de classificação e salvamento de relatórios
│   └── visualization.py             # Geração de gráficos (correlação, distribuição e confusão)
│
├── implementations/                 # Implementações independentes de modelos
│   ├── svm/
│   │   └── main.py                  # Script principal do classificador SVM
│   ├── decision_tree/
│   │   └── main.py                  # Script principal do classificador Decision Tree (Árvore de Decisão)
│   └── neural_network/
│       └── main.py                  # Script principal do classificador Neural Network (Rede Neural)
│
├── data/
│   └── openweather_weather_airpollution_top3cities_per_country_J.csv
│
├── models/                          # Modelos serializados salvos por algoritmo
│   ├── svm/
│   │   └── modelo_svm.pkl
│   ├── decision_tree/
│   └── neural_network/
│
├── outputs/                         # Gráficos e métricas gerados por algoritmo
│   ├── svm/
│   │   ├── class_distribution.png
│   │   ├── correlation_heatmap.png
│   │   ├── confusion_matrix.png
│   │   └── metrics.txt
│   ├── decision_tree/
│   └── neural_network/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Dataset

O projeto utiliza o conjunto de dados `data/openweather_weather_airpollution_top3cities_per_country_J.csv`, que reúne medições meteorológicas e de poluição de cidades de múltiplos países via OpenWeather API.

Todas as três implementações de aprendizado de máquina utilizam exatamente este mesmo conjunto de dados de entrada para garantir a reprodutibilidade dos experimentos e a comparação justa entre os modelos.

---

## Common Methodology

A metodologia de preprocessamento, análise exploratória e avaliação de desempenho é compartilhada e padronizada para todas as implementações:

1. **Carga do Dataset**: Leitura do arquivo bruto utilizando funções comuns.
2. **Definição de Variáveis**:
   - **Variável Alvo (Target)**: `target_aqi_class` (valores ordinais de 1 a 5).
   - **Variáveis Preditoras (Features)**: Exatamente 10 variáveis incluindo poluentes atmosféricos e dados meteorológicos.
3. **Tratamento de Valores Nulos**: Remoção consistente de linhas que contêm valores ausentes em qualquer uma das variáveis selecionadas.
4. **Análise Estatística Descritiva**: Realizada no dataset completo filtrado para obter a estatística de todas as variáveis.
5. **Matriz de Correlação e Fator de Inflação de Variância (VIF)**: Cálculo das relações estatísticas lineares e estudo de multicolinearidade.
6. **Amostragem Estratificada**: Amostragem de **10.000 registros** mantendo as proporções originais das classes do target para o treinamento dos modelos.
7. **Divisão de Dados**: Particionamento dos dados em **80% para treinamento** e **20% para teste**, estratificados pela classe alvo.
8. **Esquema de Validação Cruzada**: `StratifiedKFold` com **10 folds** para busca de hiperparâmetros.
9. **Otimização via GridSearchCV**: Busca em grade sistemática dos hiperparâmetros específicos de cada modelo sobre a partição de treinamento.
10. **Métricas de Avaliação**: Avaliação de desempenho realizada no conjunto de teste utilizando métricas de Acurácia Global, Precisão, Recall, F1-Score (com médias ponderadas e macro) e visualização gráfica via Matriz de Confusão.

---

## Feature Selection Justification

As variáveis preditoras foram selecionadas utilizando três critérios metodológicos complementares:

1. **Relevância Ambiental**: Priorização de poluentes atmosféricos e variáveis meteorológicas diretamente associadas à qualidade do ar. O conjunto selecionado abrange todos os poluentes critérios padrão da EPA/WHO (`pm2_5`, `pm10`, `o3`, `no2`, `so2`, `co`, `no`, `nh3`) e parâmetros climáticos chaves (`temp`, `humidity`) que afetam a dispersão e as reações químicas na atmosfera.
2. **Validação Estatística e Multicolinearidade**: O estudo estatístico das variáveis revelou que não há multicolinearidade problemática, com todos os valores de VIF abaixo do limite crítico de 10:
   - `pm2_5` ($6,80$) e `pm10` ($4,25$) possuem os maiores VIFs devido à sua correlação física natural, mas abaixo do limite de 10.
   - Variáveis como `co` ($3,07$) e `no2` ($2,55$) mostram correlação moderada ligada a fontes comuns de emissão.
   - Poluentes e meteorologia adicionais possuem VIFs muito baixos ($1,2$ a $1,8$).
3. **Prevenção de Vazamento de Dados (Data Leakage)**: Excluímos explicitamente qualquer variável derivada do cálculo do AQI final para evitar o vazamento de informações para as features preditoras. As seguintes variáveis foram excluídas como preditoras:
   - `aqi`
   - `aqi_label`
   - `target_aqi_class`

---

# Machine Learning Implementations

Cada uma das subseções a seguir detalha o comportamento, resultados e execução de cada um dos modelos.

## Support Vector Machine (SVM)

### Descrição do Modelo
A implementação de SVM utiliza o algoritmo `SVC` (Support Vector Classifier) com padronização prévia dos dados através do `StandardScaler` aplicada em pipeline para evitar vazamentos na escala dos dados de treino para o teste.

### Hiperparâmetros Avaliados
- Parâmetro de regularização ($C$): `[0.1, 1, 10]`
- Tipo de Kernel: `['linear', 'rbf']`
- Coeficiente Gamma: `['scale', 'auto']`

### Melhores Parâmetros Encontrados
- **Kernel**: `rbf`
- **C**: `10`
- **Gamma**: `'scale'`

### Desempenho no Teste (2.000 Registros)
- **Acurácia Geral**: **95,75%**
- **Acurácia Média na Validação Cruzada (CV)**: **94,56%**
- **F1-Score Ponderado (Weighted)**: **95,75%**

#### Relatório de Classificação Detalhado:
```
              precision    recall  f1-score   support

    1 (good)       0.98      0.98      0.98       999
    2 (fair)       0.94      0.95      0.95       639
3 (moderate)       0.95      0.89      0.92       217
    4 (poor)       0.89      0.92      0.90        92
5 (very_poor)      0.89      0.92      0.91        53

    accuracy                           0.96      2000
   macro avg       0.93      0.93      0.93      2000
weighted avg       0.96      0.96      0.96      2000
```

### Arquivos e Localizações
- **Diretório de Saídas**: `outputs/svm/` (contém gráficos de correlação, distribuição, matriz de confusão e relatório `metrics.txt`).
- **Modelo Serializado**: `models/svm/modelo_svm.pkl`.

### Execução
O modelo SVM é executado por meio do comando:
```bash
python implementations/svm/main.py
```

---

## Decision Tree

### Objetivo
Classificar a qualidade do ar utilizando um modelo baseado em Árvore de Decisão (`DecisionTreeClassifier`), explorando profundidade máxima e critérios de divisão para obter regras legíveis.

### Execução
O modelo de Árvore de Decisão é iniciado por meio do comando:
```bash
python implementations/decision_tree/main.py
```

*Nota: Esta implementação e seus respectivos resultados serão finalizados e integrados posteriormente por outro membro da equipe.*

---

## Neural Network

### Objetivo
Classificar a qualidade do ar utilizando Redes Neurais Artificiais (Multi-Layer Perceptrons - `MLPClassifier`), otimizando arquiteturas de camadas ocultas e taxas de aprendizado.

### Execução
O modelo de Rede Neural é iniciado por meio do comando:
```bash
python implementations/neural_network/main.py
```

*Nota: Esta implementação e seus respectivos resultados serão finalizados e integrados posteriormente por outro membro da equipe.*

---

## Outputs

Cada implementação gera independentemente os seus próprios artefatos nos diretórios dedicados:
- **Gráficos e Relatórios**: salvos sob `outputs/<nome_do_algoritmo>/`.
- **Modelos Treinados (.pkl)**: salvos sob `models/<nome_do_algoritmo>/`.

---

## Installation

Siga os passos abaixo para preparar o ambiente local e executar qualquer uma das implementações. Todas elas utilizam os módulos compartilhados localizados no diretório `common/`.

### 1. Clonar o Repositório
```bash
git clone https://github.com/RobsonOliveiraSouza/environmental-health.git
cd environmental-health
```

### 2. Criar o Ambiente Virtual

#### Linux/macOS
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows
```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 4. Executar uma Implementação

#### SVM (Support Vector Machine)
```bash
python implementations/svm/main.py
```

#### Decision Tree
```bash
python implementations/decision_tree/main.py
```

#### Neural Network
```bash
python implementations/neural_network/main.py
```

---

## Future Work

O repositório foi construído de forma que a equipe possa testar diferentes classes de algoritmos sob as mesmas condições metodológicas de preparação de dados. No futuro, compararemos o desempenho final dos modelos SVM, Árvore de Decisão e Rede Neural a fim de determinar o melhor classificador para o índice de qualidade do ar com base em métricas de F1-score e tempo de processamento.
