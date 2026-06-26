# Classificação de Qualidade do Ar - Projeto Multimodelo

Este projeto implementa e avalia modelos de Aprendizado de Máquina para classificar a qualidade do ar de diversas cidades globais com base em fatores climáticos e poluentes atmosféricos, utilizando o conjunto de dados da OpenWeather API.

A arquitetura do projeto foi estruturada para suportar múltiplos modelos independentes desenvolvidos por diferentes membros da equipe, reaproveitando módulos comuns para preprocessamento, seleção de features, avaliação e visualização.

---

## Estrutura do Projeto

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

## Como Executar

Para executar o pipeline do modelo SVM a partir do diretório raiz do projeto:

```bash
python implementations/svm/main.py
```

Isso carregará o dataset completo, gerará as análises exploratórias, treinará o modelo SVM otimizado em uma amostra estratificada de 10.000 registros, salvará as figuras e métricas em `outputs/svm/` e o modelo treinado final em `models/svm/modelo_svm.pkl`.

---

## Justificativa de Seleção de Features (Feature Selection Justification)
As variáveis preditoras (features) foram selecionadas utilizando três critérios metodológicos complementares:

1. **Relevância Ambiental**: Priorizamos poluentes atmosféricos e variáveis meteorológicas diretamente associadas à qualidade do ar. O conjunto selecionado abrange todos os poluentes critérios padrão da EPA/WHO (`pm2_5`, `pm10`, `o3`, `no2`, `so2`, `co`, `no`, `nh3`) e parâmetros climáticos chaves (`temp`, `humidity`) que afetam a dispersão e as reações químicas na atmosfera (por exemplo, a formação fotoquímica do ozônio).
2. **Validação Estatística e Multicolinearidade**: Analisamos a correlação linear com a variável alvo e calculamos o **Fator de Inflação de Variância (VIF)**. O estudo revelou que não há multicolinearidade problemática, com todos os valores de VIF abaixo do limite crítico de 10:
   - `pm2_5` ($6,80$) e `pm10` ($4,25$) possuem os maiores VIFs, o que é natural devido à correlação física entre frações de particulados, porém totalmente aceitável.
   - Variáveis como `co` ($3,07$) e `no2` ($2,55$) mostram correlação moderada ligada a fontes comuns de emissão (tráfego de veículos).
   - Poluentes e meteorologia adicionais possuem VIFs muito baixos ($1,2$ a $1,8$).
3. **Prevenção de Vazamento de Dados (Data Leakage)**: Excluímos explicitamente qualquer variável derivada do cálculo do AQI final para evitar o vazamento de informações para as features preditoras. As seguintes variáveis foram excluídas como preditoras:
   - `aqi`
   - `aqi_label`
   - `target_aqi_class`

## Variáveis do Modelo

### Variável Alvo (Target)
- **`target_aqi_class`**: Variável numérica (inteiros de 1 a 5) representando a classe da qualidade do ar (1 = bom, 5 = muito ruim).
- A coluna `aqi_label` correspondente (good, fair, etc.) é utilizada exclusivamente para a interpretação e exibição legível dos resultados.

### Variáveis Preditoras (Features)
Selecionamos exatamente as seguintes 10 variáveis preditoras:
1. **`co`**: Monóxido de Carbono ($\mu g/m^3$)
2. **`no`**: Monóxido de Nitrogênio ($\mu g/m^3$)
3. **`no2`**: Dióxido de Nitrogênio ($\mu g/m^3$)
4. **`o3`**: Ozônio ($\mu g/m^3$)
5. **`so2`**: Dióxido de Enxofre ($\mu g/m^3$)
6. **`pm2_5`**: Material Particulado Fino com diâmetro menor que 2.5 micrometros ($\mu g/m^3$)
7. **`pm10`**: Material Particulado Inalável Grosso ($\mu g/m^3$)
8. **`nh3`**: Amônia ($\mu g/m^3$)
9. **`temp`**: Temperatura (°C)
10. **`humidity`**: Umidade Relativa do Ar (%)

## Metodologia

1. **Análise Exploratória Completa**: Realizamos a análise estatística descritiva (resumo estatístico e contagem de tipos/nulos) e a geração da matriz de correlação das features utilizando o **dataset filtrado completo** (todos os ~37 mil registros). Isso garante a máxima precisão estatística nas análises iniciais do projeto.
2. **Tratamento de Nulos**: Remoção de linhas com valores nulos nas features selecionadas sobre o dataset completo.
3. **Amostragem Estratificada (10.000 registros)**: Apenas para a etapa de modelagem, criamos uma amostra estratificada contendo exatamente **10.000 registros**. 
   - *Justificativa técnica*: Esta abordagem preserva a distribuição original das classes (incluindo as classes minoritárias 4 e 5) e garante que o classificador SVM (que possui complexidade computacional quadrática a cúbica em relação ao número de amostras, $O(N^2)$ a $O(N^3)$) treine e otimize hiperparâmetros via busca em grade de maneira eficiente sem comprometer a representatividade dos dados.
4. **Divisão de Dados**: 80% das amostras de treino (8.000 registros) e 20% de teste (2.000 registros).
5. **Pipeline de Machine Learning**:
   - `StandardScaler`: Padronização das features (média 0 e desvio padrão 1), etapa essencial para o algoritmo SVM.
   - `SVC`: Classificador SVM com suporte à busca de hiperparâmetros.
6. **Validação Cruzada**: `StratifiedKFold` com 10 folds para estabilidade e consistência na avaliação dos hiperparâmetros.
7. **GridSearchCV**: Otimização sistemática dos hiperparâmetros de penalidade ($C$), tipo de kernel e coeficiente gamma.

---

## Resultados do Modelo SVM

### Melhores Hiperparâmetros
- **Kernel**: `rbf` (Função de Base Radial)
- **Regularização ($C$)**: `10`
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
