# Análise de Sentimentos com Deep Learning

Esta é uma prova de conceito (PoC) que demonstra a implementação de um modelo de análise de sentimentos usando PyTorch Lightning e o dataset IMDB do Kaggle.

## Descrição

Este projeto implementa uma rede neural LSTM bidirecional para classificar resenhas de filmes como positivas ou negativas. O modelo é treinado usando o dataset IMDB, que contém 50.000 resenhas de filmes com rótulos de sentimento.

## Características

- Pré-processamento de texto completo
- Modelo LSTM bidirecional com embeddings
- Implementação com PyTorch Lightning para treinamento eficiente
- Divisão de dados em conjuntos de treino (70%), validação (15%) e teste (15%)
- Métricas de avaliação: acurácia, precisão, recall e F1-score
- Visualizações: matriz de confusão e evolução das métricas durante o treinamento
- Geração automática de relatório em Markdown com métricas finais
- Suporte a GPU para aceleração do treinamento
- Estrutura modular para facilitar manutenção e extensão

## Requisitos

Os requisitos do projeto estão listados no arquivo `requirements.txt`. Para instalá-los, execute:

```bash
pip install -r requirements.txt
```

## Dataset

Este projeto utiliza o [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) do Kaggle.

### Instruções para download do dataset:

1. Acesse o link: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
2. Faça login ou crie uma conta no Kaggle
3. Baixe o arquivo "IMDB Dataset.csv"
4. Coloque o arquivo na pasta raiz deste projeto

## Estrutura do Projeto

O projeto está organizado em uma estrutura modular:

```
.
├── src/        # Módulo principal
│   ├── __init__.py                   # Inicialização do módulo
│   ├── data.py                       # Processamento de dados e dataset
│   ├── model.py                      # Definição do modelo LSTM
│   ├── metrics.py                    # Funções de avaliação e métricas
│   ├── train.py                      # Funções de treinamento
│   └── main.py                       # Script principal do módulo
├── run_sentiment_analysis.py         # Script para executar o módulo
├── requirements.txt                  # Dependências do projeto
└── README.md                         # Documentação
```

## Como executar

Para treinar e avaliar o modelo, você pode executar:

```bash
python run_sentiment_analysis.py
```

Alternativamente, você pode importar e utilizar o módulo em seus próprios scripts:

```python
from src.main import main
main()
```

## Fluxo de Execução

Após a execução, o código irá:
1. Carregar e processar o dataset IMDB
2. Dividir os dados em conjuntos de treino (70%), validação (15%) e teste (15%)
3. Treinar o modelo LSTM bidirecional usando o conjunto de treino
4. Monitorar o desempenho no conjunto de validação durante o treinamento
5. Salvar o melhor modelo com base na perda de validação
6. Avaliar o modelo final no conjunto de teste
7. Gerar um relatório de métricas em formato Markdown
8. Demonstrar a previsão com exemplos de resenhas positivas e negativas

## Métricas e Avaliação

O sistema calcula e salva as seguintes métricas:
- **Acurácia**: proporção de previsões corretas
- **Precisão**: taxa de verdadeiros positivos entre todos os resultados positivos previstos
- **Recall**: taxa de verdadeiros positivos entre todos os resultados realmente positivos
- **F1-score**: média harmônica entre precisão e recall

Um relatório completo é gerado automaticamente no arquivo `metrics_report.md` após a execução.

## Utilização de GPU

O código detecta automaticamente a disponibilidade de GPU e utiliza-a para aceleração se disponível. Você pode verificar qual dispositivo está sendo utilizado nas mensagens de log durante a execução.

## Adaptação para Google Colab

Este código pode ser facilmente adaptado para execução no Google Colab:

1. Crie um novo notebook no Google Colab
2. Faça upload da pasta `src` e do arquivo `run_sentiment_analysis.py`
3. Instale as dependências com `!pip install -r requirements.txt` ou instalando cada pacote individualmente
4. Para baixar o dataset no Colab, você pode usar a API do Kaggle:
   ```python
   !pip install kaggle
   !mkdir -p ~/.kaggle
   !echo '{"username":"seu_usuario","key":"sua_chave_api"}' > ~/.kaggle/kaggle.json
   !chmod 600 ~/.kaggle/kaggle.json
   !kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
   !unzip imdb-dataset-of-50k-movie-reviews.zip
   ```
   (Substitua "seu_usuario" e "sua_chave_api" pelas suas credenciais do Kaggle)
5. Execute o script com `%run run_sentiment_analysis.py`
