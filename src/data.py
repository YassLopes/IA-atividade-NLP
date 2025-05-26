import os
import re
import string
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords

# Download de recursos necessários do NLTK
nltk.download('stopwords', quiet=True)

class IMDBDataset(Dataset):
    """
    Dataset para o IMDB que processa resenhas de filmes para análise de sentimentos.
    """
    def __init__(self, reviews, targets, tokenizer, max_length):
        """
        Inicializa o dataset.
        
        Args:
            reviews: Lista de resenhas de texto
            targets: Lista de rótulos (0 para negativo, 1 para positivo)
            tokenizer: Tokenizador para processamento de texto
            max_length: Comprimento máximo da sequência
        """
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]
        
        # Pré-processamento do texto
        review = self.preprocess_text(review)
        
        # Tokenização do texto
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }
    
    def preprocess_text(self, text):
        """
        Pré-processa o texto aplicando transformações como remoção 
        de HTML, pontuação, números e espaços extras.
        """
        # Converter para minúsculo
        text = text.lower()
        
        # Remover HTML tags
        text = re.sub('<.*?>', '', text)
        
        # Remover pontuação
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remover números
        text = re.sub(r'\d+', '', text)
        
        # Remover espaços extras
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

def prepare_data(batch_size, max_token_len, device):
    """
    Prepara o dataset IMDB para análise de sentimentos.
    
    Args:
        batch_size: Tamanho do batch para os dataloaders
        max_token_len: Comprimento máximo do token para tokenização
        device: Dispositivo para processamento (cuda/cpu)
    
    Returns:
        train_loader, val_loader, test_loader, tokenizer: Os dataloaders e o tokenizer
    """
    # Nome do arquivo que deve ser baixado do Kaggle
    kaggle_file = 'IMDB Dataset.csv'
    
    # Verificar se o dataset foi baixado
    if not os.path.exists(kaggle_file):
        print(f"ERRO: O arquivo '{kaggle_file}' não foi encontrado.")
        print("Por favor, baixe o dataset do Kaggle manualmente:")
        print("1. Acesse: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
        print("2. Baixe o arquivo 'IMDB Dataset.csv'")
        print("3. Coloque o arquivo na pasta raiz do projeto")
        print("4. Execute este script novamente")
        exit(1)
    
    print(f"Carregando o dataset IMDB do Kaggle...")
    df = pd.read_csv(kaggle_file)
    
    print(f"Dataset carregado com {len(df)} amostras")
    
    # Converter sentimentos para valores numéricos
    sentiment_mapping = {'positive': 1, 'negative': 0}
    df['sentiment'] = df['sentiment'].map(sentiment_mapping)
    
    # Dividir em treino, validação e teste (70%, 15%, 15%)
    # Primeiro, dividimos em treino e temp (85% / 15%)
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
    # Em seguida, dividimos temp em validação e teste (70% / 15%)
    train_df, val_df = train_test_split(train_df, test_size=0.1765, random_state=42)  # 0.15/0.85 = 0.1765
    
    print(f"Conjunto de treino: {len(train_df)} amostras ({len(train_df)/len(df):.1%})")
    print(f"Conjunto de validação: {len(val_df)} amostras ({len(val_df)/len(df):.1%})")
    print(f"Conjunto de teste: {len(test_df)} amostras ({len(test_df)/len(df):.1%})")
    
    # Inicializar o tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Criar datasets
    train_dataset = IMDBDataset(
        reviews=train_df['review'].to_numpy(),
        targets=train_df['sentiment'].to_numpy(),
        tokenizer=tokenizer,
        max_length=max_token_len
    )
    
    val_dataset = IMDBDataset(
        reviews=val_df['review'].to_numpy(),
        targets=val_df['sentiment'].to_numpy(),
        tokenizer=tokenizer,
        max_length=max_token_len
    )
    
    test_dataset = IMDBDataset(
        reviews=test_df['review'].to_numpy(),
        targets=test_df['sentiment'].to_numpy(),
        tokenizer=tokenizer,
        max_length=max_token_len
    )
    
    # Calcular os workers de forma otimizada com base no hardware disponível
    num_workers = min(os.cpu_count() or 4, 4)  # Limitar a no máximo 4 workers
    
    # Criar dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False  # Melhora o desempenho com GPU
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False
    )
    
    return train_loader, val_loader, test_loader, tokenizer

def predict_text(model, tokenizer, text, max_token_len, device):
    """
    Faz a previsão de sentimento para um texto.
    
    Args:
        model: O modelo treinado
        tokenizer: O tokenizador
        text: O texto para classificar
        max_token_len: Comprimento máximo do token
        device: Dispositivo para processamento (cuda/cpu)
    
    Returns:
        sentiment: O sentimento previsto ("positivo" ou "negativo")
    """
    # Pré-processar e tokenizar
    processed_text = re.sub('<.*?>', '', text.lower())
    processed_text = processed_text.translate(str.maketrans('', '', string.punctuation))
    processed_text = re.sub(r'\d+', '', processed_text)
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    
    encoding = tokenizer.encode_plus(
        processed_text,
        add_special_tokens=True,
        max_length=max_token_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    # Fazer previsão
    model.eval()
    model = model.to(device)  # Mover modelo para GPU se disponível
    
    with torch.no_grad():
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        output = model(input_ids, attention_mask)
        _, prediction = torch.max(output, dim=1)
        
    sentiment = "positivo" if prediction.item() == 1 else "negativo"
    return sentiment 