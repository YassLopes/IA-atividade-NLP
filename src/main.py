import torch
import pytorch_lightning as pl
import datetime
import os
import argparse

from src.data import prepare_data, predict_text
from src.model import SentimentClassifier
from src.train import train_model
from src.metrics import evaluate_model, test_model

# Configurações
BATCH_SIZE = 32
MAX_TOKEN_LEN = 256
NUM_EPOCHS = 10
LEARNING_RATE = 2e-5

def setup_environment():
    """
    Configura o ambiente de execução e verifica a disponibilidade de GPU.
    
    Returns:
        device: O dispositivo a ser utilizado (cuda/cpu)
    """
    # Definir seed para reprodutibilidade
    pl.seed_everything(42)
    
    # Verificar disponibilidade de GPU
    if torch.cuda.is_available():
        device = "cuda"
        print(f"GPU disponível: {torch.cuda.get_device_name(0)}")
        print(f"Número de GPUs disponíveis: {torch.cuda.device_count()}")
        print(f"Memória GPU disponível: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        device = "cpu"
        print("GPU não disponível. Usando CPU.")
    
    return device

def find_latest_checkpoint():
    """
    Encontra o checkpoint mais recente na pasta checkpoints.
    
    Returns:
        str: Caminho para o checkpoint mais recente ou None se não encontrar
    """
    if not os.path.exists('checkpoints'):
        return None
    
    checkpoints = [os.path.join('checkpoints', f) for f in os.listdir('checkpoints') if f.endswith('.ckpt')]
    if not checkpoints:
        return None
    
    # Ordena por data de modificação (mais recente primeiro)
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    return latest_checkpoint

def main(train_new_model=False):
    """
    Função principal que executa todo o pipeline de análise de sentimentos.
    
    Args:
        train_new_model: Se True, treina um novo modelo; se False, tenta carregar um existente
    """
    print("\n==== Iniciando análise de sentimentos com IMDB Dataset ====")
    
    # Configurar ambiente
    device = setup_environment()
    print(f"Dispositivo utilizado: {device.upper()}")
    
    # Preparar dados
    train_loader, val_loader, test_loader, tokenizer = prepare_data(
        batch_size=BATCH_SIZE,
        max_token_len=MAX_TOKEN_LEN,
        device=device
    )
    
    # Verificar se devemos treinar ou carregar um modelo existente
    if not train_new_model:
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path:
            print(f"\nCarregando modelo existente: {checkpoint_path}")
            model = SentimentClassifier.load_from_checkpoint(checkpoint_path)
            trainer = None
        else:
            print("\nNenhum checkpoint encontrado. Iniciando treinamento de um novo modelo...")
            train_new_model = True
    
    if train_new_model:
        # Inicializar modelo
        model = SentimentClassifier(learning_rate=LEARNING_RATE)
        
        # Treinar modelo
        print("\nIniciando treinamento do modelo...")
        model, trainer = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=NUM_EPOCHS,
            device=device
        )
    
    # Avaliar modelo no conjunto de validação
    print("\nAvaliando o modelo no conjunto de validação...")
    val_accuracy, val_precision, val_recall, val_f1 = evaluate_model(
        model=model,
        val_loader=val_loader,
        device=device
    )
    
    # Avaliar modelo no conjunto de teste
    print("\nAvaliando o modelo no conjunto de teste...")
    test_accuracy, test_precision, test_recall, test_f1 = test_model(
        model=model,
        test_loader=test_loader,
        device=device
    )
    
    print("\nAvaliação concluída!")
    print(f"Acurácia de validação: {val_accuracy:.4f}")
    print(f"Acurácia de teste: {test_accuracy:.4f}")
    
    # Exemplos de previsão
    print("\nExemplos de previsão:")
    
    example_text_pos = "This movie was amazing, I really enjoyed it and would recommend it to everyone!"
    sentiment_pos = predict_text(model, tokenizer, example_text_pos, MAX_TOKEN_LEN, device)
    print(f"Texto: '{example_text_pos}'")
    print(f"Sentimento previsto: {sentiment_pos}")
    
    example_text_neg = "This movie was terrible, I couldn't even finish watching it. The acting was poor and the plot made no sense."
    sentiment_neg = predict_text(model, tokenizer, example_text_neg, MAX_TOKEN_LEN, device)
    print(f"Texto: '{example_text_neg}'")
    print(f"Sentimento previsto: {sentiment_neg}")

if __name__ == "__main__":
    # Configurar argumentos de linha de comando
    parser = argparse.ArgumentParser(description='Análise de Sentimentos em Resenhas IMDB')
    parser.add_argument('--train', action='store_true', help='Força o treinamento de um novo modelo')
    args = parser.parse_args()
    
    main(train_new_model=args.train) 