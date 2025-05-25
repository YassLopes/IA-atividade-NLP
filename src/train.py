import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import datetime

def train_model(model, train_loader, val_loader, num_epochs=10, device="cpu"):
    """
    Treina o modelo de análise de sentimentos usando os conjuntos de treino e validação.
    
    Args:
        model: Modelo a ser treinado
        train_loader: DataLoader para o conjunto de treino
        val_loader: DataLoader para o conjunto de validação
        num_epochs: Número máximo de épocas para treinamento
        device: Dispositivo para processamento (cuda/cpu)
    
    Returns:
        model, trainer: O modelo treinado e o objeto trainer
    """
    # Criar pasta para checkpoints se não existir
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    
    # Configurar callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='best-checkpoint-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        verbose=True,
        mode='min'
    )
    
    # Configurar parâmetros do trainer com base na disponibilidade de GPU
    trainer_kwargs = {
        'max_epochs': num_epochs,
        'callbacks': [checkpoint_callback, early_stopping_callback],
        'log_every_n_steps': 10,
    }
    
    # Configurar o acelerador com base no dispositivo
    if device == "cuda":
        num_gpus = torch.cuda.device_count()
        trainer_kwargs['accelerator'] = 'gpu'
        trainer_kwargs['devices'] = num_gpus
        print(f"Treinando com {num_gpus} GPU(s)")
    else:
        trainer_kwargs['accelerator'] = 'cpu'
        print("Treinando com CPU")
    
    # Inicializar o trainer
    trainer = pl.Trainer(**trainer_kwargs)
    
    # Treinar o modelo
    start_time = datetime.datetime.now()
    print(f"Iniciando treinamento às {start_time.strftime('%H:%M:%S')}")
    
    trainer.fit(model, train_loader, val_loader)
    
    end_time = datetime.datetime.now()
    training_time = end_time - start_time
    print(f"Treinamento concluído! Tempo total: {training_time}")
    
    # Carregar o melhor modelo baseado no checkpoint
    best_model_path = checkpoint_callback.best_model_path
    print(f"\nMelhor modelo salvo em: {best_model_path}")
    
    if best_model_path:
        # Usar a classe do modelo para carregar o checkpoint, não a instância
        model_class = type(model)
        trained_model = model_class.load_from_checkpoint(best_model_path)
        print(f"Carregado o melhor modelo (val_loss: {checkpoint_callback.best_model_score:.4f})")
        return trained_model, trainer
    
    return model, trainer 