import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class SentimentClassifier(pl.LightningModule):
    """
    Modelo de classificação de sentimentos baseado em LSTM bidirecional.
    """
    def __init__(self, learning_rate=2e-5, n_classes=2):
        """
        Inicializa o modelo de classificação.
        
        Args:
            learning_rate: Taxa de aprendizado para o otimizador
            n_classes: Número de classes para classificação
        """
        super(SentimentClassifier, self).__init__()
        
        self.learning_rate = learning_rate
        
        # Camada de embedding
        self.embedding = nn.Embedding(30522, 768)  # 30522 é o tamanho do vocabulário do BERT
        
        # Camadas LSTM
        self.lstm = nn.LSTM(768, 256, batch_first=True, bidirectional=True)
        
        # Camadas fully connected
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, n_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.3)
        
        # Métricas para validação
        self.val_accuracy = []
        self.val_precision = []
        self.val_recall = []
        self.val_f1 = []
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass do modelo.
        
        Args:
            input_ids: IDs dos tokens de entrada
            attention_mask: Máscara de atenção para os tokens
            
        Returns:
            output: Logits de saída do modelo
        """
        # Embeddings
        embedded = self.embedding(input_ids)
        
        # Aplicar a máscara de atenção ao embedding
        embedded = embedded * attention_mask.unsqueeze(2)
        
        # LSTM
        lstm_output, (hidden, cell) = self.lstm(embedded)
        
        # Concatenar as saídas das duas direções
        hidden = torch.cat((hidden[0,:,:], hidden[1,:,:]), dim=1)
        
        # Fully connected
        output = self.dropout(F.relu(self.fc1(hidden)))
        output = self.fc2(output)
        
        return output
    
    def training_step(self, batch, batch_idx):
        """
        Passo de treinamento.
        
        Args:
            batch: Batch de dados
            batch_idx: Índice do batch
            
        Returns:
            loss: Perda calculada para o batch
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        targets = batch['targets']
        
        outputs = self(input_ids, attention_mask)
        loss = F.cross_entropy(outputs, targets)
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Passo de validação.
        
        Args:
            batch: Batch de dados
            batch_idx: Índice do batch
            
        Returns:
            dict: Dicionário com métricas de validação
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        targets = batch['targets']
        
        outputs = self(input_ids, attention_mask)
        loss = F.cross_entropy(outputs, targets)
        
        # Calcular métricas
        _, preds = torch.max(outputs, dim=1)
        
        # Converter para numpy para cálculo das métricas
        preds_np = preds.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        # Registrar métricas
        val_acc = accuracy_score(targets_np, preds_np)
        val_prec = precision_score(targets_np, preds_np, average='binary')
        val_rec = recall_score(targets_np, preds_np, average='binary')
        val_f1 = f1_score(targets_np, preds_np, average='binary')
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', val_acc, prog_bar=True)
        self.log('val_prec', val_prec, prog_bar=True)
        self.log('val_rec', val_rec, prog_bar=True)
        self.log('val_f1', val_f1, prog_bar=True)
        
        # Armazenar para análise posterior
        self.val_accuracy.append(val_acc)
        self.val_precision.append(val_prec)
        self.val_recall.append(val_rec)
        self.val_f1.append(val_f1)
        
        return {'val_loss': loss, 'val_acc': val_acc}
    
    def configure_optimizers(self):
        """
        Configura o otimizador para o treinamento.
        
        Returns:
            optimizer: Otimizador configurado
        """
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate) 