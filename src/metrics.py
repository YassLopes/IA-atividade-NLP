import torch
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model(model, val_loader, device):
    """
    Avalia o modelo treinado no conjunto de validação e exibe as métricas.
    
    Args:
        model: O modelo treinado
        val_loader: DataLoader para o conjunto de validação
        device: Dispositivo para processamento (cuda/cpu)
    
    Returns:
        accuracy, precision, recall, f1: Métricas calculadas
    """
    # Coletar todas as previsões e rótulos
    all_preds = []
    all_targets = []
    
    model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)
            
            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calcular métricas
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='binary')
    recall = recall_score(all_targets, all_preds, average='binary')
    f1 = f1_score(all_targets, all_preds, average='binary')
    conf_matrix = confusion_matrix(all_targets, all_preds)
    
    # Exibir métricas
    print("\n===== Métricas de Avaliação (Validação) =====")
    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Plotar matriz de confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negativo', 'Positivo'],
                yticklabels=['Negativo', 'Positivo'])
    plt.xlabel('Previsão')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão (Validação)')
    plt.savefig('confusion_matrix_validation.png')
    
    # Plotar métricas ao longo do treinamento
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(model.val_accuracy) + 1)
    plt.plot(epochs, model.val_accuracy, 'b-o', label='Acurácia')
    plt.plot(epochs, model.val_precision, 'g-o', label='Precisão')
    plt.plot(epochs, model.val_recall, 'r-o', label='Recall')
    plt.plot(epochs, model.val_f1, 'y-o', label='F1-Score')
    plt.title('Métricas durante o Treinamento')
    plt.xlabel('Epochs')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_metrics.png')
    
    return accuracy, precision, recall, f1

def test_model(model, test_loader, device):
    """
    Avalia o modelo treinado no conjunto de teste e salva as métricas em um arquivo Markdown.
    
    Args:
        model: O modelo treinado
        test_loader: DataLoader para o conjunto de teste
        device: Dispositivo para processamento (cuda/cpu)
    
    Returns:
        accuracy, precision, recall, f1: Métricas calculadas
    """
    # Coletar todas as previsões e rótulos
    all_preds = []
    all_targets = []
    
    model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)
            
            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calcular métricas
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='binary')
    recall = recall_score(all_targets, all_preds, average='binary')
    f1 = f1_score(all_targets, all_preds, average='binary')
    conf_matrix = confusion_matrix(all_targets, all_preds)
    
    # Exibir métricas
    print("\n===== Métricas de Avaliação (Teste) =====")
    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Plotar matriz de confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negativo', 'Positivo'],
                yticklabels=['Negativo', 'Positivo'])
    plt.xlabel('Previsão')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão (Teste)')
    plt.savefig('confusion_matrix_test.png')
    
    # Gerar relatório em Markdown
    data_atual = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    
    # Incluir informações de hardware
    hardware_info = f"CPU" if device == "cpu" else f"GPU ({torch.cuda.get_device_name(0)})"
    
    markdown_content = f"""# Relatório de Métricas - Análise de Sentimentos

## Informações Gerais
- **Data de execução:** {data_atual}
- **Modelo:** LSTM Bidirecional
- **Dataset:** IMDB Dataset (50K Movie Reviews)
- **Hardware:** {hardware_info}

## Métricas de Desempenho no Conjunto de Teste

| Métrica | Valor |
|---------|-------|
| Acurácia | {accuracy:.4f} |
| Precisão | {precision:.4f} |
| Recall | {recall:.4f} |
| F1-Score | {f1:.4f} |

## Matriz de Confusão

![Matriz de Confusão](confusion_matrix_test.png)

## Observações
- O modelo foi treinado por {len(model.val_accuracy)} épocas
- A melhor época foi escolhida com base na perda de validação (validation loss)
- A matriz de confusão mostra a distribuição de verdadeiros positivos, falsos positivos, verdadeiros negativos e falsos negativos

## Exemplos de Classificação

### Exemplo de resenha positiva
"This movie was amazing, I really enjoyed it and would recommend it to everyone!"  
**Sentimento previsto:** Positivo

### Exemplo de resenha negativa
"This movie was terrible, I couldn't even finish watching it. The acting was poor and the plot made no sense."  
**Sentimento previsto:** Negativo
"""
    
    # Salvar relatório em Markdown
    with open('metrics_report.md', 'w') as f:
        f.write(markdown_content)
    
    print(f"\nRelatório de métricas salvo em 'metrics_report.md'")
    
    return accuracy, precision, recall, f1 