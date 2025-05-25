#!/usr/bin/env python
"""
Script para executar a análise de sentimentos usando o módulo src.
"""
import sys
from src.main import main

if __name__ == "__main__":
    # Passamos todos os argumentos da linha de comando para o script principal
    train_mode = "--train" in sys.argv
    main(train_new_model=train_mode) 