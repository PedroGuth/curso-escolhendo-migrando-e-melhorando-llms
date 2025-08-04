#!/bin/bash

# Script para executar o Playground Interativo no macOS
# Resolve problemas de permissões automaticamente

echo "🍎 Playground Interativo - macOS"
echo "================================"

# Verificar se o ambiente virtual existe
if [ ! -d "playground_env" ]; then
    echo "⚠️ Ambiente virtual não encontrado!"
    echo "🔄 Execute primeiro: python setup_playground_macos.py"
    exit 1
fi

# Ativar ambiente virtual
echo "🔄 Ativando ambiente virtual..."
source playground_env/bin/activate

# Verificar se as dependências estão instaladas
echo "🔍 Verificando dependências..."
python -c "import streamlit, pandas, numpy, plotly" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Dependências não encontradas!"
    echo "🔄 Execute: python setup_playground_macos.py"
    exit 1
fi

echo "✅ Dependências OK!"
echo "🚀 Iniciando playground..."

# Executar o playground
streamlit run playground_interativo.py

# Desativar ambiente virtual ao sair
deactivate 