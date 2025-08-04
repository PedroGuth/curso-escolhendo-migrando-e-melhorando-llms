#!/usr/bin/env python3
"""
Script de Setup para o Playground Interativo
Resolve automaticamente problemas de compatibilidade do NumPy
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Executa um comando e mostra o progresso"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Concluído!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Erro:")
        print(f"   {e.stderr}")
        return False

def check_python_version():
    """Verifica se a versão do Python é compatível"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ é necessário!")
        print(f"   Versão atual: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK!")
    return True

def install_dependencies():
    """Instala as dependências de forma segura"""
    
    print("🚀 Iniciando instalação do Playground Interativo...")
    print("=" * 50)
    
    # Verificar versão do Python
    if not check_python_version():
        return False
    
    # Atualizar pip
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Atualizando pip"):
        return False
    
    # Desinstalar versões conflitantes
    print("🧹 Limpando versões conflitantes...")
    run_command(f"{sys.executable} -m pip uninstall -y matplotlib seaborn scipy", "Removendo bibliotecas conflitantes")
    
    # Instalar NumPy primeiro (versão específica)
    if not run_command(f"{sys.executable} -m pip install 'numpy>=1.24.0,<2.0.0'", "Instalando NumPy"):
        return False
    
    # Instalar pandas
    if not run_command(f"{sys.executable} -m pip install 'pandas>=1.5.0'", "Instalando Pandas"):
        return False
    
    # Instalar plotly
    if not run_command(f"{sys.executable} -m pip install 'plotly>=5.15.0'", "Instalando Plotly"):
        return False
    
    # Instalar streamlit
    if not run_command(f"{sys.executable} -m pip install 'streamlit>=1.28.0'", "Instalando Streamlit"):
        return False
    
    # Instalar dependências opcionais
    run_command(f"{sys.executable} -m pip install 'openpyxl>=3.0.0'", "Instalando OpenPyXL")
    run_command(f"{sys.executable} -m pip install 'xlrd>=2.0.0'", "Instalando XLRD")
    
    return True

def test_installation():
    """Testa se a instalação foi bem-sucedida"""
    print("\n🧪 Testando instalação...")
    
    test_code = """
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Teste básico
df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
fig = px.scatter(df, x='x', y='y')
print("✅ Todas as bibliotecas importadas com sucesso!")
print("✅ Playground pronto para uso!")
"""
    
    try:
        result = subprocess.run([sys.executable, "-c", test_code], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Erro no teste:")
        print(e.stderr)
        return False

def show_next_steps():
    """Mostra os próximos passos"""
    print("\n🎉 Instalação Concluída!")
    print("=" * 50)
    print("📋 Próximos Passos:")
    print("1. Execute o playground:")
    print("   streamlit run playground_interativo.py")
    print("")
    print("2. O playground abrirá automaticamente no navegador")
    print("   (geralmente em http://localhost:8501)")
    print("")
    print("3. Navegue pelas seções usando o menu lateral")
    print("")
    print("4. Experimente o simulador interativo!")
    print("")
    print("💡 Dica: Se encontrar problemas, tente:")
    print("   - Usar um ambiente virtual")
    print("   - Verificar se a porta 8501 está livre")
    print("   - Usar navegadores modernos (Chrome, Firefox, Safari)")

def main():
    """Função principal"""
    print("🎮 Setup do Playground Interativo - Migração de Modelos GenAI")
    print("=" * 60)
    
    # Instalar dependências
    if not install_dependencies():
        print("\n❌ Falha na instalação!")
        print("💡 Tente:")
        print("   1. Usar um ambiente virtual")
        print("   2. Atualizar o Python para versão 3.8+")
        print("   3. Verificar permissões de instalação")
        return
    
    # Testar instalação
    if not test_installation():
        print("\n❌ Falha no teste!")
        return
    
    # Mostrar próximos passos
    show_next_steps()

if __name__ == "__main__":
    main() 