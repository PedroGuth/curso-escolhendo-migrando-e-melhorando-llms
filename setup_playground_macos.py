#!/usr/bin/env python3
"""
Script de Setup para macOS - Playground Interativo
Resolve problemas de permissões e compatibilidade no macOS
"""

import subprocess
import sys
import os
import platform

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

def check_system():
    """Verifica se é macOS e detecta problemas de permissão"""
    if platform.system() != "Darwin":
        print("⚠️ Este script é específico para macOS!")
        return False
    
    print(f"🍎 macOS detectado: {platform.mac_ver()[0]}")
    
    # Verificar se Python está no diretório do sistema
    python_path = sys.executable
    if "/Library/Frameworks/" in python_path:
        print("⚠️ Python instalado no diretório do sistema (requer sudo)")
        return True
    else:
        print("✅ Python instalado em localização segura")
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

def install_homebrew():
    """Instala Homebrew se não estiver disponível"""
    try:
        result = subprocess.run(["brew", "--version"], capture_output=True, text=True)
        print("✅ Homebrew já instalado")
        return True
    except FileNotFoundError:
        print("🔄 Instalando Homebrew...")
        install_command = '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
        print("⚠️ Execute o comando abaixo no terminal:")
        print(f"   {install_command}")
        print("⚠️ Depois execute novamente este script")
        return False

def create_virtual_environment():
    """Cria ambiente virtual para evitar problemas de permissão"""
    venv_name = "playground_env"
    
    if os.path.exists(venv_name):
        print(f"✅ Ambiente virtual '{venv_name}' já existe")
        return venv_name
    
    print(f"🔄 Criando ambiente virtual '{venv_name}'...")
    if run_command(f"{sys.executable} -m venv {venv_name}", "Criando ambiente virtual"):
        return venv_name
    return None

def install_in_virtual_environment(venv_name):
    """Instala dependências no ambiente virtual"""
    # Determinar o comando de ativação baseado no shell
    shell = os.environ.get('SHELL', '')
    if 'zsh' in shell:
        activate_cmd = f"source {venv_name}/bin/activate"
    elif 'bash' in shell:
        activate_cmd = f"source {venv_name}/bin/activate"
    else:
        activate_cmd = f"source {venv_name}/bin/activate"
    
    print("🔄 Instalando dependências no ambiente virtual...")
    
    # Comandos para instalar no ambiente virtual
    commands = [
        f"{venv_name}/bin/pip install --upgrade pip",
        f"{venv_name}/bin/pip install 'numpy>=1.24.0,<2.0.0'",
        f"{venv_name}/bin/pip install 'pandas>=1.5.0'",
        f"{venv_name}/bin/pip install 'plotly>=5.15.0'",
        f"{venv_name}/bin/pip install 'streamlit>=1.28.0'",
        f"{venv_name}/bin/pip install 'openpyxl>=3.0.0'",
        f"{venv_name}/bin/pip install 'xlrd>=2.0.0'"
    ]
    
    for cmd in commands:
        if not run_command(cmd, f"Executando: {cmd.split()[-1]}"):
            return False
    
    return True

def test_installation(venv_name):
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
        result = subprocess.run([f"{venv_name}/bin/python", "-c", test_code], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Erro no teste:")
        print(e.stderr)
        return False

def show_next_steps(venv_name):
    """Mostra os próximos passos"""
    print("\n🎉 Instalação Concluída!")
    print("=" * 50)
    print("📋 Próximos Passos:")
    print("")
    print("1. Ative o ambiente virtual:")
    print(f"   source {venv_name}/bin/activate")
    print("")
    print("2. Execute o playground:")
    print("   streamlit run playground_interativo.py")
    print("")
    print("3. O playground abrirá automaticamente no navegador")
    print("   (geralmente em http://localhost:8501)")
    print("")
    print("4. Navegue pelas seções usando o menu lateral")
    print("")
    print("5. Experimente o simulador interativo!")
    print("")
    print("💡 Dicas:")
    print("   - Sempre ative o ambiente virtual antes de usar")
    print("   - Para desativar: deactivate")
    print("   - Use navegadores modernos (Chrome, Firefox, Safari)")

def show_alternative_solutions():
    """Mostra soluções alternativas"""
    print("\n🔧 Soluções Alternativas:")
    print("=" * 40)
    print("")
    print("1. Usar Homebrew Python (Recomendado):")
    print("   brew install python@3.11")
    print("   brew link python@3.11")
    print("")
    print("2. Usar pyenv:")
    print("   brew install pyenv")
    print("   pyenv install 3.11.0")
    print("   pyenv global 3.11.0")
    print("")
    print("3. Usar Anaconda/Miniconda:")
    print("   brew install --cask miniconda")
    print("   conda create -n playground python=3.11")
    print("   conda activate playground")
    print("")
    print("4. Instalar com sudo (não recomendado):")
    print("   sudo pip install -r requirements_streamlit.txt")

def main():
    """Função principal"""
    print("🍎 Setup do Playground Interativo - macOS")
    print("=" * 50)
    
    # Verificar sistema
    if not check_system():
        print("❌ Este script é específico para macOS!")
        return
    
    # Verificar versão do Python
    if not check_python_version():
        return
    
    # Verificar se Python está no diretório do sistema
    system_python = "/Library/Frameworks/" in sys.executable
    
    if system_python:
        print("\n⚠️ Python detectado no diretório do sistema!")
        print("🔧 Usando ambiente virtual para evitar problemas de permissão...")
        
        # Criar ambiente virtual
        venv_name = create_virtual_environment()
        if not venv_name:
            print("❌ Falha ao criar ambiente virtual!")
            show_alternative_solutions()
            return
        
        # Instalar no ambiente virtual
        if not install_in_virtual_environment(venv_name):
            print("❌ Falha na instalação!")
            show_alternative_solutions()
            return
        
        # Testar instalação
        if not test_installation(venv_name):
            print("❌ Falha no teste!")
            return
        
        # Mostrar próximos passos
        show_next_steps(venv_name)
        
    else:
        print("\n✅ Python em localização segura!")
        print("🔧 Instalando diretamente...")
        
        # Tentar instalação direta
        if run_command(f"{sys.executable} -m pip install --upgrade pip", "Atualizando pip"):
            if run_command(f"{sys.executable} -m pip install 'numpy>=1.24.0,<2.0.0'", "Instalando NumPy"):
                if run_command(f"{sys.executable} -m pip install streamlit pandas plotly", "Instalando outras dependências"):
                    print("✅ Instalação concluída!")
                    print("\n📋 Para executar:")
                    print("   streamlit run playground_interativo.py")
                else:
                    show_alternative_solutions()
            else:
                show_alternative_solutions()
        else:
            show_alternative_solutions()

if __name__ == "__main__":
    main() 