# 🎮 Playground Interativo - Migração de Modelos GenAI

## 🎯 Sobre o Playground

Este playground interativo permite que você explore os resultados de cada notebook do workshop de forma visual e interativa, sem precisar executar código! É como ter uma versão "demo" de todos os resultados.

## 🚀 Como Usar

### 🍎 macOS (Recomendado)

Se você está no macOS e encontrou problemas de permissão:

```bash
# 1. Execute o script específico para macOS
python setup_playground_macos.py

# 2. Execute o playground (script automático)
./run_playground_macos.sh
```

**Ou manualmente:**
```bash
# 1. Ativar ambiente virtual
source playground_env/bin/activate

# 2. Executar playground
streamlit run playground_interativo.py

# 3. Desativar quando terminar
deactivate
```

### 🐧 Linux / Windows

#### 1. Instalação Automática (Recomendado)

```bash
# Execute o script de setup automático
python setup_playground.py
```

Este script resolve automaticamente problemas de compatibilidade e instala todas as dependências necessárias.

#### 2. Instalação Manual

```bash
# Instalar dependências
pip install -r requirements_streamlit.txt

# Ou instalar individualmente
pip install streamlit pandas numpy plotly
```

#### 3. Executar o Playground

```bash
# No terminal, na pasta do projeto
streamlit run playground_interativo.py
```

O playground abrirá automaticamente no seu navegador (geralmente em `http://localhost:8501`).

## 🐛 Solução de Problemas

### 🍎 macOS - Problemas de Permissão

Se você encontrar erros como:
```
ERROR: Could not install packages due to an OSError: Cannot move the non-empty directory
```

**Solução Automática (Recomendada):**
```bash
python setup_playground_macos.py
```

**Soluções Alternativas:**

1. **Usar Homebrew Python:**
```bash
brew install python@3.11
brew link python@3.11
python setup_playground.py
```

2. **Usar pyenv:**
```bash
brew install pyenv
pyenv install 3.11.0
pyenv global 3.11.0
python setup_playground.py
```

3. **Usar Anaconda/Miniconda:**
```bash
brew install --cask miniconda
conda create -n playground python=3.11
conda activate playground
pip install -r requirements_streamlit.txt
```

### Erro de Compatibilidade do NumPy

Se você encontrar o erro:
```
ValueError: numpy.dtype size changed, may indicate binary incompatibility
```

**Solução 1 - Script Automático (Recomendado):**
```bash
python setup_playground.py
```

**Solução 2 - Manual:**
```bash
# Desinstalar versões conflitantes
pip uninstall -y matplotlib seaborn scipy

# Instalar NumPy versão específica
pip install 'numpy>=1.24.0,<2.0.0'

# Reinstalar outras dependências
pip install streamlit pandas plotly
```

**Solução 3 - Ambiente Virtual:**
```bash
# Criar ambiente virtual
python -m venv playground_env

# Ativar ambiente (Windows)
playground_env\Scripts\activate

# Ativar ambiente (Mac/Linux)
source playground_env/bin/activate

# Instalar dependências
pip install -r requirements_streamlit.txt
```

### Outros Problemas Comuns

#### Erro de Dependências
```bash
# Atualizar pip
pip install --upgrade pip

# Reinstalar dependências
pip install --force-reinstall streamlit pandas plotly
```

#### Porta Ocupada
```bash
# Usar porta diferente
streamlit run playground_interativo.py --server.port 8502
```

#### Problemas de Performance
- Use navegadores modernos (Chrome, Firefox, Safari)
- Feche outras abas pesadas
- Considere usar modo headless para servidores

## 📚 Seções Disponíveis

### 🏠 Visão Geral
- Introdução ao workshop
- Objetivos e metodologia
- Métricas rápidas

### 📊 Passo 1 - Dados
- Análise do dataset
- Baseline do modelo fonte
- Distribuição de latência

### ⚡ Passo 2 - Prompts
- Comparação de prompts otimizados
- Análise de otimização
- Custos da otimização

### ⏱️ Passo 3 - Latência
- Métricas de latência por modelo
- Gráficos de comparação
- Insights de performance

### 🎯 Passo 4 - Qualidade
- Scores de qualidade (LLM-as-a-Judge)
- Gráfico de radar
- Custos da avaliação

### 📈 Passo 5 - Comparação
- Dashboard consolidado
- Comparação 3D
- Recomendações por caso de uso

### 🎮 Simulador Interativo
- **Experimente diferentes cenários!**
- Ajuste parâmetros em tempo real
- Veja como a escolha do modelo muda

## 🎯 Recursos Interativos

### Simulador de Cenários
- **Requisições por mês**: 1.000 - 100.000
- **Peso da Latência**: 0% - 100%
- **Peso do Custo**: 0% - 100%
- **Threshold de Qualidade**: 0.0 - 1.0
- **Orçamento Mensal**: $10 - $1.000

### Visualizações Dinâmicas
- Gráficos interativos com Plotly
- Gráfico de radar para qualidade
- Comparação 3D
- Tabelas detalhadas

### Insights Automáticos
- Recomendações baseadas em dados
- Análise de economia
- Comparações de performance

## 💡 Casos de Uso

### 🚀 Aplicações em Tempo Real
- Chatbots
- Assistentes virtuais
- **Recomendação**: Amazon Nova Lite (menor latência)

### 💰 Processamento em Lote
- Análise de documentos
- Relatórios automáticos
- **Recomendação**: Amazon Nova Lite (menor custo)

### 🎯 Aplicações Críticas
- Resumos médicos
- Documentos legais
- **Recomendação**: Claude 3.5 Haiku (melhor qualidade)

## 🔧 Personalização

### Dados Simulados
Os dados são baseados nos resultados reais do workshop, mas você pode modificar a função `generate_sample_data()` para testar cenários diferentes.

### Métricas
Você pode ajustar os pesos no simulador para refletir suas prioridades específicas:
- **Latência**: Para aplicações em tempo real
- **Custo**: Para orçamentos limitados
- **Qualidade**: Para aplicações críticas

## 🎓 Aprendizado

### Conceitos-Chave
1. **Migração Baseada em Dados**: Não apenas intuição
2. **Trade-offs**: Velocidade vs Custo vs Qualidade
3. **Avaliação Multidimensional**: Considerar todos os aspectos
4. **Decisões Informadas**: Baseadas em evidências

### Próximos Passos
- Aplique esta metodologia aos seus projetos
- Experimente com diferentes modelos
- Considere avaliação contínua
- Automatize o processo

## 📞 Suporte

Se encontrar problemas:
1. **macOS**: Execute `python setup_playground_macos.py` primeiro
2. **Outros**: Execute `python setup_playground.py` primeiro
3. Verifique se todas as dependências estão instaladas
4. Confirme que está usando Python 3.8+
5. Tente executar em um ambiente virtual limpo

## 🎉 Divirta-se!

Este playground foi criado para tornar o aprendizado mais interativo e divertido. Experimente diferentes cenários e veja como as decisões de modelo mudam baseadas nos seus requisitos específicos!

---

**💡 Dica**: Use o simulador interativo para testar cenários realistas do seu projeto antes de implementar a migração real! 