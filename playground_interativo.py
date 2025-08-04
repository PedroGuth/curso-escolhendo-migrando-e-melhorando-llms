import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
import time

# Configuração da página
st.set_page_config(
    page_title="🎯 Playground de Migração de Modelos GenAI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-metric {
        border-left-color: #28a745;
    }
    .warning-metric {
        border-left-color: #ffc107;
    }
    .danger-metric {
        border-left-color: #dc3545;
    }
    .model-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Dados simulados baseados nos resultados reais
def generate_sample_data():
    """Gera dados simulados baseados nos resultados reais do workshop"""
    
    # Dados de latência
    latency_data = {
        'model': ['source_model', 'amazon.nova-lite-v1:0', 'us.anthropic.claude-3-5-haiku-20241022-v1:0'],
        'latency_mean': [1.2, 0.49, 0.85],
        'latency_p50': [1.15, 0.485, 0.82],
        'latency_p90': [1.45, 0.640, 1.1],
        'latency_std': [0.3, 0.15, 0.25],
        'avg_cost': [0.0025, 0.000078, 0.0012],
        'avg_input_tokens': [850, 780, 820],
        'avg_output_tokens': [120, 95, 110],
        'sample_size': [73, 10, 10]
    }
    
    # Dados de qualidade
    quality_data = {
        'model': ['source_model', 'amazon.nova-lite-v1:0', 'us.anthropic.claude-3-5-haiku-20241022-v1:0'],
        'Builtin.Correctness': [0.95, 1.00, 1.00],
        'Builtin.Completeness': [0.925, 0.900, 1.000],
        'Builtin.ProfessionalStyleAndTone': [1.0, 1.0, 1.0]
    }
    
    return pd.DataFrame(latency_data), pd.DataFrame(quality_data)

def main():
    # Header principal
    st.markdown('<h1 class="main-header">🎯 Playground de Migração de Modelos GenAI</h1>', unsafe_allow_html=True)
    
    # Sidebar para navegação
    st.sidebar.title("📚 Navegação")
    page = st.sidebar.selectbox(
        "Escolha uma seção:",
        ["🏠 Visão Geral", "📊 Passo 1 - Dados", "⚡ Passo 2 - Prompts", "⏱️ Passo 3 - Latência", 
         "🎯 Passo 4 - Qualidade", "📈 Passo 5 - Comparação", "🎮 Simulador Interativo"]
    )
    
    # Gerando dados simulados
    latency_df, quality_df = generate_sample_data()
    
    if page == "🏠 Visão Geral":
        show_overview()
    
    elif page == "📊 Passo 1 - Dados":
        show_step1_data(latency_df, quality_df)
    
    elif page == "⚡ Passo 2 - Prompts":
        show_step2_prompts()
    
    elif page == "⏱️ Passo 3 - Latência":
        show_step3_latency(latency_df)
    
    elif page == "🎯 Passo 4 - Qualidade":
        show_step4_quality(quality_df)
    
    elif page == "📈 Passo 5 - Comparação":
        show_step5_comparison(latency_df, quality_df)
    
    elif page == "🎮 Simulador Interativo":
        show_interactive_simulator(latency_df, quality_df)

def show_overview():
    """Mostra a visão geral do workshop"""
    
    st.markdown("## 🎯 Sobre Este Workshop")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### O Que Você Vai Aprender
        
        Este workshop te ensina a **migrar modelos de IA generativa** de forma científica e profissional. 
        É como aprender a trocar de carro, mas com dados e métricas em vez de intuição!
        
        ### 🎯 Objetivo
        Migrar de um modelo proprietário caro para alternativas mais econômicas no Amazon Bedrock, 
        mantendo ou melhorando a qualidade da sumarização de documentos.
        
        ### 📊 Metodologia
        Avaliamos modelos em **3 dimensões críticas**:
        - ⏱️ **Latência**: Velocidade de resposta
        - 💰 **Custo**: Preço por inferência
        - 🎯 **Qualidade**: Precisão dos resultados
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 Modelos Testados
        
        - **Modelo Fonte**: Proprietário (baseline)
        - **Amazon Nova Lite**: Rápido e econômico
        - **Claude 3.5 Haiku**: Bom custo-benefício
        
        ### 📈 Resultados Esperados
        
        - Relatório PDF profissional
        - Dados CSV para análise
        - Decisão baseada em evidências
        """)
    
    # Métricas rápidas
    st.markdown("## 📊 Métricas Rápidas")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Modelos Avaliados", "3", "source + 2 candidatos")
    
    with col2:
        st.metric("Amostras Testadas", "93", "total de documentos")
    
    with col3:
        st.metric("Métricas Coletadas", "12+", "latência, custo, qualidade")
    
    with col4:
        st.metric("Tempo Estimado", "2-3h", "workshop completo")

def show_step1_data(latency_df, quality_df):
    """Mostra os resultados do Passo 1 - Preparação de Dados"""
    
    st.markdown("## 📊 Passo 1: Preparação de Dados com Modelo Fonte")
    
    st.markdown("""
    ### 🎯 O Que Fizemos
    Neste passo, estabelecemos a **linha de base** para nossa migração. 
    É como marcar onde você está antes de começar uma corrida!
    """)
    
    # Dataset info
    st.markdown("### 📚 Dataset Utilizado")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **EdinburghNLP/xsum**
        - Dataset de sumarização de notícias
        - 10 amostras representativas
        - Documentos + resumos de referência
        """)
    
    with col2:
        # Simulando dados do dataset
        sample_data = {
            'document_length': [1200, 800, 1500, 950, 1100],
            'summary_length': [150, 120, 180, 140, 160]
        }
        df_sample = pd.DataFrame(sample_data)
        
        fig = px.scatter(df_sample, x='document_length', y='summary_length',
                        title='📏 Distribuição de Tamanhos',
                        labels={'document_length': 'Tamanho do Documento (chars)',
                               'summary_length': 'Tamanho do Resumo (chars)'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Baseline do modelo fonte
    st.markdown("### 🎯 Baseline do Modelo Fonte")
    
    source_data = latency_df[latency_df['model'] == 'source_model'].iloc[0]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Latência Média", f"{source_data['latency_mean']:.2f}s", 
                 "tempo de resposta")
    
    with col2:
        st.metric("Custo por Inferência", f"${source_data['avg_cost']:.4f}", 
                 "preço por resumo")
    
    with col3:
        st.metric("Qualidade (Correção)", f"{quality_df[quality_df['model'] == 'source_model']['Builtin.Correctness'].iloc[0]:.2f}", 
                 "score de 0-1")
    
    # Distribuição de latência do modelo fonte
    st.markdown("### 📊 Distribuição de Latência do Modelo Fonte")
    
    # Simulando dados de distribuição
    np.random.seed(42)
    source_latencies = np.random.normal(source_data['latency_mean'], source_data['latency_std'], 100)
    
    fig = px.histogram(x=source_latencies, nbins=20,
                      title='⏱️ Distribuição de Latência - Modelo Fonte',
                      labels={'x': 'Latência (segundos)', 'y': 'Frequência'})
    fig.add_vline(x=source_data['latency_mean'], line_dash="dash", line_color="red",
                  annotation_text=f"Média: {source_data['latency_mean']:.2f}s")
    st.plotly_chart(fig, use_container_width=True)

def show_step2_prompts():
    """Mostra os resultados do Passo 2 - Otimização de Prompts"""
    
    st.markdown("## ⚡ Passo 2: Otimização de Prompts")
    
    st.markdown("""
    ### 🎯 O Que Fizemos
    Otimizamos prompts para cada modelo usando o **Amazon Bedrock Prompt Optimizer**. 
    É como ensinar cada modelo a falar da melhor forma possível!
    """)
    
    # Prompts comparados
    st.markdown("### 📝 Prompts Comparados")
    
    prompts = {
        'Modelo Fonte': """
First, please read the article below.
{context}
Now, can you write me an extremely short abstract for it?
        """,
        'Amazon Nova Lite': """
TAREFA: Criar um resumo extremamente conciso

ARTIGO:
{context}

INSTRUÇÕES:
- Leia o artigo acima
- Crie um resumo muito curto e direto
- Mantenha apenas as informações essenciais
        """,
        'Claude 3.5 Haiku': """
Por favor, leia o seguinte artigo:

{context}

Agora, escreva um resumo extremamente breve e conciso deste artigo.
        """
    }
    
    # Mostrando prompts lado a lado
    cols = st.columns(len(prompts))
    
    for i, (model, prompt) in enumerate(prompts.items()):
        with cols[i]:
            st.markdown(f"**{model}**")
            st.code(prompt, language='text')
            st.metric("Tamanho", f"{len(prompt)} chars", 
                     f"{len(prompt.split())} palavras")
    
    # Análise de otimização
    st.markdown("### 🔍 Análise da Otimização")
    
    optimization_metrics = {
        'Modelo': ['Fonte', 'Nova Lite', 'Claude Haiku'],
        'Tamanho Original': [120, 120, 120],
        'Tamanho Otimizado': [120, 280, 180],
        'Estrutura': ['Simples', 'Estruturada', 'Natural'],
        'Eficiência': [0.7, 0.9, 0.85]
    }
    
    df_opt = pd.DataFrame(optimization_metrics)
    
    fig = px.bar(df_opt, x='Modelo', y='Tamanho Otimizado',
                 title='📏 Comparação de Tamanhos de Prompt',
                 color='Eficiência', color_continuous_scale='viridis')
    st.plotly_chart(fig, use_container_width=True)
    
    # Custo da otimização
    st.markdown("### 💰 Custo da Otimização")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Tokens Processados", "3,123", "total")
        st.metric("Custo Estimado", "$0.09", "otimização única")
    
    with col2:
        st.markdown("""
        **Detalhes do Cálculo:**
        - Taxa: $0.030 por 1.000 tokens
        - Tokens de entrada: 1.451
        - Tokens de saída: 1.672
        - Total: 3.123 tokens
        """)

def show_step3_latency(latency_df):
    """Mostra os resultados do Passo 3 - Avaliação de Latência"""
    
    st.markdown("## ⏱️ Passo 3: Avaliação de Latência")
    
    st.markdown("""
    ### 🎯 O Que Fizemos
    Medimos a velocidade de resposta de cada modelo. 
    É como cronometrar cada volta de uma corrida!
    """)
    
    # Métricas de latência
    st.markdown("### 📊 Métricas de Latência")
    
    # Criando gráfico de comparação
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Latência Média', 'Latência P90', 'Distribuição', 'Custo vs Latência'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # Gráfico 1: Latência média
    fig.add_trace(
        go.Bar(x=latency_df['model'], y=latency_df['latency_mean'],
               name='Latência Média', marker_color='lightblue'),
        row=1, col=1
    )
    
    # Gráfico 2: Latência P90
    fig.add_trace(
        go.Bar(x=latency_df['model'], y=latency_df['latency_p90'],
               name='Latência P90', marker_color='orange'),
        row=1, col=2
    )
    
    # Gráfico 3: Distribuição (simulada)
    for i, model in enumerate(latency_df['model']):
        latencies = np.random.normal(
            latency_df.iloc[i]['latency_mean'], 
            latency_df.iloc[i]['latency_std'], 
            50
        )
        fig.add_trace(
            go.Scatter(x=latencies, mode='markers', name=f'{model} (simulado)',
                      opacity=0.6),
            row=2, col=1
        )
    
    # Gráfico 4: Custo vs Latência
    fig.add_trace(
        go.Scatter(x=latency_df['latency_mean'], y=latency_df['avg_cost'],
                   mode='markers+text', text=latency_df['model'],
                   textposition="top center", name='Custo vs Latência',
                   marker=dict(size=15)),
        row=2, col=2
    )
    
    fig.update_layout(height=600, title_text="📊 Análise Completa de Latência")
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabela detalhada
    st.markdown("### 📋 Tabela Detalhada de Métricas")
    
    display_df = latency_df.copy()
    display_df['model_short'] = display_df['model'].apply(
        lambda x: x.split('.')[-1] if '.' in x else x
    )
    
    # Formatando colunas
    for col in ['latency_mean', 'latency_p50', 'latency_p90', 'latency_std', 'avg_cost']:
        if col.startswith('latency'):
            display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}s")
        else:
            display_df[col] = display_df[col].apply(lambda x: f"${x:.6f}")
    
    st.dataframe(display_df[['model_short', 'latency_mean', 'latency_p50', 'latency_p90', 'avg_cost', 'sample_size']],
                 use_container_width=True)
    
    # Insights
    st.markdown("### 💡 Insights")
    
    fastest_model = latency_df.loc[latency_df['latency_mean'].idxmin(), 'model']
    cheapest_model = latency_df.loc[latency_df['avg_cost'].idxmin(), 'model']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **🏆 Mais Rápido**: {fastest_model}
        - Latência média: {latency_df[latency_df['model'] == fastest_model]['latency_mean'].iloc[0]:.3f}s
        - Ideal para aplicações em tempo real
        """)
    
    with col2:
        st.markdown(f"""
        **💰 Mais Barato**: {cheapest_model}
        - Custo por inferência: ${latency_df[latency_df['model'] == cheapest_model]['avg_cost'].iloc[0]:.6f}
        - Ideal para processamento em lote
        """)

def show_step4_quality(quality_df):
    """Mostra os resultados do Passo 4 - Avaliação de Qualidade"""
    
    st.markdown("## 🎯 Passo 4: Avaliação de Qualidade")
    
    st.markdown("""
    ### 🎯 O Que Fizemos
    Avaliamos a qualidade das respostas usando **LLM-as-a-Judge**. 
    É como ter um juiz especializado avaliando cada resposta!
    """)
    
    # Métricas de qualidade
    st.markdown("### 📊 Métricas de Qualidade")
    
    # Criando gráfico de radar
    fig = go.Figure()
    
    metrics = ['Builtin.Correctness', 'Builtin.Completeness', 'Builtin.ProfessionalStyleAndTone']
    
    for i, model in enumerate(quality_df['model']):
        values = quality_df[quality_df['model'] == model][metrics].values[0].tolist()
        values += values[:1]  # Fechar o polígono
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],
            fill='toself',
            name=model.split('.')[-1] if '.' in model else model,
            line_color=px.colors.qualitative.Set1[i]
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="🎯 Comparação de Qualidade - Gráfico de Radar"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Gráfico de barras
    fig2 = px.bar(quality_df.melt(id_vars=['model'], value_vars=metrics),
                  x='model', y='value', color='variable',
                  title='📊 Scores de Qualidade por Métrica',
                  labels={'value': 'Score (0-1)', 'model': 'Modelo'})
    st.plotly_chart(fig2, use_container_width=True)
    
    # Tabela detalhada
    st.markdown("### 📋 Tabela Detalhada de Qualidade")
    
    display_quality = quality_df.copy()
    display_quality['model_short'] = display_quality['model'].apply(
        lambda x: x.split('.')[-1] if '.' in x else x
    )
    
    # Formatando scores
    for col in metrics:
        display_quality[col] = display_quality[col].apply(lambda x: f"{x:.3f}")
    
    st.dataframe(display_quality[['model_short'] + metrics], use_container_width=True)
    
    # Análise de custo da avaliação
    st.markdown("### 💰 Custo da Avaliação de Qualidade")
    
    evaluation_costs = {
        'Modelo': ['source_model', 'amazon.nova-lite-v1:0', 'us.anthropic.claude-3-5-haiku-20241022-v1:0'],
        'Custo por Métrica': [0.007137, 0.013481, 0.013224],
        'Total (3 métricas)': [0.021411, 0.040443, 0.039672]
    }
    
    df_costs = pd.DataFrame(evaluation_costs)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig3 = px.bar(df_costs, x='Modelo', y='Total (3 métricas)',
                      title='💰 Custo Total da Avaliação',
                      color='Total (3 métricas)', color_continuous_scale='viridis')
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        st.markdown("""
        **📊 Detalhes dos Custos:**
        
        - **Avaliador**: Amazon Nova Pro
        - **Métricas**: 3 (Correção, Completude, Estilo)
        - **Amostras**: 10 por modelo
        - **Custo Total**: ~$0.10
        
        **💡 Insight**: A avaliação representa um custo único pequeno comparado aos custos contínuos de inferência.
        """)
    
    # Insights de qualidade
    st.markdown("### 💡 Insights de Qualidade")
    
    best_correctness = quality_df.loc[quality_df['Builtin.Correctness'].idxmax(), 'model']
    best_completeness = quality_df.loc[quality_df['Builtin.Completeness'].idxmax(), 'model']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **✅ Melhor Correção**: {best_correctness}
        - Score: {quality_df[quality_df['model'] == best_correctness]['Builtin.Correctness'].iloc[0]:.3f}
        - Ideal para aplicações que precisam de precisão
        """)
    
    with col2:
        st.markdown(f"""
        **📝 Melhor Completude**: {best_completeness}
        - Score: {quality_df[quality_df['model'] == best_completeness]['Builtin.Completeness'].iloc[0]:.3f}
        - Ideal para resumos abrangentes
        """)

def show_step5_comparison(latency_df, quality_df):
    """Mostra os resultados do Passo 5 - Comparação Final"""
    
    st.markdown("## 📈 Passo 5: Comparação Final e Decisão")
    
    st.markdown("""
    ### 🎯 O Que Fizemos
    Consolidamos todas as métricas e geramos um relatório final. 
    É como reunir todos os juízes para dar o veredicto final!
    """)
    
    # Combinando dados
    combined_df = latency_df.merge(quality_df, on='model')
    
    # Dashboard principal
    st.markdown("### 📊 Dashboard de Comparação")
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        fastest = combined_df.loc[combined_df['latency_mean'].idxmin()]
        st.metric("🏆 Mais Rápido", fastest['model'].split('.')[-1], 
                 f"{fastest['latency_mean']:.3f}s")
    
    with col2:
        cheapest = combined_df.loc[combined_df['avg_cost'].idxmin()]
        st.metric("💰 Mais Barato", cheapest['model'].split('.')[-1], 
                 f"${cheapest['avg_cost']:.6f}")
    
    with col3:
        best_quality = combined_df.loc[combined_df['Builtin.Correctness'].idxmax()]
        st.metric("🎯 Melhor Qualidade", best_quality['model'].split('.')[-1], 
                 f"{best_quality['Builtin.Correctness']:.3f}")
    
    with col4:
        best_overall = combined_df.loc[(combined_df['latency_mean'] * combined_df['avg_cost']).idxmin()]
        st.metric("⚖️ Melhor Custo-Benefício", best_overall['model'].split('.')[-1], 
                 "latência × custo")
    
    # Gráfico de comparação 3D
    st.markdown("### 🌟 Comparação 3D")
    
    fig = px.scatter_3d(combined_df, 
                        x='latency_mean', 
                        y='avg_cost', 
                        z='Builtin.Correctness',
                        size='sample_size',
                        color='model',
                        title='🎯 Comparação 3D: Latência vs Custo vs Qualidade',
                        labels={'latency_mean': 'Latência (s)', 
                               'avg_cost': 'Custo ($)', 
                               'Builtin.Correctness': 'Qualidade'})
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Matriz de decisão
    st.markdown("### 🎯 Matriz de Decisão")
    
    # Calculando scores normalizados
    combined_df['latency_score'] = 1 - (combined_df['latency_mean'] - combined_df['latency_mean'].min()) / (combined_df['latency_mean'].max() - combined_df['latency_mean'].min())
    combined_df['cost_score'] = 1 - (combined_df['avg_cost'] - combined_df['avg_cost'].min()) / (combined_df['avg_cost'].max() - combined_df['avg_cost'].min())
    combined_df['quality_score'] = combined_df['Builtin.Correctness']
    
    # Score composto
    combined_df['composite_score'] = (combined_df['latency_score'] * 0.3 + 
                                     combined_df['cost_score'] * 0.4 + 
                                     combined_df['quality_score'] * 0.3)
    
    fig2 = px.bar(combined_df, x='model', y='composite_score',
                  color='composite_score',
                  title='🏆 Score Composto (30% Latência + 40% Custo + 30% Qualidade)',
                  labels={'composite_score': 'Score Composto', 'model': 'Modelo'})
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Recomendações por caso de uso
    st.markdown("### 💡 Recomendações por Caso de Uso")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🚀 Aplicações em Tempo Real**
        - Chatbots, assistentes virtuais
        - **Recomendação**: Amazon Nova Lite
        - **Razão**: Menor latência (0.49s)
        """)
    
    with col2:
        st.markdown("""
        **💰 Processamento em Lote**
        - Análise de documentos, relatórios
        - **Recomendação**: Amazon Nova Lite
        - **Razão**: Menor custo ($0.000078)
        """)
    
    with col3:
        st.markdown("""
        **🎯 Aplicações Críticas**
        - Resumos médicos, legais
        - **Recomendação**: Claude 3.5 Haiku
        - **Razão**: Melhor qualidade (1.000)
        """)
    
    # Resumo da análise
    st.markdown("### 📋 Resumo da Análise")
    
    st.markdown("""
    **🎉 Resultados Principais:**
    
    1. **Amazon Nova Lite** se destaca em velocidade e custo
    2. **Claude 3.5 Haiku** oferece a melhor qualidade
    3. **Modelo Fonte** tem boa qualidade mas custo alto
    
    **💡 Decisão Recomendada:**
    Para a maioria dos casos de uso, **Amazon Nova Lite** oferece o melhor custo-benefício,
    combinando boa performance com baixo custo.
    """)
    
    # Simulação de economia
    st.markdown("### 💰 Simulação de Economia")
    
    monthly_requests = st.slider("Requisições por mês:", 1000, 100000, 10000, 1000)
    
    source_cost = combined_df[combined_df['model'] == 'source_model']['avg_cost'].iloc[0] * monthly_requests
    nova_cost = combined_df[combined_df['model'] == 'amazon.nova-lite-v1:0']['avg_cost'].iloc[0] * monthly_requests
    savings = source_cost - nova_cost
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Custo Modelo Fonte", f"${source_cost:.2f}/mês")
    
    with col2:
        st.metric("Custo Nova Lite", f"${nova_cost:.2f}/mês")
    
    with col3:
        st.metric("Economia Mensal", f"${savings:.2f}", f"{savings/source_cost*100:.1f}%")

def show_interactive_simulator(latency_df, quality_df):
    """Mostra o simulador interativo"""
    
    st.markdown("## 🎮 Simulador Interativo")
    
    st.markdown("""
    ### 🎯 Experimente Diferentes Cenários
    
    Use este simulador para testar como diferentes parâmetros afetam a escolha do modelo!
    """)
    
    # Parâmetros do simulador
    st.markdown("### ⚙️ Configuração do Cenário")
    
    col1, col2 = st.columns(2)
    
    with col1:
        monthly_requests = st.slider("Requisições por mês:", 1000, 100000, 10000, 1000)
        latency_weight = st.slider("Peso da Latência (%):", 0, 100, 30, 5)
        quality_threshold = st.slider("Threshold de Qualidade Mínima:", 0.0, 1.0, 0.9, 0.05)
    
    with col2:
        budget_constraint = st.slider("Orçamento Mensal ($):", 10, 1000, 100, 10)
        cost_weight = st.slider("Peso do Custo (%):", 0, 100, 40, 5)
        quality_weight = 100 - latency_weight - cost_weight
    
    st.markdown(f"**Peso da Qualidade**: {quality_weight}%")
    
    # Simulação
    combined_df = latency_df.merge(quality_df, on='model')
    
    # Filtrando por threshold de qualidade
    filtered_df = combined_df[combined_df['Builtin.Correctness'] >= quality_threshold].copy()
    
    if len(filtered_df) == 0:
        st.warning("⚠️ Nenhum modelo atende ao threshold de qualidade mínimo!")
        return
    
    # Calculando custos mensais
    filtered_df['monthly_cost'] = filtered_df['avg_cost'] * monthly_requests
    
    # Filtrando por orçamento
    budget_filtered = filtered_df[filtered_df['monthly_cost'] <= budget_constraint].copy()
    
    if len(budget_filtered) == 0:
        st.warning("⚠️ Nenhum modelo cabe no orçamento!")
        return
    
    # Calculando scores
    budget_filtered['latency_score'] = 1 - (budget_filtered['latency_mean'] - budget_filtered['latency_mean'].min()) / (budget_filtered['latency_mean'].max() - budget_filtered['latency_mean'].min())
    budget_filtered['cost_score'] = 1 - (budget_filtered['monthly_cost'] - budget_filtered['monthly_cost'].min()) / (budget_filtered['monthly_cost'].max() - budget_filtered['monthly_cost'].min())
    budget_filtered['quality_score'] = budget_filtered['Builtin.Correctness']
    
    # Score composto
    budget_filtered['composite_score'] = (budget_filtered['latency_score'] * latency_weight/100 + 
                                         budget_filtered['cost_score'] * cost_weight/100 + 
                                         budget_filtered['quality_score'] * quality_weight/100)
    
    # Resultados
    st.markdown("### 📊 Resultados da Simulação")
    
    best_model = budget_filtered.loc[budget_filtered['composite_score'].idxmax()]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🏆 Modelo Recomendado", best_model['model'].split('.')[-1])
    
    with col2:
        st.metric("📊 Score Composto", f"{best_model['composite_score']:.3f}")
    
    with col3:
        st.metric("💰 Custo Mensal", f"${best_model['monthly_cost']:.2f}")
    
    # Gráfico de comparação
    fig = px.bar(budget_filtered, x='model', y='composite_score',
                 color='monthly_cost',
                 title='🏆 Comparação dos Modelos Elegíveis',
                 labels={'composite_score': 'Score Composto', 'model': 'Modelo', 'monthly_cost': 'Custo Mensal ($)'})
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabela detalhada
    st.markdown("### 📋 Análise Detalhada")
    
    display_sim = budget_filtered.copy()
    display_sim['model_short'] = display_sim['model'].apply(
        lambda x: x.split('.')[-1] if '.' in x else x
    )
    
    # Formatando colunas
    display_sim['latency_mean'] = display_sim['latency_mean'].apply(lambda x: f"{x:.3f}s")
    display_sim['monthly_cost'] = display_sim['monthly_cost'].apply(lambda x: f"${x:.2f}")
    display_sim['Builtin.Correctness'] = display_sim['Builtin.Correctness'].apply(lambda x: f"{x:.3f}")
    display_sim['composite_score'] = display_sim['composite_score'].apply(lambda x: f"{x:.3f}")
    
    st.dataframe(display_sim[['model_short', 'latency_mean', 'monthly_cost', 'Builtin.Correctness', 'composite_score']],
                 use_container_width=True)
    
    # Insights
    st.markdown("### 💡 Insights da Simulação")
    
    if best_model['model'] == 'amazon.nova-lite-v1:0':
        st.success("✅ **Amazon Nova Lite** é a melhor escolha para este cenário - oferece boa performance com baixo custo!")
    elif best_model['model'] == 'us.anthropic.claude-3-5-haiku-20241022-v1:0':
        st.info("ℹ️ **Claude 3.5 Haiku** é a melhor escolha - oferece excelente qualidade!")
    else:
        st.warning("⚠️ **Modelo Fonte** ainda é a melhor opção - considere ajustar os parâmetros!")
    
    # Economia vs modelo fonte
    if 'source_model' in budget_filtered['model'].values:
        source_cost = budget_filtered[budget_filtered['model'] == 'source_model']['monthly_cost'].iloc[0]
        savings = source_cost - best_model['monthly_cost']
        
        if savings > 0:
            st.metric("💰 Economia vs Modelo Fonte", f"${savings:.2f}/mês", f"{savings/source_cost*100:.1f}%")

if __name__ == "__main__":
    main() 