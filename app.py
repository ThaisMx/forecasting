import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sheets_connector import GoogleSheetsConnector
from timegpt_forecaster import TimeGPTForecaster
import logging
import datetime
import time

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuração da página
st.set_page_config(
    page_title="Previsão de Métricas",
    page_icon="📊",
    layout="wide"
)

# Título e descrição
st.title("📊 Sistema de Previsão de Métricas")
st.markdown("""
Este sistema utiliza TimeGPT para prever:
- Vendas totais
- Investimentos em tráfego pago
- Custo por lead (CPL)

Recursos disponíveis:
- Previsões com diferentes horizontes
- Validação cruzada
- Detecção de anomalias
- Fine-tuning com funções de perda específicas
- Suporte a variáveis exógenas
- Visualizações interativas
""")

# Inicialização dos conectores
@st.cache_resource
def init_connectors():
    return GoogleSheetsConnector(), TimeGPTForecaster()

sheets_connector, forecaster = init_connectors()

# Sidebar para configurações
st.sidebar.header("⚙️ Configurações")

# Configurações gerais
st.sidebar.subheader("Configurações de Previsão")
horizon = st.sidebar.slider("Horizonte de previsão", 7, 90, 30)
freq = st.sidebar.selectbox("Frequência", ["D", "W", "M"], 
                           format_func=lambda x: "Diária" if x == "D" else "Semanal" if x == "W" else "Mensal")
model = st.sidebar.selectbox("Modelo", ["timegpt-1", "timegpt-1-long-horizon"], 
                            help="timegpt-1-long-horizon é melhor para previsões de longo prazo")

# Opções avançadas
st.sidebar.subheader("Opções Avançadas")
show_advanced = st.sidebar.checkbox("Mostrar opções avançadas", False)

if show_advanced:
    # Intervalos de previsão
    include_intervals = st.sidebar.checkbox("Incluir intervalos de previsão", True)
    interval_levels = st.sidebar.multiselect("Níveis de intervalo", [50, 60, 70, 80, 90, 95, 99], default=[80, 90])
    
    # Controle de valor mínimo
    min_value_factor = st.sidebar.slider(
        "Fator de valor mínimo", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5, 
        step=0.05,
        help="Garante que os valores previstos não sejam menores que esta porcentagem da média histórica"
    )
    
    # Fine-tuning
    use_fine_tuning = st.sidebar.checkbox("Utilizar fine-tuning", False)
    if use_fine_tuning:
        loss_function = st.sidebar.selectbox("Função de perda", ["mse", "mae", "rmse", "mape", "smape"],
                                            help="Escolha a função de perda para otimizar o modelo")
    
    # Cross-validation
    use_cross_validation = st.sidebar.checkbox("Utilizar validação cruzada", False)
    if use_cross_validation:
        n_windows = st.sidebar.slider("Número de janelas", 2, 10, 5)
    
    # Detecção de anomalias
    use_anomaly_detection = st.sidebar.checkbox("Detectar anomalias", False)
    if use_anomaly_detection:
        sensitivity = st.sidebar.slider("Sensibilidade", 0.1, 1.0, 0.5, 0.1)
    
    # Opções de cache
    if st.sidebar.button("Limpar Cache"):
        sheets_connector.clear_cache()
        st.success("✅ Cache limpo com sucesso!")

# Cache para dados carregados, para evitar chamadas repetitivas à API
@st.cache_data(ttl=300)  # TTL de 5 minutos
def load_sheet_data(sheet_name):
    try:
        return sheets_connector.get_sheet_data(sheet_name)
    except Exception as e:
        logger.exception(f"Erro ao carregar dados: {str(e)}")
        return None

def process_tab(tab_name, sheet_name):
    with st.spinner(f"Carregando dados de {sheet_name}..."):
        try:
            # Carregar dados com cache
            df = load_sheet_data(sheet_name)
            if df is None:
                st.error(f"❌ Erro ao carregar dados da aba {sheet_name}")
                
                # Opção para tentar novamente após erro
                if st.button(f"Tentar novamente - {sheet_name}", key=f"retry_{sheet_name}"):
                    st.experimental_rerun()
                    
                return
            
            # Log dos dados carregados
            logger.info(f"Dados carregados para {sheet_name}:")
            logger.info(f"Número de linhas: {len(df)}")
            logger.info(f"Colunas: {df.columns.tolist()}")
            
            if len(df) > 0:
                logger.info(f"Primeiros valores: {df['y'].head().tolist()}")
                logger.info(f"Últimos valores: {df['y'].tail().tolist()}")
            else:
                logger.warning(f"Nenhum dado encontrado na aba {sheet_name}")
                st.warning(f"⚠️ Nenhum dado encontrado na aba {sheet_name}")
                return
            
            # Verificar se há dados suficientes
            if len(df) < 10:
                st.warning("⚠️ Poucos dados disponíveis para previsão confiável (mínimo recomendado: 10 pontos)")
            
            # Mostrar dados históricos
            st.subheader("📈 Dados Históricos")
            
            # Usar plotly para visualização interativa dos dados históricos
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines+markers', name='Histórico'))
            fig.update_layout(title=f'Dados Históricos - {tab_name}', 
                              xaxis_title='Data',
                              yaxis_title='Valor')
            st.plotly_chart(fig, use_container_width=True)
            
            # Opção para ver dados em formato tabular
            if st.checkbox(f"Ver tabela de dados para {tab_name}", False):
                st.dataframe(df)
            
            # Variáveis exógenas (se disponíveis)
            exog_cols = [col for col in df.columns if col not in ['ds', 'y']]
            has_exog = len(exog_cols) > 0
            
            if has_exog:
                st.info(f"🔍 Variáveis exógenas detectadas: {', '.join(exog_cols)}")
                use_exog = st.checkbox("Utilizar variáveis exógenas na previsão", True)
            else:
                use_exog = False
                if show_advanced:
                    st.info("ℹ️ Nenhuma variável exógena detectada. Você pode adicionar colunas extras na planilha para melhorar a previsão.")
            
            # Detecção de anomalias (se ativada)
            if show_advanced and use_anomaly_detection:
                st.subheader("🔍 Detecção de Anomalias")
                
                # Cache para detecção de anomalias
                @st.cache_data(ttl=300)
                def detect_anomalies(df_json, sensitivity):
                    df_local = pd.read_json(df_json)
                    return forecaster.detect_anomalies(df_local, sensitivity=sensitivity)
                
                with st.spinner("Detectando anomalias..."):
                    try:
                        # Converter DataFrame para JSON para usar com cache
                        df_json = df.to_json()
                        anomalies_df = detect_anomalies(df_json, sensitivity)
                        
                        if anomalies_df is not None:
                            # Filtrar apenas anomalias
                            anomalies = anomalies_df[anomalies_df['anomaly'] == True]
                            
                            if len(anomalies) > 0:
                                st.warning(f"⚠️ {len(anomalies)} anomalias detectadas!")
                                
                                # Visualização das anomalias
                                fig = go.Figure()
                                # Dados normais
                                fig.add_trace(go.Scatter(
                                    x=df['ds'], 
                                    y=df['y'], 
                                    mode='lines', 
                                    name='Dados'
                                ))
                                # Anomalias
                                fig.add_trace(go.Scatter(
                                    x=anomalies['ds'], 
                                    y=anomalies['y'], 
                                    mode='markers', 
                                    marker=dict(color='red', size=10),
                                    name='Anomalias'
                                ))
                                fig.update_layout(title='Detecção de Anomalias')
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Opção para exibir tabela de anomalias
                                if st.checkbox("Ver detalhes das anomalias", False):
                                    st.dataframe(anomalies)
                            else:
                                st.success("✅ Nenhuma anomalia detectada com o nível de sensibilidade atual.")
                        else:
                            st.error("❌ Erro ao detectar anomalias")
                    except Exception as e:
                        st.error(f"❌ Erro na detecção de anomalias: {str(e)}")
                        logger.exception("Erro detalhado:")
            
            # Cross-validation (se ativada)
            if show_advanced and use_cross_validation:
                st.subheader("🔄 Validação Cruzada")
                
                # Cache para validação cruzada
                @st.cache_data(ttl=300)
                def run_cross_validation(df_json, h, n_windows, freq, level, model, exog_json=None):
                    df_local = pd.read_json(df_json)
                    X_df = pd.read_json(exog_json) if exog_json else None
                    return forecaster.cross_validation(
                        df=df_local,
                        h=h,
                        n_windows=n_windows,
                        freq=freq,
                        level=level,
                        model=model,
                        X_df=X_df
                    )
                
                with st.spinner("Realizando validação cruzada..."):
                    try:
                        # Preparar parâmetros
                        X_df = df[exog_cols] if use_exog and has_exog else None
                        
                        # Converter DataFrames para JSON para usar com cache
                        df_json = df.to_json()
                        exog_json = X_df.to_json() if X_df is not None else None
                        
                        cv_results = run_cross_validation(
                            df_json,
                            horizon,
                            n_windows,
                            freq,
                            interval_levels if include_intervals else None,
                            model,
                            exog_json
                        )
                        
                        if cv_results is not None:
                            st.success("✅ Validação cruzada concluída com sucesso!")
                            
                            # Calcular métricas de erro
                            if 'y' in cv_results.columns and 'TimeGPT' in cv_results.columns:
                                cv_results['error'] = cv_results['y'] - cv_results['TimeGPT']
                                cv_results['abs_error'] = abs(cv_results['error'])
                                cv_results['squared_error'] = cv_results['error'] ** 2
                                
                                # Calcular métricas agregadas
                                mae = cv_results['abs_error'].mean()
                                mse = cv_results['squared_error'].mean()
                                rmse = mse ** 0.5
                                
                                # Exibir métricas
                                col1, col2, col3 = st.columns(3)
                                col1.metric("MAE", f"{mae:.2f}")
                                col2.metric("MSE", f"{mse:.2f}")
                                col3.metric("RMSE", f"{rmse:.2f}")
                            
                            # Visualização dos resultados
                            if 'cutoff' in cv_results.columns:
                                cutoffs = cv_results['cutoff'].unique()
                                
                                for cutoff in cutoffs:
                                    cutoff_str = pd.to_datetime(cutoff).strftime('%Y-%m-%d')
                                    st.subheader(f"Janela - Corte em: {cutoff_str}")
                                    
                                    window_results = cv_results[cv_results['cutoff'] == cutoff]
                                    
                                    fig = go.Figure()
                                    # Valores reais
                                    fig.add_trace(go.Scatter(
                                        x=window_results['ds'], 
                                        y=window_results['y'], 
                                        mode='lines+markers', 
                                        name='Valor Real'
                                    ))
                                    # Valores previstos
                                    fig.add_trace(go.Scatter(
                                        x=window_results['ds'], 
                                        y=window_results['TimeGPT'], 
                                        mode='lines', 
                                        name='TimeGPT'
                                    ))
                                    fig.update_layout(title=f'Resultados da Validação Cruzada - Corte: {cutoff_str}')
                                    st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("❌ Erro ao realizar validação cruzada")
                    except Exception as e:
                        st.error(f"❌ Erro na validação cruzada: {str(e)}")
                        logger.exception("Erro detalhado:")
            
            # Botão para gerar previsão
            forecast_button = st.button(f"Gerar Previsão para {tab_name}", key=f"btn_{sheet_name}")
            
            if forecast_button:
                with st.spinner("Gerando previsão..."):
                    try:
                        # Criar uma barra de progresso
                        progress_bar = st.progress(0)
                        
                        # Fine-tuning (se ativado)
                        model_id = None
                        if show_advanced and use_fine_tuning:
                            with st.spinner("Realizando fine-tuning..."):
                                try:
                                    model_id = forecaster.fine_tune(
                                        df=df,
                                        h=horizon,
                                        freq=freq,
                                        loss_function=loss_function,
                                        X_df=df[exog_cols] if use_exog and has_exog else None
                                    )
                                    if model_id:
                                        st.success(f"✅ Fine-tuning concluído com sucesso! Model ID: {model_id}")
                                        progress_bar.progress(33)
                                    else:
                                        st.error("❌ Erro ao realizar fine-tuning")
                                except Exception as e:
                                    st.error(f"❌ Erro durante fine-tuning: {str(e)}")
                                    logger.exception("Erro detalhado:")
                        else:
                            progress_bar.progress(33)
                        
                        # Gerar previsão
                        with st.spinner("Gerando previsão..."):
                            try:
                                forecast_df = forecaster.get_forecast(
                                    df=df.copy(),
                                    h=horizon,
                                    freq=freq,
                                    level=interval_levels if show_advanced and include_intervals else [80, 90],
                                    model=model_id if model_id else model,
                                    X_df=df[exog_cols] if use_exog and has_exog else None,
                                    min_value_factor=min_value_factor if show_advanced else 0.5
                                )
                                
                                progress_bar.progress(67)
                                
                                if forecast_df is not None:
                                    st.success("✅ Previsão gerada com sucesso!")
                                    
                                    # Exibir resumo da previsão
                                    if 'resumo' in forecast_df.attrs:
                                        st.markdown(forecast_df.attrs['resumo'])
                                    
                                    # Criar visualização com plotly
                                    if show_advanced:
                                        # Usar o método de plotagem da classe TimeGPTForecaster
                                        fig = forecaster.plot_forecast(
                                            historical_df=df,
                                            forecast_df=forecast_df,
                                            include_intervals=include_intervals
                                        )
                                    else:
                                        # Criar gráfico simples
                                        fig = go.Figure()
                                        # Dados históricos
                                        fig.add_trace(go.Scatter(
                                            x=df['ds'], 
                                            y=df['y'], 
                                            mode='lines', 
                                            name='Histórico'
                                        ))
                                        # Previsão
                                        fig.add_trace(go.Scatter(
                                            x=forecast_df['ds'], 
                                            y=forecast_df['y_pred'], 
                                            mode='lines', 
                                            name='Previsão',
                                            line=dict(color='red')
                                        ))
                                        fig.update_layout(
                                            title='Previsão com TimeGPT',
                                            xaxis_title='Data',
                                            yaxis_title='Valor'
                                        )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Exibir dados detalhados da previsão
                                    if st.checkbox("Ver dados da previsão", False):
                                        st.dataframe(forecast_df)
                                    
                                    # Salvar previsão
                                    if st.checkbox("Salvar previsão no Google Sheets", True):
                                        try:
                                            if sheets_connector.write_forecast(sheet_name, forecast_df):
                                                st.success("✅ Previsão salva com sucesso!")
                                            else:
                                                st.warning("⚠️ Não foi possível salvar a previsão no Google Sheets")
                                        except Exception as e:
                                            st.error(f"❌ Erro ao salvar previsão: {str(e)}")
                                            logger.exception("Erro detalhado:")
                                    
                                    # Download dos dados
                                    csv = forecast_df.to_csv(index=False)
                                    st.download_button(
                                        label="📥 Download da previsão (CSV)",
                                        data=csv,
                                        file_name=f'previsao_{sheet_name}_{datetime.datetime.now().strftime("%Y%m%d")}.csv',
                                        mime='text/csv',
                                    )
                                    
                                    progress_bar.progress(100)
                                else:
                                    st.error("❌ Erro ao gerar previsão. Verifique os logs para mais detalhes.")
                            except Exception as e:
                                st.error(f"❌ Erro ao gerar previsão: {str(e)}")
                                logger.exception("Erro detalhado:")
                    except Exception as e:
                        st.error(f"❌ Erro ao processar previsão: {str(e)}")
                        logger.exception("Erro detalhado:")
        except Exception as e:
            st.error(f"❌ Erro ao processar aba {sheet_name}: {str(e)}")
            logger.exception("Erro detalhado:")

# Tabs para diferentes métricas
tab_vendas, tab_trafego, tab_cpl = st.tabs(["Vendas Totais", "Investimento em Tráfego", "CPL"])

with tab_vendas:
    st.header("📈 Previsão de Vendas Totais")
    process_tab("Vendas Totais", "Vendas")

with tab_trafego:
    st.header("💰 Previsão de Investimento em Tráfego")
    process_tab("Investimento em Tráfego", "Trafego")

with tab_cpl:
    st.header("🎯 Previsão de CPL")
    process_tab("CPL", "CPL")

# Seção de ajuda
with st.expander("❓ Ajuda e Informações"):
    st.markdown("""
    ### Como usar este sistema
    
    1. **Configurações Básicas**:
        - Ajuste o horizonte de previsão na barra lateral
        - Selecione a frequência dos dados (diária, semanal ou mensal)
        - Escolha o modelo apropriado para seu caso
    
    2. **Opções Avançadas**:
        - Ative "Mostrar opções avançadas" para recursos adicionais
        - Defina intervalos de previsão para quantificar a incerteza
        - Use fine-tuning para adaptar o modelo aos seus dados
        - Aplique validação cruzada para avaliar a qualidade das previsões
        - Detecte anomalias nos dados históricos
        - Limpe o cache se necessário para atualizar os dados
    
    3. **Interpretação dos Resultados**:
        - O sistema mostra tendências e variações percentuais
        - Os intervalos de previsão indicam o nível de confiança
        - As anomalias detectadas podem ajudar a identificar outliers
        
    4. **Recursos do TimeGPT**:
        - Previsões precisas sem necessidade de configuração complexa
        - Captura automaticamente sazonalidades e tendências
        - Oferece suporte a variáveis exógenas para melhorar precisão
        - Adapta-se a diferentes domínios de dados
    
    5. **Solução de Problemas**:
        - Se ocorrer um erro "Quota excedida", aguarde alguns minutos e tente novamente
        - Use o botão "Limpar Cache" se os dados parecerem desatualizados
        - Em caso de erros persistentes, verifique a conexão com a internet
    """)

# Footer
st.markdown("---")
st.markdown("Desenvolvido por Thais Maximiana | [GitHub](https://github.com/ThaisMx/)") 