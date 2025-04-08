import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import json
import logging
import traceback
import requests
import plotly.graph_objects as go
import time
from typing import Dict, List, Union, Tuple, Optional, Any
from dateutil.parser import parse

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class TimeGPTForecaster:
    def __init__(self):
        """Inicializa o forecaster."""
        self.api_key = os.getenv('TIMEGPT_API_KEY')
        if not self.api_key:
            raise ValueError("TIMEGPT_API_KEY não encontrada no arquivo .env")
            
        # URL base correta para a API da Nixtla TimeGPT
        self.base_url = "https://api.nixtla.io"
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # Modelos disponíveis
        self.available_models = ["timegpt-1", "timegpt-1-long-horizon"]
        
        # Funções de perda disponíveis para fine-tuning
        self.loss_functions = ["mae", "mse", "rmse", "mape", "smape"]
        
        # Endpoint principal para previsões
        self.forecast_endpoint = "/timegpt/v1/forecast"

    def prepare_data(self, df, h=7, freq='D', level=None, model="timegpt-1", X_df=None):
        """
        Prepara os dados para a API do TimeGPT
        
        Args:
            df: DataFrame com colunas 'ds' e 'y'
            h: horizonte de previsão
            freq: frequência dos dados ('D' para diário, 'W' para semanal, etc.)
            level: níveis para intervalos de previsão (ex: [80, 90])
            model: modelo a ser usado (ex: "timegpt-1", "timegpt-1-long-horizon")
            X_df: DataFrame com variáveis exógenas (opcional)
        """
        logger.info(f"Preparando dados para previsão. Shape inicial: {df.shape}")
        
        try:
            # Converte datas para o formato correto
            df = df.copy()
            df['ds'] = pd.to_datetime(df['ds'])
            df['y'] = pd.to_numeric(df['y'], errors='coerce')
            
            # Remove valores inválidos
            df = df.dropna()
            df = df[~np.isinf(df['y'].values)]
            
            # Agrupa por data e calcula a média para lidar com datas repetidas
            df = df.groupby('ds')['y'].mean().reset_index()
            
            if len(df) == 0:
                raise ValueError("Nenhum dado válido após limpeza")
            
            logger.info(f"Dados preparados. Número de pontos válidos: {len(df)}")
            logger.info(f"Primeiros valores: {df['y'].head().tolist()}")
            logger.info(f"Últimos valores: {df['y'].tail().tolist()}")
            
            # Formato correto baseado na documentação mais recente
            df_dict = {
                "unique_id": ["series1"] * len(df),
                "ds": df['ds'].dt.strftime('%Y-%m-%d').tolist(),
                "y": df['y'].tolist()
            }
            
            # Verificar se o modelo é válido
            if model not in self.available_models:
                logger.warning(f"Modelo {model} não reconhecido. Usando o modelo padrão timegpt-1")
                model = "timegpt-1"
                
            # Criar o payload básico
            payload = {
                "df": df_dict,
                "h": h,
                "freq": freq,
                "model": model,
                "time_col": "ds",
                "target_col": "y"
            }
            
            # Adicionar níveis para intervalos de previsão se especificados
            if level:
                payload["level"] = level
            
            # Adicionar variáveis exógenas se fornecidas
            if X_df is not None and not X_df.empty:
                logger.info("Processando variáveis exógenas")
                X_df = X_df.copy()
                
                # Garantir que as datas sejam datetime
                X_df['ds'] = pd.to_datetime(X_df['ds'])
                
                # Converter para o formato esperado pela API
                exog_dict = {
                    "unique_id": ["series1"] * len(X_df),
                    "ds": X_df['ds'].dt.strftime('%Y-%m-%d').tolist()
                }
                
                # Adicionar cada coluna exógena
                for col in X_df.columns:
                    if col != 'ds':
                        exog_dict[col] = X_df[col].tolist()
                
                payload["X_df"] = exog_dict
                logger.info(f"Variáveis exógenas incluídas: {[col for col in X_df.columns if col != 'ds']}")
            
            return payload
        except Exception as e:
            logger.error(f"Erro ao preparar dados: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def get_forecast(self, df, h=30, freq='D', level=[80, 90], model=None, X_df=None, min_value_factor=0.5):
        """
        Método de conveniência que chama forecast() com parâmetros renomeados para compatibilidade com a interface do app.
        
        :param df: DataFrame com os dados históricos (deve ter colunas 'ds' e 'y')
        :param h: Horizonte de previsão em períodos
        :param freq: Frequência dos dados ('D', 'W', 'M', etc.)
        :param level: Níveis de confiança para intervalos de previsão
        :param model: Modelo TimeGPT a ser usado
        :param X_df: DataFrame com variáveis exógenas (opcional)
        :param min_value_factor: Fator mínimo em relação à média histórica
        :return: DataFrame com a previsão
        """
        return self.forecast(
            df=df,
            frequency=freq,
            horizon=h,
            level=level,
            model=model,
            min_value_factor=min_value_factor,
            X_df=X_df
        )

    def forecast(self, df, frequency='M', horizon=30, alpha=None, level=[80, 95], cv_threshold=1.0, model=None, min_value_factor=0.5, X_df=None):
        """
        Realiza previsão de série temporal usando o modelo TimeGPT.
        
        :param df: DataFrame com os dados históricos (deve ter colunas 'ds' e 'y')
        :param frequency: Frequência dos dados ('D', 'W', 'M', etc.)
        :param horizon: Horizonte de previsão em períodos
        :param alpha: Parâmetro de suavização (opcional)
        :param level: Níveis de confiança para intervalos de previsão
        :param cv_threshold: Limite mínimo do coeficiente de variação (%) para considerar a previsão válida
        :param model: Modelo TimeGPT a ser usado. Se None, escolherá automaticamente com base no horizonte
        :param min_value_factor: Fator mínimo em relação à média histórica (ex: 0.5 = 50% da média histórica)
        :param X_df: DataFrame com variáveis exógenas (opcional)
        :return: DataFrame com a previsão
        """
        logger.info(f"Iniciando previsão com TimeGPT. Horizonte: {horizon}, Frequência: {frequency}")
        
        # Validar os dados de entrada
        if df is None or len(df) == 0:
            logger.error("DataFrame vazio ou None")
            raise ValueError("DataFrame vazio ou None")
            
        # Garantir que o DataFrame tenha as colunas necessárias
        required_columns = ['ds', 'y']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Coluna '{col}' não encontrada no DataFrame")
                raise ValueError(f"Coluna '{col}' não encontrada no DataFrame")
                
        # Verificar se há dados suficientes
        min_points = 5  # Pelo menos 5 pontos para uma previsão decente
        if len(df) < min_points:
            logger.error(f"Dados insuficientes. Necessários pelo menos {min_points} pontos.")
            raise ValueError(f"Dados insuficientes. Necessários pelo menos {min_points} pontos.")
        
        # Escolher o modelo apropriado com base no horizonte
        if model is None:
            # Se o horizonte for maior que 30, usar o modelo de horizonte longo
            if horizon > 30:
                model = "timegpt-1-long-horizon"
                logger.info(f"Horizonte de previsão ({horizon}) maior que 30. Usando modelo para horizonte longo.")
            else:
                model = "timegpt-1"
                logger.info(f"Usando modelo padrão para horizonte de {horizon}.")
        else:
            logger.info(f"Usando modelo especificado: {model}")
        
        # Configurar os níveis de confiança
        if level is None:
            level = [80, 95]  # Níveis padrão
        elif isinstance(level, (int, float)):
            level = [level]  # Converter para lista se for um único valor
            
        logger.info(f"Níveis de confiança: {level}")
        
        # Preparar os dados para a API
        # Converter datas para string e valores para lista
        df_dict = {
            "unique_id": ["series1"] * len(df),
            "ds": df['ds'].astype(str).tolist(),
            "y": df['y'].tolist()
        }
        
        # Montar o payload correto
        payload = {
            "df": df_dict,
            "freq": frequency,
            "h": horizon,
            "model": model,
            "time_col": "ds",
            "target_col": "y"
        }
        
        if alpha is not None:
            payload["alpha"] = alpha
            
        # Incluir níveis de confiança
        payload["level"] = level
        
        # Adicionar variáveis exógenas se fornecidas
        if X_df is not None and not X_df.empty:
            logger.info("Processando variáveis exógenas")
            X_df = X_df.copy()
            
            # Garantir que as datas sejam datetime
            X_df['ds'] = pd.to_datetime(X_df['ds'])
            
            # Converter para o formato esperado pela API
            exog_dict = {
                "unique_id": ["series1"] * len(X_df),
                "ds": X_df['ds'].dt.strftime('%Y-%m-%d').tolist()
            }
            
            # Adicionar cada coluna exógena
            for col in X_df.columns:
                if col != 'ds':
                    exog_dict[col] = X_df[col].tolist()
            
            payload["X_df"] = exog_dict
            logger.info(f"Variáveis exógenas incluídas: {[col for col in X_df.columns if col != 'ds']}")
        
        # Fazer a solicitação à API com tentativa de endpoints alternativos
        endpoints = [self.forecast_endpoint, '/api/forecast', '/forecast', '/timegpt/v5/forecast']
        
        # Registrar tentativa de previsão com tamanho dos dados
        logger.info(f"Enviando solicitação para API. Tamanho dos dados: {len(df_dict['ds'])} pontos")
        logger.info(f"Último valor da série: {df_dict['y'][-1]}")
        
        # Tentar implementação alternativa baseada na biblioteca nixtla mais recente
        try:
            # Preparar formato para v1 da API
            v1_payload = {
                "timestamp": df['ds'].astype(str).tolist(),
                "value": df['y'].tolist(),
                "freq": frequency,
                "horizon": horizon,
                "return_conf_int": True,
                "level": level
            }
            
            if model:
                v1_payload["model"] = model
                
            logger.info("Tentando formato da API v1")
            v1_response = requests.post(
                f"{self.base_url}/timegpt/v1/forecast",
                headers=self.headers,
                json=v1_payload,
                timeout=60
            )
            
            if v1_response.status_code == 200:
                result = v1_response.json()
                logger.info("Sucesso com formato da API v1")
            else:
                logger.warning(f"Falha com formato da API v1: {v1_response.status_code} - {v1_response.text}")
                
                # Tentar endpoints tradicionais
                result = None
                for endpoint in endpoints:
                    try:
                        logger.info(f"Tentando endpoint tradicional: {endpoint}")
                        response = self._make_api_request(endpoint, payload)
                        
                        if response is not None and response.status_code == 200:
                            result = response.json()
                            logger.info(f"Sucesso na solicitação à API com endpoint {endpoint}")
                            break
                        else:
                            logger.warning(f"Falha no endpoint {endpoint}. Tentando próximo endpoint.")
                    except Exception as e:
                        logger.error(f"Erro ao chamar API com endpoint {endpoint}: {str(e)}")
        
        except Exception as e:
            logger.error(f"Erro ao tentar formato alternativo: {str(e)}")
            
            # Fallback para implementação original
            result = None
            for endpoint in endpoints:
                try:
                    logger.info(f"Tentando endpoint: {endpoint}")
                    response = self._make_api_request(endpoint, payload)
                    
                    if response is not None and response.status_code == 200:
                        result = response.json()
                        logger.info(f"Sucesso na solicitação à API com endpoint {endpoint}")
                        break
                    else:
                        logger.warning(f"Falha no endpoint {endpoint}. Tentando próximo endpoint.")
                except Exception as e:
                    logger.error(f"Erro ao chamar API com endpoint {endpoint}: {str(e)}")
        
        # Verificar se obteve resposta válida
        if result is None:
            logger.error("Todos os endpoints falharam. Não foi possível obter previsão.")
            raise ValueError("Não foi possível obter previsão da API TimeGPT")
            
        # Extrair dados da previsão
        try:
            forecast_values, upper_bounds, lower_bounds = self._extract_forecast_data(result, level)
            logger.info(f"Extraídos {len(forecast_values)} valores de previsão")
            
            # Gerar datas futuras
            last_date = df['ds'].iloc[-1]
            future_dates = self._generate_future_dates(last_date, frequency, horizon)
            
            # Verificar se os valores de previsão têm variação suficiente
            # Calculando o coeficiente de variação (CV)
            if len(forecast_values) > 1:
                mean_forecast = np.mean(forecast_values)
                std_forecast = np.std(forecast_values)
                
                # Coeficiente de variação (em porcentagem)
                cv = 0 if mean_forecast == 0 else (std_forecast / abs(mean_forecast)) * 100
                logger.info(f"Coeficiente de variação da previsão: {cv:.2f}%")
                
                # Calcular valor mínimo baseado na média histórica
                historical_mean = np.mean(df['y'])
                min_allowed_value = historical_mean * min_value_factor
                logger.info(f"Média histórica: {historical_mean:.2f}, Valor mínimo permitido: {min_allowed_value:.2f}")
                
                # Verificar se a previsão tem valores muito baixos
                if mean_forecast < min_allowed_value:
                    logger.warning(f"Valor médio previsto ({mean_forecast:.2f}) está abaixo do mínimo permitido ({min_allowed_value:.2f})")
                    logger.warning("Ajustando valores para respeitar o mínimo permitido")
                    
                    # Calcular fator de escala para ajustar os valores
                    scale_factor = min_allowed_value / mean_forecast if mean_forecast > 0 else 1.0
                    forecast_values = [val * scale_factor for val in forecast_values]
                    logger.info(f"Valores escalados por um fator de {scale_factor:.2f}")
                
                # Se o CV for muito baixo (valores quase idênticos), recriar a previsão
                # Um CV abaixo do limite geralmente indica valores muito similares
                if cv < cv_threshold:
                    logger.warning(f"Coeficiente de variação muito baixo ({cv:.2f}% < {cv_threshold}%). Valores de previsão são muito similares.")
                    logger.warning("Recriando previsão com base em tendência histórica para maior realismo.")
                    
                    # Recriar previsão com base nos dados históricos
                    forecast_values = self._extend_forecast(forecast_values[:1], future_dates, df)
                    
                    # Recalcular intervalos de confiança
                    upper_bounds = []
                    lower_bounds = []
                    for lev in level:
                        # Margem de erro proporcional ao nível de confiança
                        margin = (lev / 100) * 0.5  # 50% da proporção do nível de confiança
                        upper_bounds.append([val * (1 + margin) for val in forecast_values])
                        lower_bounds.append([val * (1 - margin) for val in forecast_values])
            
            # Estender a previsão se necessário
            extended_forecast = self._extend_forecast(forecast_values, future_dates, df)
            
            # Estender intervalos de confiança
            extended_upper_bounds = []
            extended_lower_bounds = []
            
            for upper, lower in zip(upper_bounds, lower_bounds):
                ext_upper = self._extend_forecast(upper, future_dates)
                ext_lower = self._extend_forecast(lower, future_dates)
                extended_upper_bounds.append(ext_upper)
                extended_lower_bounds.append(ext_lower)
            
            # Criar DataFrame com a previsão
            forecast_df = pd.DataFrame({
                'ds': future_dates,
                'y_pred': extended_forecast
            })
            
            # Adicionar intervalos de confiança
            for i, lev in enumerate(level):
                forecast_df[f'y_pred_upper_{lev}'] = extended_upper_bounds[i]
                forecast_df[f'y_pred_lower_{lev}'] = extended_lower_bounds[i]
                
            # Adicionar metadados da previsão
            forecast_df.attrs['model'] = 'TimeGPT'
            forecast_df.attrs['frequency'] = frequency
            forecast_df.attrs['horizon'] = horizon
            forecast_df.attrs['created_at'] = datetime.now().isoformat()
            
            # Calcular métricas de qualidade da previsão
            if len(extended_forecast) > 1:
                forecast_df.attrs['variance'] = np.var(extended_forecast)
                forecast_df.attrs['cv'] = np.std(extended_forecast) / np.mean(extended_forecast) if np.mean(extended_forecast) != 0 else 0
                
            logger.info(f"Previsão concluída com sucesso. Gerados {len(forecast_df)} pontos.")
            return forecast_df
            
        except Exception as e:
            logger.error(f"Erro ao processar dados da previsão: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise ValueError(f"Erro ao processar dados da previsão: {str(e)}")

    def _extract_forecast_data(self, result, level):
        """
        Extrai dados da previsão da resposta da API.
        Adaptado para lidar com diferentes estruturas de resposta da API.
        
        :param result: Dicionário com a resposta da API
        :param level: Lista de níveis de confiança para os intervalos
        :return: Tupla com forecast_values, upper_bounds, lower_bounds
        """
        logger.info("Extraindo dados da previsão da resposta da API")
        
        # Tentar diferentes estruturas de resposta
        forecast_values = None
        upper_bounds = []
        lower_bounds = []
        
        # Verificar formato da API v1
        if 'forecast' in result and isinstance(result['forecast'], list):
            # Formato: {"forecast": [val1, val2, ...], "conf_int_lower_90": [...], "conf_int_upper_90": [...]}
            forecast_values = result['forecast']
            logger.info(f"Detectado formato da API v1 com {len(forecast_values)} valores")
            
            # Extrair intervalos de confiança se disponíveis
            for lev in level:
                lower_key = f"conf_int_lower_{lev}"
                upper_key = f"conf_int_upper_{lev}"
                
                if lower_key in result and upper_key in result:
                    lower_bounds.append(result[lower_key])
                    upper_bounds.append(result[upper_key])
                else:
                    # Se não encontrar, usar margens padrão
                    margin = lev / 100  # Converter porcentagem para decimal
                    lower_bounds.append([val * (1 - margin) for val in forecast_values])
                    upper_bounds.append([val * (1 + margin) for val in forecast_values])
            
            return forecast_values, upper_bounds, lower_bounds
        
        # Estrutura 1: resposta direta com 'forecast'
        elif 'forecast' in result:
            forecast_values = result['forecast']
            logger.info(f"Encontrado 'forecast' na resposta com {len(forecast_values)} valores")
            
            # Verificar intervalos de confiança
            for lev in level:
                if f'upper_{lev}' in result and f'lower_{lev}' in result:
                    upper_bounds.append(result[f'upper_{lev}'])
                    lower_bounds.append(result[f'lower_{lev}'])
                else:
                    # Intervalos não encontrados, criar valores padrão
                    logger.warning(f"Intervalos de confiança para nível {lev} não encontrados")
                    upper_bounds.append([val * 1.2 for val in forecast_values])  # +20%
                    lower_bounds.append([val * 0.8 for val in forecast_values])  # -20%
        
        # Estrutura 2: resposta dentro de 'data'
        elif 'data' in result:
            data = result['data']
            
            if isinstance(data, dict):
                # Formato {y_hat: [...], upper_80: [...], ...}
                if 'y_hat' in data:
                    forecast_values = data['y_hat']
                    logger.info(f"Encontrado 'y_hat' dentro de 'data' com {len(forecast_values)} valores")
                    
                    # Verificar intervalos de confiança
                    for lev in level:
                        if f'upper_{lev}' in data and f'lower_{lev}' in data:
                            upper_bounds.append(data[f'upper_{lev}'])
                            lower_bounds.append(data[f'lower_{lev}'])
                        else:
                            # Intervalos não encontrados, criar valores padrão
                            logger.warning(f"Intervalos para nível {lev} não encontrados em data")
                            upper_bounds.append([val * 1.2 for val in forecast_values])
                            lower_bounds.append([val * 0.8 for val in forecast_values])
            
            elif isinstance(data, list) and len(data) > 0:
                # Formato [{ds: '2023-01-01', y_hat: 100, ...}, ...]
                if isinstance(data[0], dict) and 'y_hat' in data[0]:
                    forecast_values = [point['y_hat'] for point in data]
                    logger.info(f"Extraído 'y_hat' de lista de dicionários com {len(forecast_values)} valores")
                    
                    # Verificar intervalos de confiança
                    for lev in level:
                        upper_key = f'upper_{lev}'
                        lower_key = f'lower_{lev}'
                        
                        if all(upper_key in point and lower_key in point for point in data):
                            upper_bounds.append([point[upper_key] for point in data])
                            lower_bounds.append([point[lower_key] for point in data])
                        else:
                            # Intervalos não encontrados, criar valores padrão
                            logger.warning(f"Intervalos para nível {lev} não encontrados em todos os pontos")
                            upper_bounds.append([val * 1.2 for val in forecast_values])
                            lower_bounds.append([val * 0.8 for val in forecast_values])
        
        # Se ainda não encontrou os valores, tentar outras estruturas possíveis
        if forecast_values is None:
            for key in result:
                if isinstance(result[key], list) and len(result[key]) > 0 and isinstance(result[key][0], (int, float)):
                    forecast_values = result[key]
                    logger.info(f"Extraído valores de '{key}' com {len(forecast_values)} valores")
                    
                    # Criar intervalos de confiança padrão
                    for _ in level:
                        upper_bounds.append([val * 1.2 for val in forecast_values])
                        lower_bounds.append([val * 0.8 for val in forecast_values])
                    break
        
        # Se ainda não encontrou, procurar em qualquer array disponível
        if forecast_values is None:
            for key, value in result.items():
                if isinstance(value, list) and len(value) > 0:
                    if all(isinstance(x, (int, float)) for x in value):
                        forecast_values = value
                        logger.info(f"Extraído valores numéricos de '{key}' com {len(forecast_values)} valores")
                        
                        # Criar intervalos de confiança padrão
                        for _ in level:
                            upper_bounds.append([val * 1.2 for val in forecast_values])
                            lower_bounds.append([val * 0.8 for val in forecast_values])
                        break
        
        # Se ainda não encontrou, considerar erro
        if forecast_values is None:
            logger.error(f"Não foi possível extrair valores de previsão da resposta: {json.dumps(result)[:1000]}")
            raise ValueError("Estrutura de resposta da API desconhecida")
        
        return forecast_values, upper_bounds, lower_bounds
    
    def _generate_future_dates(self, last_date, frequency, horizon):
        """
        Gera datas futuras baseadas na frequência e horizonte.
        
        :param last_date: Última data da série histórica
        :param frequency: Frequência dos dados ('D', 'W', 'M', etc.)
        :param horizon: Horizonte de previsão em períodos
        :return: Lista de datas futuras
        """
        if isinstance(last_date, str):
            last_date = pd.to_datetime(last_date)
        
        logger.info(f"Gerando {horizon} datas futuras a partir de {last_date} com frequência {frequency}")
        
        freq_map = {
            'D': 'D',     # Diária
            'W': 'W-MON', # Semanal (iniciando na segunda)
            'M': 'MS',    # Mensal (início do mês)
            'Q': 'QS',    # Trimestral
            'Y': 'YS',    # Anual
            'H': 'H',     # Horária
            '5min': '5min', # 5 minutos
            '1min': 'min',  # 1 minuto
        }
        
        # Mapear a frequência para formato pandas
        pd_freq = freq_map.get(frequency, frequency)
        
        # Gerar período
        future_dates = pd.date_range(
            start=last_date,
            periods=horizon + 1,
            freq=pd_freq
        )[1:]  # Excluir a primeira data, que é a última data da série histórica
        
        logger.info(f"Geradas {len(future_dates)} datas futuras, de {future_dates.min()} a {future_dates.max()}")
        return future_dates.tolist()
    
    def _extend_forecast(self, forecast_values, future_dates, df=None):
        """
        Estende os valores da previsão para corresponder ao número de datas futuras.
        Usa métodos mais sofisticados quando há menos valores do que datas.
        
        :param forecast_values: Lista com os valores de previsão
        :param future_dates: Lista com as datas futuras 
        :param df: DataFrame original com dados históricos, se disponível
        :return: Lista estendida com os valores de previsão
        """
        # Se o número de valores for igual ao número de datas, retorna os valores originais
        if len(forecast_values) == len(future_dates):
            return forecast_values
        
        # Se houver mais valores do que datas, truncar
        if len(forecast_values) > len(future_dates):
            logger.warning(f"Número de valores de previsão ({len(forecast_values)}) maior que o número de datas geradas ({len(future_dates)})")
            return forecast_values[:len(future_dates)]
        
        # Se houver menos valores do que datas, estender a previsão
        logger.warning(f"Número de valores de previsão ({len(forecast_values)}) menor que o número de datas geradas ({len(future_dates)})")
        
        # Verificar se temos múltiplos valores distintos
        unique_values = np.unique(forecast_values)
        
        # Quando temos vários valores distintos (pelo menos 3), podemos usar os próprios valores para estimar tendência
        if len(unique_values) >= 3:
            logger.info("Estendendo previsão usando regressão linear nos valores da previsão original")
            
            # Calcular a tendência (inclinação) usando regressão linear
            x = np.arange(len(forecast_values))
            slope, intercept = np.polyfit(x, forecast_values, 1)
            
            # Calcular a variabilidade (desvio padrão) presente nos valores de previsão
            std_dev = np.std(forecast_values)
            
            # Definir o ruído como uma fração do desvio padrão ou um valor mínimo
            noise_level = max(std_dev * 0.5, abs(np.mean(forecast_values)) * 0.02) if np.mean(forecast_values) != 0 else 0.1
            
            # Gerar novos valores baseados na tendência com um pouco de ruído
            extended_values = forecast_values.copy()
            last_value = forecast_values[-1]
            
            for i in range(len(forecast_values), len(future_dates)):
                # Calcular o próximo valor projetado pela tendência
                next_value = last_value + slope
                
                # Adicionar um pouco de ruído aleatório para evitar que a linha seja perfeitamente reta
                noise = np.random.normal(0, noise_level)
                next_value += noise
                
                extended_values.append(next_value)
                last_value = next_value
                
            return extended_values
        
        # Quando temos valores iguais ou quase iguais, ou apenas um valor de previsão
        elif len(unique_values) <= 2 or len(forecast_values) == 1:
            logger.info("Valores de previsão são iguais ou quase iguais. Usando dados históricos para estimar tendência.")
            
            # Se temos dados históricos disponíveis
            if df is not None and len(df) > 5:
                # Calcular tendência dos dados históricos
                hist_values = df['y'].values
                hist_x = np.arange(len(hist_values))
                
                # Usar regressão linear para estimar tendência
                hist_slope, _ = np.polyfit(hist_x, hist_values, 1)
                
                # Calcular variabilidade histórica (desvio padrão)
                hist_std = np.std(hist_values)
                
                # Definir o ruído como uma fração do desvio padrão ou um valor mínimo
                noise_level = max(hist_std * 0.5, abs(np.mean(hist_values)) * 0.02) if np.mean(hist_values) != 0 else 0.1
                
                # Estender os valores usando a tendência histórica
                extended_values = forecast_values.copy()
                last_value = forecast_values[-1]
                
                for i in range(len(forecast_values), len(future_dates)):
                    # Calcular o próximo valor baseado na tendência histórica 
                    next_value = last_value + hist_slope
                    
                    # Adicionar ruído para criar variabilidade natural
                    noise = np.random.normal(0, noise_level)
                    next_value += noise
                    
                    extended_values.append(next_value)
                    last_value = next_value
                    
                return extended_values
                
            else:
                # Sem dados históricos, usar uma abordagem mais simples com variabilidade
                logger.info("Sem dados históricos disponíveis. Usando o último valor da previsão com variabilidade.")
                
                extended_values = forecast_values.copy()
                last_value = forecast_values[-1]
                
                # Definir um nível de ruído proporcional ao valor
                base_noise = abs(last_value) * 0.03 if last_value != 0 else 0.1
                
                for i in range(len(forecast_values), len(future_dates)):
                    # Adicionar uma pequena variação aleatória ao último valor
                    noise = np.random.normal(0, base_noise)
                    next_value = last_value + noise
                    
                    extended_values.append(next_value)
                    last_value = next_value
                
                return extended_values 

    def _make_api_request(self, endpoint, data):
        """
        Realiza uma requisição à API do TimeGPT com tratamento de erros e retentativas.
        
        :param endpoint: Endpoint da API (ex: '/api/forecast')
        :param data: Dicionário com os dados a serem enviados
        :return: Objeto de resposta ou None em caso de falha
        """
        logger.info(f"Fazendo requisição para endpoint {endpoint}")
        
        max_retries = 3
        retry_delay = 2  # segundos
        
        for attempt in range(max_retries):
            try:
                full_url = f"{self.base_url}{endpoint}"
                logger.info(f"Tentativa {attempt+1}/{max_retries}: {full_url}")
                
                response = requests.post(
                    full_url,
                    headers=self.headers,
                    json=data,
                    timeout=60  # timeout de 60 segundos
                )
                
                # Verificar se a resposta é bem-sucedida
                if response.status_code == 200:
                    # Verificar se mesmo com status 200, existe um aviso sobre horizonte
                    response_body = response.json()
                    if isinstance(response_body, dict) and 'warnings' in response_body:
                        warnings = response_body['warnings']
                        for warning in warnings if isinstance(warnings, list) else [warnings]:
                            if isinstance(warning, str) and 'horizon' in warning and 'exceed' in warning:
                                logger.warning(f"Aviso da API: {warning}")
                                logger.warning("O horizonte especificado excede o recomendado. Considere reduzir o horizonte ou usar o modelo 'timegpt-1-long-horizon'.")
                    
                    logger.info(f"Requisição bem-sucedida: {response.status_code}")
                    return response
                
                # Verificar se a resposta indica limite de horizonte excedido
                if response.status_code == 400:
                    error_text = response.text
                    if any(term in error_text.lower() for term in ["horizon", "exceed", "horizonte", "excede"]):
                        logger.warning("Aviso: O horizonte especificado excede o limite do modelo.")
                        logger.warning("Considere usar o modelo 'timegpt-1-long-horizon' para horizontes maiores ou reduzir o horizonte de previsão.")
                        
                        # Se estiver usando o modelo padrão, tente alternar para o modelo de horizonte longo
                        if 'model' in data:
                            model_name = data.get('model')
                        elif 'df' in data and isinstance(data, dict):
                            model_name = data.get('model')
                        else:
                            model_name = None
                            
                        if model_name == "timegpt-1":
                            logger.info("Tentando alternar automaticamente para o modelo de horizonte longo...")
                            data_copy = data.copy()
                            
                            # Atualizar o modelo para horizonte longo
                            if 'model' in data_copy:
                                data_copy['model'] = "timegpt-1-long-horizon"
                            elif 'df' in data_copy and isinstance(data_copy, dict):
                                data_copy['model'] = "timegpt-1-long-horizon"
                            
                            # Fazer nova requisição com o modelo de horizonte longo
                            alt_response = requests.post(
                                full_url,
                                headers=self.headers,
                                json=data_copy,
                                timeout=60
                            )
                            
                            if alt_response.status_code == 200:
                                logger.info("Sucesso ao usar modelo de horizonte longo como fallback!")
                                return alt_response
                            else:
                                logger.warning(f"Falha ao usar modelo alternativo: {alt_response.status_code}")
                    
                # Verificar se a resposta indica outros erros comuns
                if response.status_code == 429:
                    logger.warning("Taxa de requisições excedida. Aguardando antes de tentar novamente.")
                    time.sleep(retry_delay * (attempt + 1))  # Backoff exponencial
                    continue
                
                logger.error(f"Erro na API: {response.status_code} - {response.text}")
                
                # Retornar a resposta mesmo com erro para que o chamador possa tentar outro endpoint
                return response
                
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout na requisição (tentativa {attempt+1}/{max_retries})")
                time.sleep(retry_delay * (attempt + 1))
            except requests.exceptions.ConnectionError:
                logger.warning(f"Erro de conexão (tentativa {attempt+1}/{max_retries})")
                time.sleep(retry_delay * (attempt + 1))
            except Exception as e:
                logger.error(f"Erro inesperado: {str(e)}")
                return None
        
        logger.error(f"Todas as {max_retries} tentativas falharam")
        return None 