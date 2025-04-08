import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
import logging
import time
from functools import lru_cache

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class GoogleSheetsConnector:
    def __init__(self):
        """Inicializa o conector com as credenciais do Google Sheets."""
        try:
            self.scope = ['https://spreadsheets.google.com/feeds',
                     'https://www.googleapis.com/auth/drive']
            self.credentials = ServiceAccountCredentials.from_json_keyfile_name(
                'credentials.json', self.scope)
            self.client = gspread.authorize(self.credentials)
            self.sheet_id = os.getenv('GOOGLE_SHEET_ID')
            self.sheet = self.client.open_by_key(self.sheet_id)
            
            # Controle de taxa de requisição
            self.last_request_time = 0
            self.min_request_interval = 1.0  # Intervalo mínimo de 1 segundo entre requisições
            
            # Cache para armazenar dados de planilhas
            self.sheet_cache = {}
            self.cache_ttl = 300  # Time-to-live do cache em segundos (5 minutos)
            self.cache_last_updated = {}
            
            logger.info("GoogleSheetsConnector inicializado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar GoogleSheetsConnector: {str(e)}")
            raise

    def _rate_limit(self):
        """Implementa controle de taxa para requisições ao Google Sheets."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.min_request_interval:
            wait_time = self.min_request_interval - elapsed
            logger.info(f"Aplicando rate limiting: aguardando {wait_time:.2f} segundos")
            time.sleep(wait_time)
        
        self.last_request_time = time.time()

    def get_sheet_data(self, sheet_name):
        """
        Obtém dados de uma aba específica, com cache e controle de taxa.
        
        Args:
            sheet_name: Nome da aba
            
        Returns:
            DataFrame com colunas 'ds' e 'y'
        """
        # Verificar se os dados estão no cache e se são válidos
        current_time = time.time()
        if (sheet_name in self.sheet_cache and 
            sheet_name in self.cache_last_updated and 
            current_time - self.cache_last_updated[sheet_name] < self.cache_ttl):
            logger.info(f"Usando dados em cache para a aba: {sheet_name}")
            return self.sheet_cache[sheet_name].copy()
        
        try:
            logger.info(f"Obtendo dados da aba: {sheet_name}")
            
            # Aplicar rate limiting antes da requisição
            self._rate_limit()
            
            try:
                worksheet = self.sheet.worksheet(sheet_name)
                data = worksheet.get_all_records()
            except gspread.exceptions.APIError as e:
                if "RESOURCE_EXHAUSTED" in str(e):
                    logger.warning(f"Quota excedida. Aguardando 60 segundos antes de tentar novamente: {str(e)}")
                    time.sleep(60)  # Esperar 60 segundos antes de tentar novamente
                    
                    # Tentar novamente após espera
                    worksheet = self.sheet.worksheet(sheet_name)
                    data = worksheet.get_all_records()
                else:
                    raise
                
            df = pd.DataFrame(data)
            
            # Verificar se o DataFrame não está vazio
            if df.empty:
                logger.warning(f"Nenhum dado encontrado na aba {sheet_name}")
                return None
            
            # Verificar se as colunas necessárias existem
            if 'ds' not in df.columns or 'y' not in df.columns:
                logger.error(f"Colunas obrigatórias 'ds' e 'y' não encontradas na aba {sheet_name}")
                return None
            
            # Converter coluna de data
            df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
            
            # Converter coluna de valores para numérico
            df['y'] = pd.to_numeric(df['y'], errors='coerce')
            
            # Remover linhas com valores inválidos
            df = df.dropna(subset=['ds', 'y'])
            
            # Armazenar no cache
            self.sheet_cache[sheet_name] = df.copy()
            self.cache_last_updated[sheet_name] = current_time
            
            logger.info(f"Dados carregados com sucesso: {len(df)} linhas")
            logger.debug(f"Primeiros valores: {df.head().to_dict()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Erro ao obter dados da aba {sheet_name}: {str(e)}")
            return None

    def write_forecast(self, sheet_name, forecast_df):
        """
        Escreve os resultados da previsão em uma nova aba.
        
        Args:
            sheet_name: Nome da aba original
            forecast_df: DataFrame com a previsão
        """
        try:
            new_sheet_name = f"Previsão_{sheet_name}"
            
            # Aplicar rate limiting antes da requisição
            self._rate_limit()
            
            # Verificar se a aba já existe
            try:
                worksheet = self.sheet.worksheet(new_sheet_name)
            except gspread.exceptions.WorksheetNotFound:
                worksheet = self.sheet.add_worksheet(new_sheet_name, rows=1000, cols=20)
            except gspread.exceptions.APIError as e:
                if "RESOURCE_EXHAUSTED" in str(e):
                    logger.warning(f"Quota excedida ao criar aba. Aguardando 60 segundos: {str(e)}")
                    time.sleep(60)  # Esperar 60 segundos antes de tentar novamente
                    
                    # Tentar novamente após espera
                    try:
                        worksheet = self.sheet.worksheet(new_sheet_name)
                    except gspread.exceptions.WorksheetNotFound:
                        worksheet = self.sheet.add_worksheet(new_sheet_name, rows=1000, cols=20)
                else:
                    raise
            
            # Preparar dados para escrita
            forecast_data = forecast_df.copy()
            forecast_data['ds'] = forecast_data['ds'].dt.strftime('%Y-%m-%d')
            
            # Limpar aba existente
            self._rate_limit()
            worksheet.clear()
            
            # Preparar os dados em lotes para reduzir o número de requisições
            all_data = [list(forecast_df.columns)]  # Cabeçalho
            
            for _, row in forecast_data.iterrows():
                all_data.append(row.tolist())
            
            # Enviar todos os dados de uma vez
            self._rate_limit()
            worksheet.update(all_data)
            
            logger.info(f"Previsão escrita com sucesso na aba {new_sheet_name}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao escrever previsão: {str(e)}")
            return False
            
    def clear_cache(self):
        """Limpa o cache de dados."""
        self.sheet_cache = {}
        self.cache_last_updated = {}
        logger.info("Cache de dados limpo") 