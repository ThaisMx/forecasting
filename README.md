# Sistema de Previsão de Métricas com TimeGPT

Este projeto implementa um sistema avançado de previsão de métricas utilizando o TimeGPT da Nixtla, integrado com Google Sheets e apresentado através de uma interface web moderna.

## 📊 Funcionalidades

- **Previsões de Séries Temporais:**
  - Vendas totais
  - Investimento em tráfego pago
  - Custo por Lead (CPL)
- **Recursos Avançados do TimeGPT:**
  - Cross-validation para validação de modelos
  - Fine-tuning com funções de perda personalizadas (MAE, MSE, RMSE, MAPE, SMAPE)
  - Detecção de anomalias em dados históricos
  - Suporte a variáveis exógenas para melhorar as previsões
  - Intervalos de previsão para quantificar incerteza
  - Múltiplas opções de frequência (diária, semanal, mensal)
  - Diferentes modelos de previsão (padrão e longo prazo)
- **Visualizações Interativas**
  - Gráficos interativos com Plotly
  - Exibição de intervalos de confiança
  - Destaque para anomalias detectadas
- **Recursos de Colaboração:**
  - Exportação automática para Google Sheets
  - Download de previsões em CSV
  - Interface intuitiva para não-especialistas
- **Performance e Estabilidade:**
  - Sistema de cache para melhorar desempenho
  - Tratamento robusto de erros
  - Proteção contra limites de quota do Google Sheets API
  - Tentativas automáticas em endpoints alternativos da API

## 🚀 Tecnologias Utilizadas

- Python 3.8+
- Streamlit (Frontend)
- TimeGPT API da Nixtla (Motor de previsão)
- Google Sheets API (Armazenamento e colaboração)
- Plotly (Visualização interativa)
- Pandas (Manipulação de dados)

## ⚙️ Configuração

1. Clone o repositório:
```bash
git clone https://github.com/ThaisMx/forecasting.git
cd forecasting
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Configure as variáveis de ambiente:
- Crie um arquivo `.env` na raiz do projeto
- Adicione suas credenciais:
```
TIMEGPT_API_KEY=sua_chave_api
GOOGLE_SHEET_ID=id_da_sua_planilha
```

4. Configure o Google Sheets:
- Certifique-se de que o arquivo de credenciais do serviço (`credentials.json`) está na raiz do projeto
- A planilha deve ter as seguintes abas:
  - Vendas (com colunas 'ds' para data e 'y' para valores)
  - Trafego (com colunas 'ds' para data e 'y' para valores)
  - CPL (com colunas 'ds' para data e 'y' para valores)

## 🎮 Como Usar

1. Inicie a aplicação:
```bash
streamlit run app.py
```

2. Acesse a interface web:
- Abra seu navegador em `http://localhost:8501`
- Use a barra lateral para configurar o horizonte de previsão e outras opções
- Escolha entre configurações básicas e avançadas:
  - Básicas: horizonte, frequência, modelo
  - Avançadas: intervalos de previsão, fine-tuning, validação cruzada, detecção de anomalias
- Selecione a métrica desejada nas abas
- Clique em "Gerar Previsão" para criar novas previsões

## 📈 Estrutura do Projeto

```
forecasting/
├── app.py                 # Aplicação principal (Streamlit)
├── sheets_connector.py    # Conexão com Google Sheets
├── timegpt_forecaster.py  # Integração com TimeGPT
├── requirements.txt       # Dependências
├── .env                   # Variáveis de ambiente
└── README.md              # Documentação
```

## 🚀 Recursos Avançados

### Cross-Validation
Utilize a validação cruzada para avaliar a performance do modelo em diferentes janelas temporais, garantindo robustez.

### Fine-Tuning
Adapte o modelo TimeGPT ao seu caso específico, escolhendo a função de perda que melhor representa sua métrica de negócio.

### Variáveis Exógenas
Melhore suas previsões incluindo variáveis externas que influenciam sua série temporal, como datas especiais, eventos, ou outras métricas correlacionadas.

### Detecção de Anomalias
Identifique outliers e valores anômalos em seus dados históricos para compreender melhor padrões incomuns.

## 🆕 Melhorias Implementadas

### 1. Tratamento de Coeficiente de Variação (CV) Baixo
O sistema agora detecta automaticamente quando o TimeGPT retorna previsões com variabilidade muito baixa (CV < 1%) e:
- Registra avisos detalhados no log
- Recria previsões mais realistas baseadas nas tendências históricas
- Permite configurar o limite do CV através do parâmetro `cv_threshold`

### 2. Suporte a Horizontes Longos
- Detecção automática de horizontes maiores que 30 períodos
- Troca automática para o modelo `timegpt-1-long-horizon` quando necessário
- Extensão inteligente de previsões quando a API retorna menos valores que o esperado

### 3. Resiliência a Erros
- Tratamento de avisos específicos como "The specified horizon exceeds the model horizon"
- Fallback automático para modelos alternativos quando apropriado
- Retentativas com backoff exponencial para erros temporários
- Análise detalhada de respostas da API para extrair informações úteis mesmo em casos de erro

### 4. Adaptação a Diferentes Estruturas de Resposta
- Suporte a múltiplos formatos de resposta da API TimeGPT
- Identificação automática de campos em respostas desconhecidas
- Capacidade de migrar entre versões da API sem alterações no código

## 🛠️ Solução de Problemas

### Erros de Quota no Google Sheets
Se encontrar o erro "Quota exceeded for quota metric 'Read requests'":
- Aguarde alguns minutos antes de tentar novamente
- Utilize o botão "Limpar Cache" nas opções avançadas
- Evite recarregar a página repetidamente

### Erros na API do TimeGPT
Se encontrar erros de API:
- Verifique se sua chave API está correta em `.env`
- O sistema tentará endpoints alternativos automaticamente
- Confira se você está dentro dos limites de uso da sua conta TimeGPT

### Previsões com Valores Muito Similares
Se suas previsões apresentarem quase o mesmo valor em todos os períodos:
- Isso pode indicar um aviso do TimeGPT sobre horizonte de previsão longo demais
- Tente reduzir o horizonte de previsão ou use o modelo `timegpt-1-long-horizon`
- Ajuste o parâmetro `cv_threshold` (padrão é 1.0%) para valores como 0.5% para ser mais sensível a variações pequenas
- Verifique se seus dados históricos têm variabilidade suficiente
- Use o botão "Limpar Cache" para garantir novos resultados da API

### Previsões com Valores Muito Baixos
Se suas previsões estiverem retornando valores muito abaixo do que é aceitável para seu negócio:
- Use a opção "Fator de valor mínimo" nas configurações avançadas
- Esta opção garante que os valores previstos não sejam menores que uma porcentagem da média histórica
- Por exemplo, um fator de 0.5 (50%) para dados com média histórica de 47 reais estabeleceria um piso de 23.5 reais
- Aumente este valor para garantir previsões mais próximas da média histórica
- Valores sugeridos: 0.7 (70%) para previsões conservadoras, 0.5 (50%) para previsões mais flexíveis

### Outros Problemas Comuns
- **Cache desatualizado**: Use o botão "Limpar Cache" nas opções avançadas
- **Dados não aparecem**: Verifique o formato das colunas no Google Sheets (necessário 'ds' e 'y')
- **Erro "Not Found"**: O sistema tentará endpoints alternativos automaticamente
- **Previsões inconsistentes**: Verifique se sua série temporal tem dados suficientes e padrões consistentes

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor, sinta-se à vontade para submeter um Pull Request.

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 👩‍💻 Autora

Thais Maximiana
- GitHub: [@ThaisMx](https://github.com/ThaisMx/) 