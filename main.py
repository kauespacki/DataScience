import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
import dash_bootstrap_components as dbc
from scipy.optimize import curve_fit

# Tenta importar emcee, mas define uma flag se não estiver disponível
try:
    import emcee

    EMCEE_AVAILABLE = True
except ImportError:
    EMCEE_AVAILABLE = False
    print("Biblioteca 'emcee' não encontrada. O modo Bayesiano MCMC Real será desativado (usará simulação).")

# ============================
# DADOS
# ============================

coluna_2024 = [
    17.7, 17.7, 17.8, 17.6, 17.6, 17.8, 17.8, 17.6, 17, 16.4, 18.6, 18.7, 20.2, 20.9, 23.8, 24, 25.6, 26.6, 27.1, 27,
    25.5, 24.9, 21.9, 20.5, 19.6, 19.4, 19, 18.7, 18.5, 17.5, 17.3, 17, 16.8, 16.5, 18.6, 20.1, 22.1, 24.2, 25.5, 26.1,
    27, 27.3, 27.9, 27.6, 26.1, 25.7, 22.5, 21.1, 20.2, 19.8, 19.5, 19.5, 19.2, 18.7, 18.3, 18.4, 17.8, 17.8, 19.4,
    22.2,
    23.2, 25.2, 27, 27.5, 28.5, 29.8, 27.3, 26, 23.5, 22.4, 21, 20.2, 18.6, 18.4, 18.4, 18.2, 18, 18.2, 18.2, 18.1,
    17.7,
    17.6, 17.7, 18.5, 19.5, 20.9, 22.7, 23.1, 25.6, 25.7, 26.2, 26.3, 25.5, 23.5, 22.2, 20.4, 19.8, 19.1, 19, 18.3,
    17.4,
    17.2, 17.6, 17.3, 16.9, 16.6, 17.5, 18.6, 19.9, 22.6, 23, 24.7, 26.8, 28, 28, 26.3, 25.3, 24.9, 22.4, 20.7, 19.9,
    19.7, 19.6, 19.2, 19.2, 19, 18.6, 18.6, 18.5, 18.1, 18.8, 19.9, 22.8, 23.1, 24.2, 25.7, 27.2, 27.8, 27.9, 28, 26.5,
    25.4, 23.1, 21, 20.1, 19.9, 19.8, 19.8, 19.9, 20, 19.4, 19.2, 19, 19, 19, 19.7, 22.9, 25.4, 27.6, 29.8, 31.1, 32.1,
    32.8, 32.7, 32.1, 28.5, 26.9, 25.2
]

coluna_2025 = [
    20.4, 20.4, 20, 19.8, 19.7, 19.5, 19.2, 19.3, 19.1, 18.4, 18.4, 19.6, 21.6, 23.5, 25.7, 27.6, 28.6, 29.4, 29.8, 30,
    25.5, 22.9, 22.5, 21.2, 21.4, 20.6, 19.9, 19.6, 19.2, 18.9, 18.5, 17.9, 17.7, 17.8, 18.9, 20.9, 23.8, 25.9, 27.6,
    28.1, 28.8, 29.4, 28.1, 26.8, 24.1, 19.8, 20, 19.7, 19.6, 20, 19.9, 19.5, 18.9, 19, 18.5, 18.7, 18.3, 18.4, 19.1,
    20.8, 23.6, 24.6, 26.2, 27.2, 28.2, 29.3, 29.8, 30.8, 30.2, 28.8, 28.7, 27.6, 25.4, 24.7, 23.5, 22.6, 21.9, 21.6,
    20.5, 19.8, 19.3, 18.8, 19.4, 22.9, 24.5, 25.7, 27.1, 28.5, 29.3, 30.6, 30.7, 30.8, 30.1, 30.2, 29.2, 23.7, 22,
    21.5,
    20.9, 19.2, 18.5, 18.5, 18.5, 18.4, 18.3, 18.4, 18.5, 18.8, 19.6, 19.8, 22.7, 22.8, 25.3, 26.5, 27.8, 28.2, 26.4,
    24.8, 23.9, 21.7, 19.7, 19.5, 19.4, 19.1, 18.7, 18.4, 18.4, 18.3, 17.9, 17.9, 18.6, 19.9, 20.3, 22.2, 23.6, 25.5,
    25.9, 27.3, 26.5, 26.1, 24.6, 22.8, 20.6, 19.7, 18.9, 18.9, 19.1, 18.9, 18.8, 18.6, 18.5, 18.5, 18.4, 18, 18, 19.2,
    19.6, 20.3, 22.1, 23.9, 24.9, 24, 22.5, 22.7, 23.3, 21.5, 20.1, 19.5
]

x = np.arange(len(coluna_2024))


# ============================
# FUNÇÕES DE REGRESSÃO
# ============================

def parabola(x, a, b, c):
    return a * x ** 2 + b * x + c


def exponencial(x, a, b, c):
    return a * np.exp(b * x) + c


def logistica(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))


def potencia(x, a, b):
    # Evita problema com 0^b (quando b<0)
    x = np.where(x == 0, 1e-6, x)
    return a * np.power(x, b)


# ============================
# FUNÇÃO GAUSS-NEWTON (PURO)
# ============================

def gauss_newton_fit(x, y, p0, max_iter=100, tol=1e-6):
    """
    Implementa o algoritmo Gauss-Newton puro para a função exponencial.
    f(x, a, b, c) = a * exp(b*x) + c
    """
    params = np.array(p0, dtype=float)  # Garante que params seja float

    for _ in range(max_iter):
        a, b, c = params

        # 1. Calcular resíduos
        y_model = exponencial(x, a, b, c)
        residuals = y - y_model

        # 2. Calcular Jacobiana
        df_da = np.exp(b * x)
        df_db = a * x * np.exp(b * x)
        df_dc = np.ones_like(x)

        # J é (n_samples, n_params)
        J = np.stack([df_da, df_db, df_dc], axis=1)

        # 3. Resolver o sistema linear (J.T @ J) @ delta = J.T @ residuals
        # Usar np.linalg.solve é mais estável e rápido do que calcular a inversa
        try:
            JtJ = J.T @ J
            JtRes = J.T @ residuals
            # delta = (J.T * J)^-1 * (J.T * r)
            delta = np.linalg.solve(JtJ, JtRes)
        except np.linalg.LinAlgError:
            # Matriz singular, o método falha (um problema comum no Gauss-Newton puro)
            # Retorna os últimos parâmetros válidos
            return params

            # 4. Atualizar parâmetros
        params = params + delta

        # 5. Checar convergência
        if np.sum(delta ** 2) < tol ** 2:
            break

    return params


def ajustar_modelo_gn(x, y, p0):
    """Wrapper para o gauss_newton_fit para tratar erros e retornar a curva."""
    try:
        popt = gauss_newton_fit(x, np.array(y), p0)
        return exponencial(x, *popt)
    except Exception as e:
        # Captura outros erros (ex: overflow no np.exp)
        return np.full_like(y, np.nan)


# ============================
# FUNÇÕES PARA MCMC BAYESIANO (emcee)
# Parâmetros: [a, b, c]
# ============================

# Verifica se o emcee está disponível antes de definir funções que dependem dele
if EMCEE_AVAILABLE:
    def log_prior(params):
        """Define a probabilidade prévia dos parâmetros (priors)."""
        a, b, c = params
        # Define priors "planos" (pouca informação prévia)
        # Supomos que 'a' e 'c' estão entre 0 e 50 (razoável para temp)
        # --- CORREÇÃO: O prior de 'b' de -0.1 a 0.1 estava muito frouxo, causando divergências.
        # --- Restringindo para -0.01 a 0.01 ---
        if 0.0 < a < 50.0 and -0.01 < b < 0.01 and 0.0 < c < 50.0:
            return 0.0  # Probabilidade logarítmica de 0 (probabilidade de 1)
        return -np.inf  # Probabilidade logarítmica de -infinito (probabilidade de 0)


    def log_likelihood(params, x, y_obs, y_err):
        """Define a verossimilhança (likelihood) - quão bem o modelo se ajusta aos dados."""
        a, b, c = params
        y_model = exponencial(x, a, b, c)

        # Supõe que os erros são Gaussianos
        # log(L) = -0.5 * sum( ((y_obs - y_model) / y_err)**2 + log(2*pi*y_err**2) )
        sigma2 = y_err ** 2
        return -0.5 * np.sum((y_obs - y_model) ** 2 / sigma2 + np.log(2 * np.pi * sigma2))


    def log_probability(params, x, y_obs, y_err):
        """Combina o prior e a likelihood."""
        lp = log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        ll = log_likelihood(params, x, y_obs, y_err)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll


# ============================
# FUNÇÃO DE AJUSTE (SCIPY)
# ============================

def ajustar_modelo(modelo, x, y, p0=None, method='lm'):
    """
    Ajusta um modelo aos dados x, y, usando scipy.curve_fit.
    Métodos comuns: 'lm' (padrão), 'trf', 'dogbox'.
    """
    try:
        # Passa o 'method' para o curve_fit
        popt, _ = curve_fit(modelo, x, y, p0=p0, method=method, maxfev=5000)
        return modelo(x, *popt)
    except (RuntimeError, TypeError, ValueError):
        # Caso o ajuste falhe (ex: p0 ruins ou método incompatível), retorna NaNs
        return np.full_like(y, np.nan)


# Ajustes com parâmetros iniciais (cálculos não exponenciais)
# O ajuste exponencial será feito DENTRO do callback
y_linear_2024 = np.polyval(np.polyfit(x, coluna_2024, 1), x)
y_linear_2025 = np.polyval(np.polyfit(x, coluna_2025, 1), x)

y_parab_2024 = ajustar_modelo(parabola, x, coluna_2024)
y_parab_2025 = ajustar_modelo(parabola, x, coluna_2025)

y_log_2024 = ajustar_modelo(logistica, x, coluna_2024, p0=(max(coluna_2024), 0.05, len(x) / 2))
y_log_2025 = ajustar_modelo(logistica, x, coluna_2025, p0=(max(coluna_2025), 0.05, len(x) / 2))

y_pot_2024 = ajustar_modelo(potencia, x, coluna_2024, p0=(1, 0.01))
y_pot_2025 = ajustar_modelo(potencia, x, coluna_2025, p0=(1, 0.01))


# ============================
# MÉTRICAS
# ============================

def calcular_metricas(y_real, y_pred):
    mask = ~np.isnan(y_pred)
    if mask.sum() == 0:  # Se todos os y_pred forem NaN
        return np.nan, np.nan
    y_real_masked = y_real[mask]
    y_pred_masked = y_pred[mask]

    if len(y_real_masked) == 0:
        return np.nan, np.nan

    ss_res = np.sum((y_real_masked - y_pred_masked) ** 2)
    ss_tot = np.sum((y_real_masked - np.mean(y_real_masked)) ** 2)

    if ss_tot == 0:  # Evita divisão por zero se os dados reais forem constantes
        return np.nan, np.nan

    r2 = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(ss_res / len(y_real_masked))
    return r2, rmse


# ============================
# DASH
# ============================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
app.title = "Análise de Temperaturas - Data Science View"

# Imagens das teorias (verificar se estão em assets/)
# Certifique-se de ter uma pasta 'assets' no mesmo diretório do seu app_corrigido.py
# e que as imagens estejam lá.
imagens_teorias = {
    "Teorema Central do Limite": "assets/teorema.jpg",
    "Correlação": "assets/correlacao.jpg",
    "Amostragem, Distribuição Normal (Curva de Gauss ou Poisson)": "assets/amostragem.jpg",
    "T-Student": "assets/t-student.png",
    "Qui-quadrado": "assets/qui-quadrado.png"
}

# Define o label do Bayes baseado na disponibilidade da biblioteca
bayes_label = 'Métodos Bayesianos (MCMC Real)' if EMCEE_AVAILABLE else 'Métodos Bayesianos (Simulação)'

# ============================
# LAYOUT
# ============================

app.layout = dbc.Container([
    dbc.Row([dbc.Col(html.H2("📈 Análise de Temperaturas - Curitiba",
                             className="text-center text-light mt-3 mb-4"))]),

    dbc.Row([
        dbc.Col([
            html.Label("Selecione o Tipo de Regressão:", style={"color": "white", "fontSize": "18px"}),
            dcc.Dropdown(
                id='tipo-regressao',
                options=[
                    {'label': 'Linear', 'value': 'linear'},
                    {'label': 'Parabólica', 'value': 'parab'},
                    {'label': 'Exponencial', 'value': 'exp'},
                    {'label': 'Logística', 'value': 'log'},
                    {'label': 'Potência', 'value': 'pot'}
                ],
                value='linear',
                clearable=False,
                className="mb-3",
                style={'color': '#000'}
            ),

            # --- DROPDOWN ATUALIZADO ---
            html.Div(id='opcoes-otimizacao-exp', children=[
                html.Label("Método de Estimação (para Exponencial):", style={"color": "white", "fontSize": "16px"}),
                dcc.Dropdown(
                    id='metodo-otimizacao',
                    options=[
                        {'label': 'Algoritmo de Levenberg-Marquardt (Padrão)', 'value': 'lm'},
                        {'label': 'Mínimos Quadrados Não Linear (via TRF)', 'value': 'trf'},
                        {'label': 'Máxima Verossimilhança (MLE, via Dogbox)', 'value': 'dogbox'},
                        {'label': 'Gauss-Newton (Puro)', 'value': 'gauss_newton'},  # ADICIONADO DE VOLTA
                        {'label': bayes_label, 'value': 'bayes', 'disabled': not EMCEE_AVAILABLE}
                        # Desativa se emcee não estiver instalado
                    ],
                    value='lm',
                    clearable=False,
                    style={'color': '#000'},
                    className="mb-4"
                )
            ], style={'display': 'none'}),  # Oculto por padrão

            dcc.Graph(id='grafico-regressao', style={'height': '65vh'}),

            # --- TEXTO DE ANÁLISE RESUMIDA (CORRIGIDO) ---
            dcc.Markdown("""
                **Análise de Risco e Otimização para Agritech**

                **Problema Concreto:** Uma startup de *agritech* precisa prever a variação da temperatura em Curitiba para otimizar o uso de climatizadores e irrigação em estufas urbanas. O objetivo é usar um modelo matemático para prever a temperatura máxima (pico de custo de energia) e a mínima (risco de resfriamento) ao longo do dia. O modelo mais prático é aquele que tiver o **menor Erro Médio (RMSE)**.

                **Avaliação dos Modelos:** A análise mostra um resultado misto. Para 2024, a **Regressão Exponencial** apresenta os melhores resultados (RMSE pprox 3.65), capturando a tendência inicial de subida. Para 2025, a **Regressão Parabólica** é ligeiramente melhor. A conclusão principal é que **ambos os modelos são inadequADOS** para este problema. Um R2 de 0.18 ainda é muito baixo e o modelo falha em capturar os picos e vales cíclicos, sendo inútil para prever máximas e mínimas.
            """, style={'color': '#ccc', 'backgroundColor': '#2a2a2a', 'padding': '15px', 'borderRadius': '8px',
                        "marginTop": "20px"}),

            html.Hr(style={"borderColor": "#444", "marginTop": "30px"}),
            html.Label("Selecione uma teoria estatística:", style={"color": "white", "fontSize": "18px"}),
            dcc.Dropdown(
                id='dropdown-teorias',
                options=[{'label': k, 'value': k} for k in imagens_teorias.keys()],
                placeholder="Escolha uma teoria...",
                style={'color': '#000'},
                className="mb-4"
            ),
            html.Div(id="imagem-teoria", className="text-center")
        ])
    ])
], fluid=True, style={"backgroundColor": "#1E1E1E", "paddingBottom": "40px"})


# ============================
# CALLBACKS
# ============================

# --- CALLBACK ATUALIZADO ---
@app.callback(
    [Output('grafico-regressao', 'figure'),
     Output('opcoes-otimizacao-exp', 'style')],  # Nova saída para controlar a visibilidade
    [Input('tipo-regressao', 'value'),
     Input('metodo-otimizacao', 'value')]  # Nova entrada do método
)
def atualizar_grafico(tipo_regressao, metodo_opt):
    style_otimizacao = {'display': 'none'}  # Oculto por padrão
    titulo_metodo = ""

    # Converte os dados para array numpy uma vez
    y_data_2024 = np.array(coluna_2024)
    y_data_2025 = np.array(coluna_2025)

    # --- CORREÇÃO: Parâmetros iniciais (p0) melhorados ---
    # Usar o valor mínimo como estimativa para 'c' (assíntota inferior)
    # Usar o primeiro ponto (x=0) para estimar 'a' (pois y[0] = a*exp(0) + c = a + c)

    p0_c_24 = np.min(y_data_2024)
    p0_a_24 = y_data_2024[0] - p0_c_24
    # Garante que 'a' seja um valor positivo pequeno se y[0] for o mínimo
    if p0_a_24 <= 0:
        p0_a_24 = 0.1
    p0_exp_24 = (p0_a_24, 0.001, p0_c_24)  # (a, b, c)

    p0_c_25 = np.min(y_data_2025)
    p0_a_25 = y_data_2025[0] - p0_c_25
    if p0_a_25 <= 0:
        p0_a_25 = 0.1
    p0_exp_25 = (p0_a_25, 0.001, p0_c_25)
    # --- FIM DA CORREÇÃO ---

    # Cria a figura base. Os traços de dados são adicionados no final.
    fig = go.Figure()

    if tipo_regressao == 'exp':
        # 1. Mostra o dropdown de otimização
        style_otimizacao = {'display': 'block'}

        # (Os p0 agora são definidos acima)

        # --- LÓGICA ATUALIZADA: MCMC REAL OU SIMULAÇÃO ---
        if metodo_opt == 'bayes':
            if EMCEE_AVAILABLE:
                # --- Implementação Real Bayesiana com emcee (LENTO) ---
                titulo = "Exponencial"
                titulo_metodo = " (Método: Bayesiano (MCMC Real))"

                # --- MCMC para 2024 ---
                try:
                    y_err_24 = np.std(y_data_2024) * 0.5  # 1. Estima erro dos dados
                    # Usa os p0 melhorados para o ajuste base
                    popt_base_24, _ = curve_fit(exponencial, x, y_data_2024, p0=p0_exp_24, method='lm')
                    nwalkers = 32
                    ndim = 3
                    p0_walkers = popt_base_24 + 1e-4 * np.random.randn(nwalkers, ndim)  # 2. Posições iniciais

                    # 3. Configurar e rodar o sampler
                    sampler24 = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y_data_2024, y_err_24))
                    sampler24.run_mcmc(p0_walkers, 500, progress=False, skip_initial_state_check=True)  # 500 passos

                    # 4. Coletar amostras (descarta 100, afina por 10)
                    samples_24 = sampler24.get_chain(discard=100, thin=10, flat=True)

                    all_y1 = []
                    indices_24 = np.random.randint(len(samples_24), size=100)  # Pega 100 amostras

                    for idx in indices_24:
                        params_sample = samples_24[idx]
                        y_sample = exponencial(x, *params_sample)
                        all_y1.append(y_sample)
                        fig.add_trace(go.Scatter(x=x, y=y_sample, mode='lines',
                                                 line=dict(color='#00BFFF', width=0.5),
                                                 opacity=0.1, showlegend=False, hoverinfo='none'))
                    y1 = np.mean(all_y1, axis=0)  # Linha principal é a média

                except Exception as e:
                    print(f"Erro no MCMC 2024: {e}")
                    y1 = np.full_like(y_data_2024, np.nan)

                # --- MCMC para 2025 ---
                try:
                    y_err_25 = np.std(y_data_2025) * 0.5
                    popt_base_25, _ = curve_fit(exponencial, x, y_data_2025, p0=p0_exp_25, method='lm')
                    p0_walkers_25 = popt_base_25 + 1e-4 * np.random.randn(nwalkers, ndim)

                    sampler25 = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y_data_2025, y_err_25))
                    sampler25.run_mcmc(p0_walkers_25, 500, progress=False, skip_initial_state_check=True)

                    samples_25 = sampler25.get_chain(discard=100, thin=10, flat=True)
                    all_y2 = []
                    indices_25 = np.random.randint(len(samples_25), size=100)

                    for idx in indices_25:
                        params_sample = samples_25[idx]
                        y_sample = exponencial(x, *params_sample)
                        all_y2.append(y_sample)
                        fig.add_trace(go.Scatter(x=x, y=y_sample, mode='lines',
                                                 line=dict(color='#FF6347', width=0.5),
                                                 opacity=0.1, showlegend=False, hoverinfo='none'))
                    y2 = np.mean(all_y2, axis=0)

                except Exception as e:
                    print(f"Erro no MCMC 2025: {e}")
                    y2 = np.full_like(y_data_2025, np.nan)

            else:
                # --- Fallback: Simulação Visual (se emcee não estiver instalado) ---
                titulo = "Exponencial"
                titulo_metodo = " (Método: Bayesiano (Simulação))"

                # Ajuste base para 2024
                try:
                    popt_base_24, _ = curve_fit(exponencial, x, y_data_2024, p0=p0_exp_24, method='lm', maxfev=5000)
                    all_y1 = []
                    for _ in range(50):
                        p_sample = np.copy(popt_base_24)
                        p_sample[0] = p_sample[0] * np.random.normal(1, 0.05)
                        p_sample[1] = p_sample[1] + np.random.normal(0, 0.002)  # Ruído ADITIVO
                        p_sample[2] = p_sample[2] * np.random.normal(1, 0.05)
                        y_sample = exponencial(x, *params_sample)
                        all_y1.append(y_sample)
                        fig.add_trace(go.Scatter(x=x, y=y_sample, mode='lines',
                                                 line=dict(color='#00BFFF', width=0.5),
                                                 opacity=0.1, showlegend=False, hoverinfo='none'))
                    y1 = np.mean(all_y1, axis=0)
                except (RuntimeError, TypeError, ValueError):
                    y1 = np.full_like(y_data_2024, np.nan)

                # Ajuste base para 2025
                try:
                    popt_base_25, _ = curve_fit(exponencial, x, y_data_2025, p0=p0_exp_25, method='lm', maxfev=5000)
                    all_y2 = []
                    for _ in range(50):
                        p_sample = np.copy(popt_base_25)
                        p_sample[0] = p_sample[0] * np.random.normal(1, 0.05)
                        p_sample[1] = p_sample[1] + np.random.normal(0, 0.002)  # Ruído ADITIVO
                        p_sample[2] = p_sample[2] * np.random.normal(1, 0.05)
                        y_sample = exponencial(x, *p_sample)
                        all_y2.append(y_sample)
                        fig.add_trace(go.Scatter(x=x, y=y_sample, mode='lines',
                                                 line=dict(color='#FF6347', width=0.5),
                                                 opacity=0.1, showlegend=False, hoverinfo='none'))
                    y2 = np.mean(all_y2, axis=0)
                except (RuntimeError, TypeError, ValueError):
                    y2 = np.full_like(y_data_2025, np.nan)

        else:
            # --- Lógica para LM, TRF, Dogbox, E AGORA GAUSS-NEWTON ---

            if metodo_opt == 'gauss_newton':
                y1 = ajustar_modelo_gn(x, y_data_2024, p0=p0_exp_24)
                y2 = ajustar_modelo_gn(x, y_data_2025, p0=p0_exp_25)
                titulo_metodo_str = 'Gauss-Newton (Puro)'
            else:
                # Lógica original para os métodos do Scipy
                scipy_method = metodo_opt
                y1 = ajustar_modelo(exponencial, x, y_data_2024, p0=p0_exp_24, method=scipy_method)
                y2 = ajustar_modelo(exponencial, x, y_data_2025, p0=p0_exp_25, method=scipy_method)

                metodo_map = {
                    'lm': 'Levenberg-Marquardt',
                    'trf': 'NLS (via TRF)',
                    'dogbox': 'MLE (via Dogbox)',
                }
                titulo_metodo_str = metodo_map.get(metodo_opt, metodo_opt.upper())

            titulo = "Exponencial"
            titulo_metodo = f" (Método: {titulo_metodo_str})"

    else:
        # Para outras regressões, usa os valores pré-calculados
        modelos = {
            'linear': (y_linear_2024, y_linear_2025, "Linear"),
            'parab': (y_parab_2024, y_parab_2025, "Parabólica"),
            'log': (y_log_2024, y_log_2025, "Logística"),
            'pot': (y_pot_2024, y_pot_2025, "Potência")
        }
        y1, y2, titulo = modelos[tipo_regressao]

    # Calcula métricas
    r2_2024, rmse_2024 = calcular_metricas(y_data_2024, y1)
    r2_2025, rmse_2025 = calcular_metricas(y_data_2025, y2)

    # --- Gráfico ---
    # Adiciona os dados de scatter (pontos)
    fig.add_trace(go.Scatter(x=x, y=coluna_2024, mode='markers', name='2024',
                             marker=dict(color='#00BFFF', size=6, opacity=0.8),
                             hovertemplate='Hora: %{x}<br>Temp: %{y:.2f}°C'))
    fig.add_trace(go.Scatter(x=x, y=coluna_2025, mode='markers', name='2025',
                             marker=dict(color='#FF6347', size=6, opacity=0.8),
                             hovertemplate='Hora: %{x}<br>Temp: %{y:.2f}°C'))

    # Adiciona as linhas de ajuste PRINCIPAIS
    fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='Ajuste 2024',
                             line=dict(color='#00BFFF', width=2.5),
                             hovertemplate='Hora: %{x}<br>Ajuste: %{y:.2f}°C'))
    fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name='Ajuste 2025',
                             line=dict(color='#FF6347', width=2.5),
                             hovertemplate='Hora: %{x}<br>Ajuste: %{y:.2f}°C'))

    # Formata R² e RMSE para exibição, tratando NaNs
    r2_2024_str = f"{r2_2024:.4f}" if not np.isnan(r2_2024) else "N/A"
    rmse_2024_str = f"{rmse_2024:.3f}" if not np.isnan(rmse_2024) else "N/A"
    r2_2025_str = f"{r2_2025:.4f}" if not np.isnan(r2_2025) else "N/A"
    rmse_2025_str = f"{rmse_2025:.3f}" if not np.isnan(rmse_2025) else "N/A"

    fig.update_layout(
        title=dict(
            text=f"Regressão {titulo}{titulo_metodo}<br><sup style='color:#AAA'>"
                 f"R² (2024): {r2_2024_str} | RMSE (2024): {rmse_2024_str} &nbsp; "
                 f"R² (2025): {r2_2025_str} | RMSE (2025): {rmse_2025_str}</sup>",
            x=0.5, font=dict(size=22, color='white')
        ),
        xaxis=dict(title='Horas', gridcolor='#333', showspikes=True),
        yaxis=dict(title='Temperatura (°C)', gridcolor='#333', showspikes=True),
        hovermode='x unified',
        paper_bgcolor='#1E1E1E',
        plot_bgcolor='#1E1E1E',
        font=dict(color='white'),
        legend=dict(bgcolor='rgba(0,0,0,0.3)', bordercolor='#444', borderwidth=1)
    )

    # Retorna a figura e o novo estilo para o dropdown
    return fig, style_otimizacao


@app.callback(
    Output("imagem-teoria", "children"),
    Input("dropdown-teorias", "value")
)
def mostrar_imagem(teoria):
    if teoria is None:
        return html.P("Selecione uma teoria para visualizar.", style={"color": "#bbb", "fontSize": "18px"})
    caminho = imagens_teorias.get(teoria)
    if caminho is None:
        # Tenta carregar mesmo assim, caso o 'assets/' seja adicionado pelo Dash
        caminho = teoria

        # Adiciona uma verificação simples para imagens de exemplo se a pasta assets não estiver configurada
    # Esta parte é mais para robustez, as imagens reais devem estar em 'assets/'
    if teoria == "T-Student":
        caminho = "assets/t-student.png"
    elif teoria == "Qui-quadrado":
        caminho = "assets/qui-quadrado.png"

    return html.Img(src=app.get_asset_url(caminho.replace("assets/", "")), style={
        "maxWidth": "70%",  # Use maxWidth para responsividade
        "height": "auto",
        "borderRadius": "12px",
        "boxShadow": "0 0 15px rgba(255,255,200,0.2)",  # Sombra com cor atualizada
        "marginTop": "20px"
    })


# ============================
# RUN
# ============================

if __name__ == '__main__':
    app.run(debug=True)

