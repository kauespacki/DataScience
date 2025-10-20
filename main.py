import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
import dash_bootstrap_components as dbc
from scipy.optimize import curve_fit, minimize, least_squares

# ============================
# DADOS (mantidos)
# ============================

coluna_2024 = [
    17.7, 17.7, 17.8, 17.6, 17.6, 17.8, 17.8, 17.6, 17, 16.4, 18.6, 18.7, 20.2, 20.9, 23.8, 24, 25.6, 26.6, 27.1, 27,
    25.5, 24.9, 21.9, 20.5, 19.6, 19.4, 19, 18.7, 18.5, 17.5, 17.3, 17, 16.8, 16.5, 18.6, 20.1, 22.1, 24.2, 25.5, 26.1,
    27, 27.3, 27.9, 27.6, 26.1, 25.7, 22.5, 21.1, 20.2, 19.8, 19.5, 19.5, 19.2, 18.7, 18.3, 18.4, 17.8, 17.8, 19.4, 22.2,
    23.2, 25.2, 27, 27.5, 28.5, 29.8, 27.3, 26, 23.5, 22.4, 21, 20.2, 18.6, 18.4, 18.4, 18.2, 18, 18.2, 18.2, 18.1, 17.7,
    17.6, 17.7, 18.5, 19.5, 20.9, 22.7, 23.1, 25.6, 25.7, 26.2, 26.3, 25.5, 23.5, 22.2, 20.4, 19.8, 19.1, 19, 18.3, 17.4,
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
    20.5, 19.8, 19.3, 18.8, 19.4, 22.9, 24.5, 25.7, 27.1, 28.5, 29.3, 30.6, 30.7, 30.8, 30.1, 30.2, 29.2, 23.7, 22, 21.5,
    20.9, 19.2, 18.5, 18.5, 18.5, 18.4, 18.3, 18.4, 18.5, 18.8, 19.6, 19.8, 22.7, 22.8, 25.3, 26.5, 27.8, 28.2, 26.4,
    24.8, 23.9, 21.7, 19.7, 19.5, 19.4, 19.1, 18.7, 18.4, 18.4, 18.3, 17.9, 17.9, 18.6, 19.9, 20.3, 22.2, 23.6, 25.5,
    25.9, 27.3, 26.5, 26.1, 24.6, 22.8, 20.6, 19.7, 18.9, 18.9, 19.1, 18.9, 18.8, 18.6, 18.5, 18.5, 18.4, 18, 18, 19.2,
    19.6, 20.3, 22.1, 23.9, 24.9, 24, 22.5, 22.7, 23.3, 21.5, 20.1, 19.5
]

x = np.arange(len(coluna_2024))
x_float = x.astype(float)

# ============================
# MODELO EXPONENCIAL
# ============================
def modelo_exp(x, a, b, c):
    return a * np.exp(b * x) + c

# ============================
# MÉTRICAS
# ============================
def calcular_metricas(y_real, y_pred):
    mask = ~np.isnan(y_pred)
    if mask.sum() == 0:
        return np.nan, np.nan
    ss_res = np.sum((y_real[mask] - y_pred[mask]) ** 2)
    ss_tot = np.sum((y_real[mask] - np.mean(y_real[mask])) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
    rmse = np.sqrt(ss_res / mask.sum())
    return r2, rmse

# ============================
# FUNÇÕES DE AJUSTE (vários métodos)
# ============================

def fit_curve_fit(x, y, p0=(1, 0.001, 15)):
    try:
        popt, _ = curve_fit(modelo_exp, x, y, p0=p0, maxfev=10000)
        return popt, modelo_exp(x, *popt)
    except Exception as e:
        # fallback: try different p0
        try:
            popt, _ = curve_fit(modelo_exp, x, y, p0=(np.mean(y), 0.0, 0.0), maxfev=10000)
            return popt, modelo_exp(x, *popt)
        except Exception:
            return None, np.full_like(y, np.nan)

def fit_mle(x, y, p0=(1, 0.001, 15)):
    # assume gaussian errors with unknown sigma; maximize likelihood => minimize negative log-likelihood
    def nll(theta):
        a, b, c, logsig = theta
        sigma = np.exp(logsig)
        mu = modelo_exp(x, a, b, c)
        # negative log-likelihood
        n = len(y)
        res = 0.5 * n * np.log(2 * np.pi * sigma**2) + 0.5 * np.sum((y - mu)**2) / (sigma**2)
        return res
    x0 = np.array([p0[0], p0[1], p0[2], np.log(np.std(y) if np.std(y)>0 else 1.0)])
    bounds = [(-np.inf, np.inf), (-1, 1), (-np.inf, np.inf), (None, None)]
    try:
        res = minimize(nll, x0, method='L-BFGS-B')
        if not res.success:
            # try unconstrained
            res = minimize(nll, x0, method='BFGS')
        a, b, c, logsig = res.x
        popt = np.array([a, b, c])
        yhat = modelo_exp(x, *popt)
        return popt, yhat
    except Exception:
        return None, np.full_like(y, np.nan)

def fit_gauss_newton(x, y, p0=(1, 0.001, 15), max_iter=100, tol=1e-8):
    # Gauss-Newton for least squares (nonlinear)
    a, b, c = p0
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    for i in range(max_iter):
        expbx = np.exp(b * x_arr)
        mu = a * expbx + c
        r = y_arr - mu  # residuals
        # Jacobian J shape (n, 3)
        J = np.vstack([expbx, a * x_arr * expbx, np.ones_like(x_arr)]).T  # d/d a, d/d b, d/d c
        # normal eq: J^T J delta = J^T r
        JTJ = J.T @ J
        JTr = J.T @ r
        try:
            delta = np.linalg.solve(JTJ, JTr)
        except np.linalg.LinAlgError:
            # ill-conditioned, return nan
            return None, np.full_like(y_arr, np.nan)
        # update parameters (note: because r = y - mu, and delta solves for J^T J delta = J^T r,
        # we add delta to parameters)
        a_new = a + delta[0]
        b_new = b + delta[1]
        c_new = c + delta[2]
        if np.linalg.norm(delta) < tol:
            a, b, c = a_new, b_new, c_new
            break
        a, b, c = a_new, b_new, c_new
    popt = np.array([a, b, c])
    return popt, modelo_exp(x_arr, *popt)

def fit_levenberg_marquardt(x, y, p0=(1, 0.001, 15)):
    # use least_squares with method='lm'
    try:
        def resid(theta):
            a, b, c = theta
            return modelo_exp(x, a, b, c) - y
        res = least_squares(resid, x0=p0, method='lm', max_nfev=10000)
        if not res.success:
            return None, np.full_like(y, np.nan)
        popt = res.x
        return popt, modelo_exp(x, *popt)
    except Exception:
        return None, np.full_like(y, np.nan)

def fit_bayes_laplace(x, y, p0=(1, 0.001, 15), n_samples=2000):
    # Laplace approximation: compute MLE via least squares, approximate posterior ~ N(p_MLE, cov)
    try:
        # first get MLE (least squares)
        def resid(theta):
            a, b, c = theta
            return modelo_exp(x, a, b, c) - y
        res = least_squares(resid, x0=p0, method='trf', max_nfev=10000)
        if not res.success:
            return None, np.full_like(y, np.nan)
        popt = res.x
        # estimate sigma2
        residuals = resid(popt)
        n = len(y)
        sigma2 = np.sum(residuals**2) / max(1, n - len(popt))
        # approximate covariance: cov ≈ sigma2 * (J^T J)^{-1}
        J = res.jac  # jac shape (n, 3)
        JTJ = J.T @ J
        try:
            cov = sigma2 * np.linalg.inv(JTJ)
            # sample approximate posterior
            samples = np.random.multivariate_normal(popt, cov, size=n_samples)
            y_preds = np.array([modelo_exp(x, *s) for s in samples])  # shape (n_samples, n)
            y_mean = np.mean(y_preds, axis=0)
            y_std = np.std(y_preds, axis=0)
            # return posterior mean prediction (and optionally std)
            return popt, y_mean
        except np.linalg.LinAlgError:
            # fallback: return MLE prediction
            return popt, modelo_exp(x, *popt)
    except Exception:
        return None, np.full_like(y, np.nan)

# wrapper that tries all methods for a given x,y and returns dict
def ajustar_exponenciais(x, y):
    methods = {}
    # reasonable p0: a ~ (max-min), b ~ small, c ~ min
    y_arr = np.array(y)
    p0 = (max(1e-3, np.max(y_arr) - np.min(y_arr)), 0.0, np.min(y_arr))
    # 1) curve_fit (nonlinear least squares)
    popt_cf, y_cf = fit_curve_fit(x, y_arr, p0=(p0[0], 0.0005, p0[2]))
    methods['Nonlinear LS (curve_fit)'] = {'popt': popt_cf, 'yhat': y_cf}

    # 2) MLE
    popt_mle, y_mle = fit_mle(x, y_arr, p0=(p0[0], 0.0, p0[2]))
    methods['MLE'] = {'popt': popt_mle, 'yhat': y_mle}

    # 3) Gauss-Newton
    popt_gn, y_gn = fit_gauss_newton(x, y_arr, p0=(p0[0], 0.0, p0[2]))
    methods['Gauss-Newton'] = {'popt': popt_gn, 'yhat': y_gn}

    # 4) Levenberg-Marquardt
    popt_lm, y_lm = fit_levenberg_marquardt(x, y_arr, p0=(p0[0], 0.0, p0[2]))
    methods['Levenberg-Marquardt'] = {'popt': popt_lm, 'yhat': y_lm}

    # 5) Bayesian (Laplace approx)
    popt_b, y_bayes = fit_bayes_laplace(x, y_arr, p0=(p0[0], 0.0, p0[2]), n_samples=1500)
    methods['Bayes (Laplace approx)'] = {'popt': popt_b, 'yhat': y_bayes}

    return methods

# Precompute (pode-se computar on-the-fly no callback, mas já deixo função pronta)
# ============================
# IMAGENS DAS TEORIAS (mantidas)
# ============================
imagens_teorias = {
    "Teorema Central do Limite": "assets/teorema.jpg",
    "Correlação": "assets/correlacao.jpg",
    "Amostragem, Distribuição Normal (Curva de Gauss ou Poisson)": "assets/amostragem.jpg",
    "T-Student": "assets/t-student.png",
    "Qui-quadrado": "assets/qui-quadrado.png"
}

# ============================
# DASH APP (layout atualizado)
# ============================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "Clima em Curitiba - Ajustes Exponenciais"

app.layout = dbc.Container([
    dbc.Row([dbc.Col(html.H1("Clima em Curitiba", className="text-center mb-4 mt-2", style={"color": "white"}))]),

    dbc.Row([
        dbc.Col([
            html.Label("Tipo de Regressão", style={"color": "white"}),
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
                className="mb-2",
                style={'color': '#000'}
            ),
            # Método para exponencial (aparece só se 'exp' selecionado)
            html.Div(id='div-metodo-exp', children=[
                html.Label("Método de otimização (apenas para Exponencial):", style={"color": "white", "marginTop": "6px"}),
                dcc.Dropdown(
                    id='metodo-exp',
                    options=[
                        {'label': 'Nonlinear LS (curve_fit)', 'value': 'curve_fit'},
                        {'label': 'Máxima Verossimilhança (MLE)', 'value': 'mle'},
                        {'label': 'Gauss-Newton', 'value': 'gn'},
                        {'label': 'Levenberg-Marquardt', 'value': 'lm'},
                        {'label': 'Bayes (Laplace approx)', 'value': 'bayes'},
                        {'label': 'Comparar todos métodos', 'value': 'all'}
                    ],
                    value='curve_fit',
                    clearable=False,
                    className="mb-4",
                    style={'color': '#000'}
                )
            ], style={"display": "none"}),

            dcc.Graph(id='grafico-regressao', style={'height': '65vh'}),

            html.Hr(style={"borderColor": "#444"}),

            html.Label("Selecione uma teoria estatística:", style={"color": "white", "fontSize": "18px"}),
            dcc.Dropdown(
                id='dropdown-teorias',
                options=[{'label': k, 'value': k} for k in imagens_teorias.keys()],
                placeholder="Escolha uma teoria...",
                style={'color': '#000'},
                className="mb-4"
            ),
            html.Div(id="imagem-teoria", className="text-center"),
            html.Div(id='div-metricas', style={"marginTop": "18px"})
        ])
    ])
], fluid=True, style={"backgroundColor": "#121212", "paddingBottom": "50px"})

# ============================
# CALLBACKS
# ============================

@app.callback(
    Output('div-metodo-exp', 'style'),
    Input('tipo-regressao', 'value')
)
def mostrar_metodo_exp(tipo):
    if tipo == 'exp':
        return {"display": "block"}
    return {"display": "none"}

@app.callback(
    Output('grafico-regressao', 'figure'),
    Output('div-metricas', 'children'),
    Input('tipo-regressao', 'value'),
    Input('metodo-exp', 'value')
)
def atualizar_grafico(tipo, metodo_exp):
    # precompute linear/parab etc (mantidos)
    y_linear_2024 = np.polyval(np.polyfit(x_float, coluna_2024, 1), x_float)
    y_linear_2025 = np.polyval(np.polyfit(x_float, coluna_2025, 1), x_float)
    # parab:
    coef_parab_2024 = np.polyfit(x_float, coluna_2024, 2)
    coef_parab_2025 = np.polyfit(x_float, coluna_2025, 2)
    y_parab_2024 = np.polyval(coef_parab_2024, x_float)
    y_parab_2025 = np.polyval(coef_parab_2025, x_float)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=coluna_2024, mode='lines', name='2024', line=dict(color='skyblue')))
    fig.add_trace(go.Scatter(y=coluna_2025, mode='lines', name='2025', line=dict(color='lightcoral')))

    metrics_div = html.Div()  # será substituído

    if tipo == 'linear':
        titulo = "Regressão Linear"
        fig.add_trace(go.Scatter(y=y_linear_2024, mode='lines', name='Ajuste Linear 2024', line=dict(dash='dash', color='deepskyblue')))
        fig.add_trace(go.Scatter(y=y_linear_2025, mode='lines', name='Ajuste Linear 2025', line=dict(dash='dash', color='tomato')))

        r2_2024, rmse_2024 = calcular_metricas(np.array(coluna_2024), np.array(y_linear_2024))
        r2_2025, rmse_2025 = calcular_metricas(np.array(coluna_2025), np.array(y_linear_2025))
        metrics_div = html.Div([
            html.H5("Métricas (Linear)", style={"color": "white"}),
            html.P(f"R² (2024): {r2_2024:.4f}  |  RMSE (2024): {rmse_2024:.4f}", style={"color": "#ccc"}),
            html.P(f"R² (2025): {r2_2025:.4f}  |  RMSE (2025): {rmse_2025:.4f}", style={"color": "#ccc"})
        ])

    elif tipo == 'parab':
        titulo = "Regressão Parabólica"
        fig.add_trace(go.Scatter(y=y_parab_2024, mode='lines', name='Ajuste Parab 2024', line=dict(dash='dash', color='deepskyblue')))
        fig.add_trace(go.Scatter(y=y_parab_2025, mode='lines', name='Ajuste Parab 2025', line=dict(dash='dash', color='tomato')))

        r2_2024, rmse_2024 = calcular_metricas(np.array(coluna_2024), np.array(y_parab_2024))
        r2_2025, rmse_2025 = calcular_metricas(np.array(coluna_2025), np.array(y_parab_2025))
        metrics_div = html.Div([
            html.H5("Métricas (Parabólica)", style={"color": "white"}),
            html.P(f"R² (2024): {r2_2024:.4f}  |  RMSE (2024): {rmse_2024:.4f}", style={"color": "#ccc"}),
            html.P(f"R² (2025): {r2_2025:.4f}  |  RMSE (2025): {rmse_2025:.4f}", style={"color": "#ccc"})
        ])

    elif tipo == 'exp':
        titulo = "Regressão Exponencial"
        # se o usuário quer comparar todos, calculamos todos métodos e plotamos
        resultados_2024 = ajustar_exponenciais(x_float, coluna_2024)
        resultados_2025 = ajustar_exponenciais(x_float, coluna_2025)

        # decide quais métodos plotar
        if metodo_exp == 'all':
            sel_methods = list(resultados_2024.keys())
        else:
            # map value -> key name
            map_m = {
                'curve_fit': 'Nonlinear LS (curve_fit)',
                'mle': 'MLE',
                'gn': 'Gauss-Newton',
                'lm': 'Levenberg-Marquardt',
                'bayes': 'Bayes (Laplace approx)'
            }
            sel_methods = [map_m.get(metodo_exp)]

        # add traces for selected methods
        colors_methods = {
            'Nonlinear LS (curve_fit)': 'deepskyblue',
            'MLE': 'gold',
            'Gauss-Newton': 'limegreen',
            'Levenberg-Marquardt': 'magenta',
            'Bayes (Laplace approx)': 'cyan'
        }
        # style mapping for dash
        dash_styles = {
            'Nonlinear LS (curve_fit)': 'dash',
            'MLE': 'dot',
            'Gauss-Newton': 'dashdot',
            'Levenberg-Marquardt': 'longdash',
            'Bayes (Laplace approx)': 'dash'
        }

        # build metrics table rows
        header = ["Método", "R² (2024)", "RMSE (2024)", "R² (2025)", "RMSE (2025)"]
        rows = []

        for method in resultados_2024.keys():
            yhat24 = resultados_2024[method]['yhat']
            yhat25 = resultados_2025[method]['yhat']
            r2_24, rmse_24 = calcular_metricas(np.array(coluna_2024), np.array(yhat24))
            r2_25, rmse_25 = calcular_metricas(np.array(coluna_2025), np.array(yhat25))
            rows.append((method, r2_24, rmse_24, r2_25, rmse_25))

        # add traces for only selected sel_methods (but compute metrics for all and show table)
        for method_name in sel_methods:
            if method_name is None:
                continue
            res24 = resultados_2024.get(method_name)
            res25 = resultados_2025.get(method_name)
            if res24 is not None:
                fig.add_trace(go.Scatter(y=res24['yhat'], mode='lines', name=f'{method_name} (2024)',
                                         line=dict(dash=dash_styles.get(method_name, 'dash'), color=colors_methods.get(method_name,'deepskyblue'))))
            if res25 is not None:
                fig.add_trace(go.Scatter(y=res25['yhat'], mode='lines', name=f'{method_name} (2025)',
                                         line=dict(dash=dash_styles.get(method_name, 'dash'), color=colors_methods.get(method_name,'tomato'))))

        # create HTML table for metrics
        table_header = [html.Thead(html.Tr([html.Th(h) for h in header]))]
        table_body = []
        for r in rows:
            # format numbers safely (nan handling)
            def fmt(x):
                try:
                    return f"{x:.4f}"
                except:
                    return "nan"
            table_body.append(html.Tr([html.Td(r[0]), html.Td(fmt(r[1])), html.Td(fmt(r[2])), html.Td(fmt(r[3])), html.Td(fmt(r[4]))]))
        table = html.Table(table_header + [html.Tbody(table_body)], style={"color": "#ccc", "width": "100%", "marginTop": "12px", "borderCollapse": "collapse"})
        metrics_div = html.Div([
            html.H5("Comparação de métodos (Exponencial)", style={"color": "white"}),
            table,
            html.P("Observação: 'Bayes (Laplace approx)' é uma aproximação usando Laplace em torno do MLE; para inferência Bayesiana completa use MCMC.", style={"color": "#aaa", "fontSize": "12px", "marginTop": "8px"})
        ])

    elif tipo == 'log':
        titulo = "Regressão Logística (apenas polinomial ignorada aqui)"
        # kept as placeholder - original code attempted logistic fit separately
        # We show nothing special for agora
        r2_2024, rmse_2024 = np.nan, np.nan
        r2_2025, rmse_2025 = np.nan, np.nan
        metrics_div = html.Div([
            html.P("Regressão logística não implementada para comparação de métodos aqui.", style={"color": "#ccc"})
        ])

    elif tipo == 'pot':
        titulo = "Regressão de Potência (não comparativa aqui)"
        # kept as placeholder
        metrics_div = html.Div([
            html.P("Regressão de potência não implementada para comparação de métodos aqui.", style={"color": "#ccc"})
        ])

    else:
        titulo = "Regressão"

    fig.update_layout(
        title=(f"{titulo} das Temperaturas"),
        xaxis_title='Horas (0-167)',
        yaxis_title='Temperatura (°C)',
        legend_title='Séries / Ajustes',
        template='plotly_dark',
        paper_bgcolor='#111',
        plot_bgcolor='#111'
    )

    return fig, metrics_div

@app.callback(
    Output("imagem-teoria", "children"),
    Input("dropdown-teorias", "value")
)
def mostrar_imagem(teoria):
    if teoria is None:
        return html.P("Selecione uma teoria para visualizar.", style={"color": "#bbb", "fontSize": "18px"})
    caminho = imagens_teorias[teoria]
    return html.Img(src=caminho, style={
        "width": "70%",
        "borderRadius": "12px",
        "boxShadow": "0 0 15px rgba(255,255,255,0.2)",
        "marginTop": "20px"
    })

# ============================
# EXECUÇÃO
# ============================
if __name__ == '__main__':
    app.run(debug=True)
