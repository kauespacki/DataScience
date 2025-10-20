import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
import dash_bootstrap_components as dbc
from scipy.optimize import curve_fit

# ============================
# DADOS
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

# ============================
# FUNÇÕES DE REGRESSÃO
# ============================

def parabola(x, a, b, c):
    return a*x**2 + b*x + c

def exponencial(x, a, b, c):
    return a * np.exp(b*x) + c

def logistica(x, L, k, x0):
    return L / (1 + np.exp(-k*(x - x0)))

def potencia(x, a, b):
    # Evita problema com 0^b (quando b<0)
    x = np.where(x==0, 1e-6, x)
    return a * np.power(x, b)

def ajustar_modelo(modelo, x, y, p0=None):
    try:
        popt, _ = curve_fit(modelo, x, y, p0=p0, maxfev=5000)
        return modelo(x, *popt)
    except (RuntimeError, TypeError):
        # Caso o ajuste falhe, retorna NaNs
        return np.full_like(y, np.nan)

# Ajustes com parâmetros iniciais melhores
y_linear_2024 = np.polyval(np.polyfit(x, coluna_2024, 1), x)
y_linear_2025 = np.polyval(np.polyfit(x, coluna_2025, 1), x)

y_parab_2024 = ajustar_modelo(parabola, x, coluna_2024)
y_parab_2025 = ajustar_modelo(parabola, x, coluna_2025)

y_exp_2024 = ajustar_modelo(exponencial, x, coluna_2024, p0=(1, 0.001, np.mean(coluna_2024)))
y_exp_2025 = ajustar_modelo(exponencial, x, coluna_2025, p0=(1, 0.001, np.mean(coluna_2025)))

y_log_2024 = ajustar_modelo(logistica, x, coluna_2024, p0=(max(coluna_2024), 0.05, len(x)/2))
y_log_2025 = ajustar_modelo(logistica, x, coluna_2025, p0=(max(coluna_2025), 0.05, len(x)/2))

y_pot_2024 = ajustar_modelo(potencia, x, coluna_2024, p0=(1, 0.01))
y_pot_2025 = ajustar_modelo(potencia, x, coluna_2025, p0=(1, 0.01))

# ============================
# MÉTRICAS
# ============================

def calcular_metricas(y_real, y_pred):
    mask = ~np.isnan(y_pred)
    if mask.sum() == 0:
        return np.nan, np.nan
    ss_res = np.sum((y_real[mask] - y_pred[mask]) ** 2)
    ss_tot = np.sum((y_real[mask] - np.mean(y_real[mask])) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(ss_res / len(y_real[mask]))
    return r2, rmse

# ============================
# DASH
# ============================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
app.title = "Análise de Temperaturas - Data Science View"

# Imagens das teorias (verificar se estão em assets/)
imagens_teorias = {
    "Teorema Central do Limite": "assets/teorema.jpg",
    "Correlação": "assets/correlacao.jpg",
    "Amostragem, Distribuição Normal (Curva de Gauss ou Poisson)": "assets/amostragem.jpg",
    "T-Student": "assets/t-student.png",
    "Qui-quadrado": "assets/qui-quadrado.png"
}

# ============================
# LAYOUT
# ============================

app.layout = dbc.Container([
    dbc.Row([dbc.Col(html.H2("📈 Análise de Temperaturas - Curitiba",
                             className="text-center text-light mt-3 mb-4"))]),

    dbc.Row([
        dbc.Col([
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
                className="mb-4",
                style={'color': '#000'}
            ),
            dcc.Graph(id='grafico-regressao', style={'height': '70vh'}),
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

@app.callback(
    Output('grafico-regressao', 'figure'),
    Input('tipo-regressao', 'value')
)
def atualizar_grafico(tipo):
    modelos = {
        'linear': (y_linear_2024, y_linear_2025, "Linear"),
        'parab': (y_parab_2024, y_parab_2025, "Parabólica"),
        'exp': (y_exp_2024, y_exp_2025, "Exponencial"),
        'log': (y_log_2024, y_log_2025, "Logística"),
        'pot': (y_pot_2024, y_pot_2025, "Potência")
    }
    y1, y2, titulo = modelos[tipo]

    r2_2024, rmse_2024 = calcular_metricas(np.array(coluna_2024), y1)
    r2_2025, rmse_2025 = calcular_metricas(np.array(coluna_2025), y2)

    fig = go.Figure()

    # Scatter com hover
    fig.add_trace(go.Scatter(x=x, y=coluna_2024, mode='markers', name='2024',
                             marker=dict(color='#00BFFF', size=6, opacity=0.8),
                             hovertemplate='Hora: %{x}<br>Temp: %{y:.2f}°C'))
    fig.add_trace(go.Scatter(x=x, y=coluna_2025, mode='markers', name='2025',
                             marker=dict(color='#FF6347', size=6, opacity=0.8),
                             hovertemplate='Hora: %{x}<br>Temp: %{y:.2f}°C'))

    # Linhas de ajuste com hover
    fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='Ajuste 2024',
                             line=dict(color='#00BFFF', width=2.5),
                             hovertemplate='Hora: %{x}<br>Ajuste: %{y:.2f}°C'))
    fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name='Ajuste 2025',
                             line=dict(color='#FF6347', width=2.5),
                             hovertemplate='Hora: %{x}<br>Ajuste: %{y:.2f}°C'))

    fig.update_layout(
        title=dict(
            text=f"Regressão {titulo}<br><sup style='color:#AAA'>"
                 f"R² (2024): {r2_2024:.4f} | RMSE (2024): {rmse_2024:.3f} &nbsp; "
                 f"R² (2025): {r2_2025:.4f} | RMSE (2025): {rmse_2025:.3f}</sup>",
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
    return fig


@app.callback(
    Output("imagem-teoria", "children"),
    Input("dropdown-teorias", "value")
)
def mostrar_imagem(teoria):
    if teoria is None:
        return html.P("Selecione uma teoria para visualizar.", style={"color": "#bbb", "fontSize": "18px"})
    caminho = imagens_teorias.get(teoria)
    if caminho is None:
        return html.P("Imagem não encontrada.", style={"color": "#bbb", "fontSize": "18px"})
    return html.Img(src=caminho, style={
        "width": "70%",
        "borderRadius": "12px",
        "boxShadow": "0 0 15px rgba(255,255,255,0.2)",
        "marginTop": "20px"
    })

# ============================
# RUN
# ============================

if __name__ == '__main__':
    app.run(debug=True)
