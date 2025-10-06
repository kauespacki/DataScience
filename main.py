import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
import dash_bootstrap_components as dbc

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

# ============================
# REGRESSÕES
# ============================
x = np.arange(len(coluna_2024))

# Linear
coef_lin_2024 = np.polyfit(x, coluna_2024, 1)
coef_lin_2025 = np.polyfit(x, coluna_2025, 1)
y_lin_2024 = np.polyval(coef_lin_2024, x)
y_lin_2025 = np.polyval(coef_lin_2025, x)

# Não linear (polinomial de grau 3)
coef_poly_2024 = np.polyfit(x, coluna_2024, 3)
coef_poly_2025 = np.polyfit(x, coluna_2025, 3)
y_poly_2024 = np.polyval(coef_poly_2024, x)
y_poly_2025 = np.polyval(coef_poly_2025, x)

# ============================
# IMAGENS DAS TEORIAS
# ============================
imagens_teorias = {
    "Teorema Central do Limite": "assets/teorema.jpg",
    "Correlação": "assets/correlacao.jpg",
    "Amostragem, Distribuição Normal (Curva de Gauss ou Poisson)": "assets/amostragem_normal.png",
    "T-Student": "assets/t_student.png",
    "Qui-quadrado": "assets/qui_quadrado.png"
}

# ============================
# DASH APP
# ============================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "Clima em Curitiba"

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(
            html.H1(
                "Clima em Curitiba",
                className="text-center mb-4 mt-2",
                style={"color": "white"}
            )
        )
    ]),

    dbc.Row([
        dbc.Col([
            html.Label("Tipo de Regressão", style={"color": "white"}),
            dcc.Dropdown(
                id='tipo-regressao',
                options=[
                    {'label': 'Regressão Linear', 'value': 'linear'},
                    {'label': 'Regressão Não Linear (Polinomial)', 'value': 'nao_linear'}
                ],
                value='linear',
                clearable=False,
                className="mb-4",
                style={'color': '#000'}
            ),
            dcc.Graph(id='grafico-regressao', style={'height': '75vh'})
        ])
    ]),

    html.Hr(style={"borderColor": "#444"}),

    # Dropdown de teorias
    dbc.Row([
        dbc.Col([
            html.Label("Selecione uma teoria estatística:", style={"color": "white", "fontSize": "18px"}),
            dcc.Dropdown(
                id='dropdown-teorias',
                options=[{'label': k, 'value': k} for k in imagens_teorias.keys()],
                placeholder="Escolha uma teoria...",
                style={'color': '#000'},
                className="mb-4"
            ),
            html.Div(id="imagem-teoria", className="text-center")
        ], width=8)
    ], justify="center")
], fluid=True, style={"backgroundColor": "#121212", "paddingBottom": "50px"})

# ============================
# CALLBACKS
# ============================

@app.callback(
    Output('grafico-regressao', 'figure'),
    Input('tipo-regressao', 'value')
)
def atualizar_grafico(tipo):
    if tipo == 'linear':
        y1, y2 = y_lin_2024, y_lin_2025
        titulo = 'Regressão Linear das Temperaturas'
    else:
        y1, y2 = y_poly_2024, y_poly_2025
        titulo = 'Regressão Não Linear (Polinomial) das Temperaturas'

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=coluna_2024, mode='lines', name='Temperaturas 2024', line=dict(color='skyblue')))
    fig.add_trace(go.Scatter(y=coluna_2025, mode='lines', name='Temperaturas 2025', line=dict(color='lightcoral')))
    fig.add_trace(go.Scatter(y=y1, mode='lines', name='Regressão 2024', line=dict(dash='dash', color='deepskyblue')))
    fig.add_trace(go.Scatter(y=y2, mode='lines', name='Regressão 2025', line=dict(dash='dash', color='tomato')))

    fig.update_layout(
        title=titulo,
        xaxis_title='Horas (0-167)',
        yaxis_title='Temperatura (°C)',
        legend_title='Ano',
        template='plotly_dark',
        paper_bgcolor='#111',
        plot_bgcolor='#111'
    )
    return fig


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
