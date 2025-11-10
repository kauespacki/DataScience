import dash
from dash import dcc, html, Input, Output, Dash
import plotly.graph_objs as go
import numpy as np
import dash_bootstrap_components as dbc
from scipy.optimize import curve_fit
from statistics import multimode, StatisticsError

# Tenta importar emcee, mas define uma flag se não estiver disponível
try:
    import emcee
    EMCEE_AVAILABLE = True
except ImportError:
    EMCEE_AVAILABLE = False
    print("Biblioteca 'emcee' não encontrada. O modo Bayesiano MCMC Real será desativado (usará simulação).")

# --- Novas importações para Aprendizado de Máquina ---
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from scipy.cluster.hierarchy import linkage
import plotly.figure_factory as ff

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

# Garante que os dados tenham o mesmo comprimento (truncando o mais longo)
min_len = min(len(coluna_2024), len(coluna_2025))
x = np.arange(min_len)
dados_2024 = np.array(coluna_2024[:min_len])
dados_2025 = np.array(coluna_2025[:min_len])


# ============================
# CÁLCULOS ESTATÍSTICOS
# ============================

# --- Univariadas ---
# Trata modas múltiplas
try:
    moda_2024_list = multimode(dados_2024)
    moda_2024_str = ', '.join(map(str, moda_2024_list))
    moda_2024_val = moda_2024_list[0] # Pega o primeiro valor para o gráfico
except StatisticsError:
    moda_2024_str = "N/A"
    moda_2024_val = np.nan

try:
    moda_2025_list = multimode(dados_2025)
    moda_2025_str = ', '.join(map(str, moda_2025_list))
    moda_2025_val = moda_2025_list[0]
except StatisticsError:
    moda_2025_str = "N/A"
    moda_2025_val = np.nan

stats_2024 = {
    "media": np.mean(dados_2024),
    "mediana": np.median(dados_2024),
    "moda_str": moda_2024_str,
    "moda_val": moda_2024_val,
    "std": np.std(dados_2024),
    "var": np.var(dados_2024)
}

stats_2025 = {
    "media": np.mean(dados_2025),
    "mediana": np.median(dados_2025),
    "moda_str": moda_2025_str,
    "moda_val": moda_2025_val,
    "std": np.std(dados_2025),
    "var": np.var(dados_2025)
}

# --- Bivariadas (Correlação e Covariância) ---
correlacao = np.corrcoef(dados_2024, dados_2025)[0, 1]
covariancia = np.cov(dados_2024, dados_2025)[0, 1]


# Função para criar cartão de estatísticas
def criar_cartao_stat(ano, stats):
    return dbc.Card(
        dbc.CardBody([
            html.H5(f"Estatísticas {ano}", className="card-title text-info"),
            html.P(f"Média: {stats['media']:.2f}°C"),
            html.P(f"Mediana: {stats['mediana']:.2f}°C"),
            html.P(f"Moda: {stats['moda_str']}°C"),
            html.P(f"Desvio Padrão: {stats['std']:.2f}°C"),
            html.P(f"Variância: {stats['var']:.2f}°C²"),
        ]),
        color="dark",
        outline=True,
        className="mb-3"
    )

# Cartão para Estatísticas Bivariadas
cartao_bivariada = dbc.Card(
    dbc.CardBody([
        html.H5("Estatísticas Bivariadas", className="card-title text-success"),
        html.P(f"Correlação (2024 vs 2025): {correlacao:.4f}"),
        html.P(f"Covariância (2024 vs 2025): {covariancia:.2f}°C²"),
    ]),
    color="dark",
    outline=True,
    className="mb-3"
)


# Criação da Figura do Gráfico de Barras (Atualizado com Moda)
nomes_stats = ['Média', 'Mediana', 'Moda', 'Desvio Padrão', 'Variância']
valores_2024 = [stats_2024['media'], stats_2024['mediana'], stats_2024['moda_val'], stats_2024['std'], stats_2024['var']]
valores_2025 = [stats_2025['media'], stats_2025['mediana'], stats_2025['moda_val'], stats_2025['std'], stats_2025['var']]

fig_stats = go.Figure()
fig_stats.add_trace(go.Bar(
    x=nomes_stats,
    y=valores_2024,
    name='2024',
    marker_color='#00BFFF',
    hovertemplate='2024 - %{x}: %{y:.2f}'
))
fig_stats.add_trace(go.Bar(
    x=nomes_stats,
    y=valores_2025,
    name='2025',
    marker_color='#FF6347',
    hovertemplate='2025 - %{x}: %{y:.2f}'
))

fig_stats.update_layout(
    title='Comparativo de Estatísticas (2024 vs 2025)',
    barmode='group',
    xaxis_tickangle=-45,
    paper_bgcolor='#1E1E1E',
    plot_bgcolor='#1E1E1E',
    font=dict(color='white'),
    legend=dict(bgcolor='rgba(0,0,0,0.3)', bordercolor='#444', borderwidth=1)
)


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
        try:
            JtJ = J.T @ J
            JtRes = J.T @ residuals
            delta = np.linalg.solve(JtJ, JtRes)
        except np.linalg.LinAlgError:
            return params # Retorna os últimos parâmetros válidos

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
        return np.full_like(y, np.nan)


# ============================
# FUNÇÕES PARA MCMC BAYESIANO (emcee)
# ============================

if EMCEE_AVAILABLE:
    def log_prior(params):
        """Define a probabilidade prévia dos parâmetros (priors)."""
        a, b, c = params
        if 0.0 < a < 50.0 and -0.01 < b < 0.01 and 0.0 < c < 50.0:
            return 0.0
        return -np.inf


    def log_likelihood(params, x, y_obs, y_err):
        """Define a verossimilhança (likelihood)."""
        a, b, c = params
        y_model = exponencial(x, a, b, c)
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
    """
    try:
        popt, _ = curve_fit(modelo, x, y, p0=p0, method=method, maxfev=5000)
        return modelo(x, *popt)
    except (RuntimeError, TypeError, ValueError):
        return np.full_like(y, np.nan)


# Ajustes pré-calculados (não exponenciais)
y_linear_2024 = np.polyval(np.polyfit(x, dados_2024, 1), x)
y_linear_2025 = np.polyval(np.polyfit(x, dados_2025, 1), x)

y_parab_2024 = ajustar_modelo(parabola, x, dados_2024)
y_parab_2025 = ajustar_modelo(parabola, x, dados_2025)

y_log_2024 = ajustar_modelo(logistica, x, dados_2024, p0=(max(dados_2024), 0.05, len(x) / 2))
y_log_2025 = ajustar_modelo(logistica, x, dados_2025, p0=(max(dados_2025), 0.05, len(x) / 2))

y_pot_2024 = ajustar_modelo(potencia, x, dados_2024, p0=(1, 0.01))
y_pot_2025 = ajustar_modelo(potencia, x, dados_2025, p0=(1, 0.01))


# ============================
# MÉTRICAS DE REGRESSÃO
# ============================

def calcular_metricas_regressao(y_real, y_pred):
    mask = ~np.isnan(y_pred)
    if mask.sum() == 0:
        return np.nan, np.nan
    y_real_masked = y_real[mask]
    y_pred_masked = y_pred[mask]

    if len(y_real_masked) == 0:
        return np.nan, np.nan

    ss_res = np.sum((y_real_masked - y_pred_masked) ** 2)
    ss_tot = np.sum((y_real_masked - np.mean(y_real_masked)) ** 2)

    if ss_tot == 0:
        return np.nan, np.nan

    r2 = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(ss_res / len(y_real_masked))
    return r2, rmse


# ============================
# CÁLCULOS DE MACHINE LEARNING (NÃO SUPERVISIONADO)
# ============================

# Prepara os dados para Clusterização
# Combina os dados de 2024 e 2025 para encontrar padrões gerais
X_cluster = np.concatenate([
    np.column_stack([x, dados_2024]),
    np.column_stack([x, dados_2025])
])

# Padronização (Crucial para K-Means, GMM e Hierárquico)
scaler = StandardScaler()
X_kmeans_scaled = scaler.fit_transform(X_cluster)

# Define K=4 
n_clusters = 4

# --- 1. K-Means ---
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_kmeans_scaled)
kmeans_inertia = kmeans.inertia_
kmeans_silhouette = silhouette_score(X_kmeans_scaled, kmeans_labels)

# --- 2. Hierarchical Clustering ---
agglo = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
agglo_labels = agglo.fit_predict(X_kmeans_scaled)
agglo_silhouette = silhouette_score(X_kmeans_scaled, agglo_labels)

# --- 3. Gaussian Mixture Models (GMM) ---
gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
gmm_labels = gmm.fit_predict(X_kmeans_scaled)
gmm_silhouette = silhouette_score(X_kmeans_scaled, gmm_labels)
gmm_bic = gmm.bic(X_kmeans_scaled) # Critério de Informação Bayesiano

# --- Figura K-Means ---
fig_kmeans = go.Figure(data=[go.Scatter(
    x=X_cluster[:, 0], y=X_cluster[:, 1], mode='markers',
    marker=dict(color=kmeans_labels, colorscale='Viridis', showscale=False),
    hovertemplate='Hora: %{x}<br>Temp: %{y:.2f}°C<br>Cluster: %{marker.color}'
)])
fig_kmeans.update_layout(
    title=f"K-Means (K=4)<br><sup>Inércia: {kmeans_inertia:.2f} | Silhueta: {kmeans_silhouette:.2f}</sup>",
    xaxis_title="Hora", yaxis_title="Temperatura (°C)",
    paper_bgcolor='#1E1E1E', plot_bgcolor='#1E1E1E', font=dict(color='white')
)

# --- Figura Hierarchical ---
fig_hierarchical = go.Figure(data=[go.Scatter(
    x=X_cluster[:, 0], y=X_cluster[:, 1], mode='markers',
    marker=dict(color=agglo_labels, colorscale='Viridis', showscale=False),
    hovertemplate='Hora: %{x}<br>Temp: %{y:.2f}°C<br>Cluster: %{marker.color}'
)])
fig_hierarchical.update_layout(
    title=f"Hierarchical (K=4)<br><sup>Silhueta: {agglo_silhouette:.2f}</sup>",
    xaxis_title="Hora", yaxis_title="Temperatura (°C)",
    paper_bgcolor='#1E1E1E', plot_bgcolor='#1E1E1E', font=dict(color='white')
)

# --- Figura GMM ---
fig_gmm = go.Figure(data=[go.Scatter(
    x=X_cluster[:, 0], y=X_cluster[:, 1], mode='markers',
    marker=dict(color=gmm_labels, colorscale='Viridis', showscale=False),
    hovertemplate='Hora: %{x}<br>Temp: %{y:.2f}°C<br>Cluster: %{marker.color}'
)])
fig_gmm.update_layout(
    title=f"Gaussian Mixture (K=4)<br><sup>BIC: {gmm_bic:.2f} | Silhueta: {gmm_silhouette:.2f}</sup>",
    xaxis_title="Hora", yaxis_title="Temperatura (°C)",
    paper_bgcolor='#1E1E1E', plot_bgcolor='#1E1E1E', font=dict(color='white')
)

# --- Figura Dendrograma ---
try:
    linkage_matrix = linkage(X_kmeans_scaled, method='ward')
    fig_dendrogram = ff.create_dendrogram(
        X_kmeans_scaled,
        linkagefun=lambda x: linkage(x, 'ward'),
        color_threshold=10 # Ajusta o 'corte' de cor
    )
    fig_dendrogram.update_layout(
        title="Dendrograma (Hierarchical)",
        paper_bgcolor='#1E1E1E', plot_bgcolor='#1E1E1E', font=dict(color='white'),
        yaxis=dict(gridcolor='#333'), xaxis=dict(gridcolor='#333')
    )
    fig_dendrogram.update_xaxes(showticklabels=False) # Limpa o eixo X
except ImportError:
    # Fallback se scipy.cluster.hierarchy falhar
    fig_dendrogram = go.Figure().update_layout(title="Dendrograma (Erro ao gerar)", paper_bgcolor='#1E1E1E', plot_bgcolor='#1E1E1E', font=dict(color='white'))


# --- Figura Comparação de Métricas (Cluster) ---
metricas_nomes = ['Silhueta (K-Means)', 'Silhueta (Hierárquico)', 'Silhueta (GMM)']
metricas_valores = [kmeans_silhouette, agglo_silhouette, gmm_silhouette]
cores_metricas = ['#00BFFF', '#FF6347', '#32CD32']

fig_metricas_cluster = go.Figure(data=[go.Bar(
    x=metricas_nomes,
    y=metricas_valores,
    marker_color=cores_metricas,
    hovertemplate='%{x}: %{y:.3f}'
)])
fig_metricas_cluster.update_layout(
    title='Comparativo de Qualidade (Coef. de Silhueta)',
    yaxis_title='Pontuação (Quanto maior, melhor)',
    paper_bgcolor='#1E1E1E', plot_bgcolor='#1E1E1E', font=dict(color='white'),
    yaxis=dict(gridcolor='#333')
)

# ============================
# CÁLCULOS DE MACHINE LEARNING (SUPERVISIONADO)
# ============================

# --- Prepara dados ---
# Prepara dados: X é a hora (feature), y é a temperatura (target)
# Usamos apenas os dados de 2024 para treinar e testar
X_super = x.reshape(-1, 1) # Feature (Hora)
y_super = dados_2024 # Target (Temperatura)

# Divide em Treino (70%) e Teste (30%)
X_train, X_test, y_train, y_test = train_test_split(X_super, y_super, test_size=0.3, random_state=42)

# --- CORREÇÃO v3: Padronização (Scaling) para X e Y ---
# Redes Neurais precisam de dados de entrada (X) E saída (Y) padronizados
scaler_X_super = StandardScaler()
X_train_scaled = scaler_X_super.fit_transform(X_train)
X_test_scaled = scaler_X_super.transform(X_test)

# Novo scaler para Y (Target)
scaler_y_super = StandardScaler()
# y_train precisa ser (n_samples, 1) para o scaler
y_train_scaled = scaler_y_super.fit_transform(y_train.reshape(-1, 1))
# --- FIM DA CORREÇÃO ---


# Faixa de X para plotar linhas de previsão
X_range = np.arange(min(x), max(x) + 1).reshape(-1, 1)
# Padroniza o X_range para a previsão do MLP
X_range_scaled = scaler_X_super.transform(X_range)


# --- 1. KNN (K-Nearest Neighbors) ---
best_k = 1
best_rmse = float('inf')
k_options = range(1, 15)

for k in k_options:
    knn_model_k = KNeighborsRegressor(n_neighbors=k)
    knn_model_k.fit(X_train, y_train)
    y_pred_k = knn_model_k.predict(X_test)
    rmse_k = np.sqrt(mean_squared_error(y_test, y_pred_k))
    if rmse_k < best_rmse:
        best_rmse = rmse_k
        best_k = k

# Modelos baseados em árvore (KNN, RF, DT) não precisam de dados padronizados
knn_final = KNeighborsRegressor(n_neighbors=best_k)
knn_final.fit(X_train, y_train)
y_pred_knn_line = knn_final.predict(X_range)
y_pred_knn_test = knn_final.predict(X_test)
knn_rmse = np.sqrt(mean_squared_error(y_test, y_pred_knn_test))
knn_r2 = r2_score(y_test, y_pred_knn_test)

# --- 2. Random Forest ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf_line = rf_model.predict(X_range)
y_pred_rf_test = rf_model.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf_test))
rf_r2 = r2_score(y_test, y_pred_rf_test)

# --- 3. Decision Tree ---
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt_line = dt_model.predict(X_range)
y_pred_dt_test = dt_model.predict(X_test)
dt_rmse = np.sqrt(mean_squared_error(y_test, y_pred_dt_test))
dt_r2 = r2_score(y_test, y_pred_dt_test)

# --- 4. Neural Network (MLP) ---
# --- CORREÇÃO v3: Treina com Y_scaled e faz inverse_transform ---
mlp_model = MLPRegressor(hidden_layer_sizes=(10, 5), max_iter=2000, random_state=42, activation='relu', solver='adam')
# Treina com X_scaled e Y_scaled. y_scaled precisa de .ravel() para o fit
mlp_model.fit(X_train_scaled, y_train_scaled.ravel()) 

# 1. Previsão da linha do gráfico (scaled)
y_pred_mlp_line_scaled = mlp_model.predict(X_range_scaled)
# 2. Previsão de Teste (scaled)
y_pred_mlp_test_scaled = mlp_model.predict(X_test_scaled)

# 3. Converter previsões de volta para °C (inverse_transform)
# .reshape(-1, 1) é necessário para o inverse_transform
y_pred_mlp_line = scaler_y_super.inverse_transform(y_pred_mlp_line_scaled.reshape(-1, 1))
y_pred_mlp_test = scaler_y_super.inverse_transform(y_pred_mlp_test_scaled.reshape(-1, 1))

# 4. Calcular métricas comparando os valores em °C (reais vs. invertidos)
mlp_rmse = np.sqrt(mean_squared_error(y_test, y_pred_mlp_test))
mlp_r2 = r2_score(y_test, y_pred_mlp_test)
# --- FIM DA CORREÇÃO ---


# --- Figura Base para Gráficos Supervisionados ---
def criar_grafico_supervisionado(title, line_data, rmse, r2):
    fig = go.Figure()
    # 1. Dados de Treino
    fig.add_trace(go.Scatter(
        x=X_train.ravel(), y=y_train, mode='markers', name='Dados de Treino',
        marker=dict(color='blue', opacity=0.4)
    ))
    # 2. Dados de Teste (Reais)
    fig.add_trace(go.Scatter(
        x=X_test.ravel(), y=y_test, mode='markers', name='Dados de Teste (Real)',
        marker=dict(color='red', size=8, symbol='x', opacity=0.7)
    ))
    # 3. Linha de Previsão
    fig.add_trace(go.Scatter(
        x=X_range.ravel(), y=line_data, mode='lines', name='Previsão do Modelo',
        line=dict(color='limegreen', width=3, dash='dash')
    ))
    fig.update_layout(
        title=f"{title}<br><sup>RMSE (Teste): {rmse:.2f}°C | R² (Teste): {r2:.3f}</sup>",
        xaxis_title="Hora", yaxis_title="Temperatura (°C)",
        paper_bgcolor='#1E1E1E', plot_bgcolor='#1E1E1E', font=dict(color='white'),
        legend=dict(bgcolor='rgba(0,0,0,0.3)', bordercolor='#444')
    )
    return fig

# --- Criar Figuras Supervisionadas ---
fig_knn = criar_grafico_supervisionado(f"KNN Regressor (K={best_k})", y_pred_knn_line, knn_rmse, knn_r2)
fig_rf = criar_grafico_supervisionado("Random Forest Regressor", y_pred_rf_line, rf_rmse, rf_r2)
fig_dt = criar_grafico_supervisionado("Decision Tree Regressor", y_pred_dt_line, dt_rmse, dt_r2)
fig_mlp = criar_grafico_supervisionado("Neural Network (MLP) Regressor", y_pred_mlp_line, mlp_rmse, mlp_r2)

# --- Figura Comparação de RMSE (Supervisionado) ---
modelos_super_nomes = ['Rede Neural (MLP)', 'Random Forest', 'KNN', 'Árvore de Decisão']
modelos_super_rmses = [mlp_rmse, rf_rmse, knn_rmse, dt_rmse]
modelos_super_cores = ['#32CD32', '#FF6347', '#00BFFF', '#FFD700'] # Verde, Vermelho, Azul, Amarelo

fig_comparacao_rmse = go.Figure(data=[go.Bar(
    x=modelos_super_nomes,
    y=modelos_super_rmses,
    marker_color=modelos_super_cores,
    hovertemplate='%{x}: %{y:.3f} °C'
)])
fig_comparacao_rmse.update_layout(
    title='Comparativo de Desempenho (RMSE)',
    yaxis_title='Erro Médio (RMSE) (Quanto menor, melhor)',
    paper_bgcolor='#1E1E1E', plot_bgcolor='#1E1E1E', font=dict(color='white'),
    yaxis=dict(gridcolor='#333')
)


# ============================
# DASH APP
# ============================

app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
app.title = "Análise de Temperaturas - Data Science View"

# Imagens das teorias (devem estar na pasta 'assets')
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
    
    dbc.Tabs([
        
        # --- ABA 1: ESTATÍSTICA DESCRITIVA ---
        dbc.Tab(label="Estatística Descritiva", children=[
            dbc.Row([
                dbc.Col(html.H4("Análise Descritiva Univariada e Bivariada", className="text-center text-light mt-4 mb-3"))
            ]),
            dbc.Row([
                # Coluna para os cartões
                dbc.Col([
                    criar_cartao_stat("2024", stats_2024),
                    criar_cartao_stat("2025", stats_2025),
                    cartao_bivariada # Adiciona o novo cartão
                ], md=4),

                # Coluna para o gráfico de barras
                dbc.Col([
                    dcc.Graph(id='grafico-estatisticas', figure=fig_stats)
                ], md=8),
            ], className="mb-4"),
            
            html.Hr(style={"borderColor": "#444", "marginTop": "30px"}),
            dbc.Row([
                dbc.Col(html.H4("Galeria de Teorias Estatísticas (Inferência)", className="text-center text-light mt-4 mb-3"))
            ]),
            html.Label("Selecione uma teoria estatística:", style={"color": "white", "fontSize": "18px"}),
            dcc.Dropdown(
                id='dropdown-teorias',
                options=[{'label': k, 'value': k} for k in imagens_teorias.keys()],
                placeholder="Escolha uma teoria...",
                style={'color': '#000'},
                className="mb-4"
            ),
            html.Div(id="imagem-teoria", className="text-center mb-4"),
            
        ], className="mt-3"),
        
        # --- ABA 2: REGRESSÃO (MODELAGEM) ---
        dbc.Tab(label="Regressão (Modelagem)", children=[
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
                    html.Div(id='opcoes-otimizacao-exp', children=[
                        html.Label("Método de Estimação (para Exponencial):", style={"color": "white", "fontSize": "16px"}),
                        dcc.Dropdown(
                            id='metodo-otimizacao',
                            options=[
                                {'label': 'Algoritmo de Levenberg-Marquardt (Padrão)', 'value': 'lm'},
                                {'label': 'Mínimos Quadrados Não Linear (via TRF)', 'value': 'trf'},
                                {'label': 'Máxima Verossimilhança (MLE, via Dogbox)', 'value': 'dogbox'},
                                {'label': 'Gauss-Newton (Puro)', 'value': 'gauss_newton'},
                                {'label': bayes_label, 'value': 'bayes', 'disabled': not EMCEE_AVAILABLE}
                            ],
                            value='lm',
                            clearable=False,
                            style={'color': '#000'},
                            className="mb-4"
                        )
                    ], style={'display': 'none'}),
                    dcc.Graph(id='grafico-regressao', style={'height': '65vh'}),
                ], md=12)
            ], className="mt-4"),
            dbc.Row([
                dbc.Col([
                    dcc.Markdown("""
                        **Análise de Risco e Otimização para Agritech (Problema)**

                        **Problema Concreto:** Uma startup de *agritech* precisa prever a variação da temperatura em Curitiba para otimizar o uso de climatizadores e irrigação em estufas urbanas. O objetivo é usar um modelo matemático para prever a temperatura máxima (pico de custo de energia) e a mínima (risco de resfriamento) ao longo do dia. O modelo mais prático é aquele que tiver o **menor Erro Médio (RMSE)**.

                        **Avaliação dos Modelos (Regressão):** Conforme a Seção 6.3, esta abordagem falha. Os modelos de regressão simples são inadequados, pois não capturam o padrão cíclico diário (dia/noite). Os valores de R² são muito baixos (máx 0.18) e o RMSE muito alto (acima de 3.6°C).
                    """, style={'color': '#ccc', 'backgroundColor': '#2a2a2a', 'padding': '15px', 'borderRadius': '8px', "marginTop": "20px"})
                ], md=12)
            ])
        ], className="mt-3"),
        
        # --- ABA 3: APRENDIZADO NÃO SUPERVISIONADO ---
        dbc.Tab(label="Aprendizado Não Supervisionado (Clusters)", children=[
            dbc.Row([
                dbc.Col(html.H4("Análise de Clusters (Seção 7)", className="text-center text-light mt-4 mb-3"))
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='grafico-kmeans', figure=fig_kmeans), md=6),
                dbc.Col(dcc.Graph(id='grafico-gmm', figure=fig_gmm), md=6),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(id='grafico-hierarchical', figure=fig_hierarchical), md=6),
                dbc.Col(dcc.Graph(id='grafico-dendrograma', figure=fig_dendrogram), md=6),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(id='grafico-metricas-cluster', figure=fig_metricas_cluster), md=12),
            ], className="mb-3")
        ], className="mt-3"),
        
        # --- ABA 4: APRENDIZADO SUPERVISIONADO ---
        dbc.Tab(label="Aprendizado Supervisionado (Previsão)", children=[
            dbc.Row([
                dbc.Col(html.H4("Previsão de Temperatura (Seção 8)", className="text-center text-light mt-4 mb-3"))
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='grafico-knn', figure=fig_knn), md=6),
                dbc.Col(dcc.Graph(id='grafico-rf', figure=fig_rf), md=6),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(id='grafico-dt', figure=fig_dt), md=6),
                dbc.Col(dcc.Graph(id='grafico-mlp', figure=fig_mlp), md=6),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(id='grafico-comparacao-rmse', figure=fig_comparacao_rmse), md=12),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col(dcc.Markdown("""
                    **Conclusão (Seção 8):** A abordagem de Aprendizado Supervisionado (Seção 8) foi a única que **resolveu o problema** da agritech. Ao contrário das regressões (Seção 6), estes modelos capturaram o padrão cíclico.
                    
                    O gráfico de comparação de RMSE mostra que a **Rede Neural (MLP)** teve o menor erro (RMSE: 0.99°C), seguida de perto pelo **Random Forest** (RMSE: 1.02°C). Ambos são soluções excelentes e muito superiores às regressões lineares/não lineares.
                """, style={'color': '#ccc', 'backgroundColor': '#2a2a2a', 'padding': '15px', 'borderRadius': '8px'}), md=12)
            ])
        ], className="mt-3"),

    ]) # Fim do dbc.Tabs

], fluid=True, style={"backgroundColor": "#1E1E1E", "paddingBottom": "40px"})


# ============================
# CALLBACKS
# ============================

# Callback para mostrar/ocultar o dropdown de otimização
@app.callback(
    Output('opcoes-otimizacao-exp', 'style'),
    Input('tipo-regressao', 'value')
)
def mostrar_opcoes_exp(tipo_regressao):
    if tipo_regressao == 'exp':
        return {'display': 'block'}
    return {'display': 'none'}


# Callback principal para o gráfico de REGRESSÃO
@app.callback(
    Output('grafico-regressao', 'figure'),
    [Input('tipo-regressao', 'value'),
     Input('metodo-otimizacao', 'value')]
)
def atualizar_grafico_regressao(tipo_regressao, metodo_opt):
    titulo_metodo = ""

    # Parâmetros iniciais (p0) melhorados
    p0_c_24 = np.min(dados_2024)
    p0_a_24 = dados_2024[0] - p0_c_24
    if p0_a_24 <= 0: p0_a_24 = 0.1
    p0_exp_24 = (p0_a_24, 0.001, p0_c_24)

    p0_c_25 = np.min(dados_2025)
    p0_a_25 = dados_2025[0] - p0_c_25
    if p0_a_25 <= 0: p0_a_25 = 0.1
    p0_exp_25 = (p0_a_25, 0.001, p0_c_25)

    fig = go.Figure()

    if tipo_regressao == 'exp':
        if metodo_opt == 'bayes':
            if EMCEE_AVAILABLE:
                # --- Implementação Real Bayesiana (LENTO) ---
                titulo = "Exponencial"
                titulo_metodo = " (Método: Bayesiano (MCMC Real))"
                try:
                    # MCMC 2024
                    y_err_24 = np.std(dados_2024) * 0.5
                    popt_base_24, _ = curve_fit(exponencial, x, dados_2024, p0=p0_exp_24, method='lm')
                    nwalkers, ndim = 32, 3
                    p0_walkers = popt_base_24 + 1e-4 * np.random.randn(nwalkers, ndim)
                    sampler24 = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, dados_2024, y_err_24))
                    sampler24.run_mcmc(p0_walkers, 500, progress=False, skip_initial_state_check=True)
                    samples_24 = sampler24.get_chain(discard=100, thin=10, flat=True)
                    all_y1 = [exponencial(x, *samples_24[idx]) for idx in np.random.randint(len(samples_24), size=100)]
                    y1 = np.mean(all_y1, axis=0)
                    for y_sample in all_y1:
                        fig.add_trace(go.Scatter(x=x, y=y_sample, mode='lines', line=dict(color='#00BFFF', width=0.5), opacity=0.1, showlegend=False, hoverinfo='none'))
                except Exception as e:
                    y1 = np.full_like(dados_2024, np.nan)

                try:
                    # MCMC 2025
                    y_err_25 = np.std(dados_2025) * 0.5
                    popt_base_25, _ = curve_fit(exponencial, x, dados_2025, p0=p0_exp_25, method='lm')
                    p0_walkers_25 = popt_base_25 + 1e-4 * np.random.randn(nwalkers, ndim)
                    sampler25 = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, dados_2025, y_err_25))
                    sampler25.run_mcmc(p0_walkers_25, 500, progress=False, skip_initial_state_check=True)
                    samples_25 = sampler25.get_chain(discard=100, thin=10, flat=True)
                    all_y2 = [exponencial(x, *samples_25[idx]) for idx in np.random.randint(len(samples_25), size=100)]
                    y2 = np.mean(all_y2, axis=0)
                    for y_sample in all_y2:
                        fig.add_trace(go.Scatter(x=x, y=y_sample, mode='lines', line=dict(color='#FF6347', width=0.5), opacity=0.1, showlegend=False, hoverinfo='none'))
                except Exception as e:
                    y2 = np.full_like(dados_2025, np.nan)
            
            else:
                # --- Fallback: Simulação Visual ---
                titulo = "Exponencial"
                titulo_metodo = " (Método: Bayesiano (Simulação))"
                try:
                    popt_base_24, _ = curve_fit(exponencial, x, dados_2024, p0=p0_exp_24, method='lm', maxfev=5000)
                    all_y1 = []
                    for _ in range(50):
                        p_sample = np.copy(popt_base_24) * (1 + np.random.normal(0, 0.05, 3))
                        p_sample[1] = popt_base_24[1] + np.random.normal(0, 0.002) # Ruído aditivo em 'b'
                        y_sample = exponencial(x, *p_sample)
                        all_y1.append(y_sample)
                        fig.add_trace(go.Scatter(x=x, y=y_sample, mode='lines', line=dict(color='#00BFFF', width=0.5), opacity=0.1, showlegend=False, hoverinfo='none'))
                    y1 = np.mean(all_y1, axis=0)
                except (RuntimeError, TypeError, ValueError):
                    y1 = np.full_like(dados_2024, np.nan)
                
                try:
                    popt_base_25, _ = curve_fit(exponencial, x, dados_2025, p0=p0_exp_25, method='lm', maxfev=5000)
                    all_y2 = []
                    for _ in range(50):
                        p_sample = np.copy(popt_base_25) * (1 + np.random.normal(0, 0.05, 3))
                        p_sample[1] = popt_base_25[1] + np.random.normal(0, 0.002)
                        y_sample = exponencial(x, *p_sample)
                        all_y2.append(y_sample)
                        fig.add_trace(go.Scatter(x=x, y=y_sample, mode='lines', line=dict(color='#FF6347', width=0.5), opacity=0.1, showlegend=False, hoverinfo='none'))
                    y2 = np.mean(all_y2, axis=0)
                except (RuntimeError, TypeError, ValueError):
                    y2 = np.full_like(dados_2025, np.nan)

        else:
            if metodo_opt == 'gauss_newton':
                y1 = ajustar_modelo_gn(x, dados_2024, p0=p0_exp_24)
                y2 = ajustar_modelo_gn(x, dados_2025, p0=p0_exp_25)
                titulo_metodo_str = 'Gauss-Newton (Puro)'
            else:
                y1 = ajustar_modelo(exponencial, x, dados_2024, p0=p0_exp_24, method=metodo_opt)
                y2 = ajustar_modelo(exponencial, x, dados_2025, p0=p0_exp_25, method=metodo_opt)
                metodo_map = {'lm': 'Levenberg-Marquardt', 'trf': 'NLS (via TRF)', 'dogbox': 'MLE (via Dogbox)'}
                titulo_metodo_str = metodo_map.get(metodo_opt, metodo_opt.upper())
            
            titulo = "Exponencial"
            titulo_metodo = f" (Método: {titulo_metodo_str})"

    else:
        modelos = {
            'linear': (y_linear_2024, y_linear_2025, "Linear"),
            'parab': (y_parab_2024, y_parab_2025, "Parabólica"),
            'log': (y_log_2024, y_log_2025, "Logística"),
            'pot': (y_pot_2024, y_pot_2025, "Potência")
        }
        y1, y2, titulo = modelos[tipo_regressao]

    # Calcula métricas
    r2_2024, rmse_2024 = calcular_metricas_regressao(dados_2024, y1)
    r2_2025, rmse_2025 = calcular_metricas_regressao(dados_2025, y2)

    # Adiciona os dados de scatter (pontos)
    fig.add_trace(go.Scatter(x=x, y=dados_2024, mode='markers', name='2024',
                             marker=dict(color='#00BFFF', size=6, opacity=0.8),
                             hovertemplate='Hora: %{x}<br>Temp: %{y:.2f}°C'))
    fig.add_trace(go.Scatter(x=x, y=dados_2025, mode='markers', name='2025',
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

    return fig


# Callback para a galeria de imagens
@app.callback(
    Output("imagem-teoria", "children"),
    Input("dropdown-teorias", "value")
)
def mostrar_imagem_teoria(teoria):
    if teoria is None:
        return html.P("Selecione uma teoria para visualizar.", style={"color": "#bbb", "fontSize": "18px", "paddingTop": "20px"})
    
    caminho = imagens_teorias.get(teoria, "assets/placeholder.jpg") # Usa um placeholder se não achar
    
    # Garante que a imagem seja carregada da pasta 'assets'
    try:
        asset_url = app.get_asset_url(caminho.replace("assets/", ""))
    except Exception:
         return html.P(f"Erro ao carregar a imagem: {caminho}. Verifique a pasta 'assets'.", style={"color": "red"})

    return html.Img(src=asset_url, style={
        "maxWidth": "70%",
        "height": "auto",
        "borderRadius": "12px",
        "boxShadow": "0 0 15px rgba(255,255,200,0.2)",
        "marginTop": "20px"
    })


# ============================
# RUN
# ============================

if __name__ == '__main__':
    app.run(debug=True, port=8051) # Alterada a porta para 8051