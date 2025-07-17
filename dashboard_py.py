import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Configuração da página
st.set_page_config(page_title="Dashboard - House Prices", layout="wide")

# Carregar dados
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    return df

df = load_data()

# Título
st.title("🏠 Análise de Preços de Casas - Ames Housing Dataset")

# Abas
tab1, tab2, tab3, tab4 = st.tabs(["📊 Visão Geral", "📌 Filtros", "📉 Correlações", "🤖 Previsão (ML)"])

# Aba 1 - Visão Geral
with tab1:
    st.header("📊 Visão Geral dos Dados")
    st.dataframe(df.head())

    st.markdown("### Distribuição dos Preços de Venda")
    fig = px.histogram(df, x="SalePrice", nbins=50, title="Distribuição de Preços")
    st.plotly_chart(fig, use_container_width=True)

# Aba 2 - Filtros Interativos
with tab2:
    st.header("📌 Filtros Interativos")

    bairro = st.selectbox("Selecione o Bairro", df["Neighborhood"].unique())
    df_bairro = df[df["Neighborhood"] == bairro]

    st.markdown(f"### Distribuição de Preços no Bairro {bairro}")
    fig2 = px.histogram(df_bairro, x="SalePrice", nbins=30)
    st.plotly_chart(fig2, use_container_width=True)

# Aba 3 - Correlações
with tab3:
    st.header("📉 Correlação entre Variáveis")

    variaveis = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt", "SalePrice"]
    df_corr = df[variaveis].dropna()

    corr = df_corr.corr(numeric_only=True)

    fig3 = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", origin="lower")
    st.plotly_chart(fig3, use_container_width=True)

# Aba 4 - Regressão Linear
with tab4:
    st.header("📈 Previsão de Preço com Regressão Linear")

    features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
    df_model = df[features + ['SalePrice']].dropna()

    X = df_model[features]
    y = df_model['SalePrice']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    st.markdown(f"Erro médio absoluto do modelo (MAE): **${mae:,.2f}**")

    st.subheader("🔍 Faça sua própria previsão")

    input_data = {}
    for feature in features:
        valor = st.number_input(
            f"{feature}",
            float(df_model[feature].min()),
            float(df_model[feature].max()),
            float(df_model[feature].mean())
        )
        input_data[feature] = valor

    input_df = pd.DataFrame([input_data])
    pred = model.predict(input_df)[0]
    st.success(f"📊 Preço de venda previsto: **${pred:,.2f}**")

# Rodapé
st.markdown("---")
st.markdown("Desenvolvido por Petrus César - Projeto de Ciência de Dados")
