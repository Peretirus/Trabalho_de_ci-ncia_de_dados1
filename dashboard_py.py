import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Configuração da página
st.set_page_config(
    page_title='Dashboard de Preços de Casas', 
    layout='wide',
    page_icon='🏠'
)
st.title('🏡 Dashboard Interativo - Análise de Preços de Imóveis (Ames Housing)')

# Carregar dados
@st.cache_data
def carregar_dados():
    return pd.read_csv('train.csv')

df = carregar_dados()

# Sidebar com filtros
st.sidebar.header('Filtros Principais')

# Filtro por Bairro
bairros = df['Neighborhood'].unique()
bairro_escolhido = st.sidebar.selectbox(
    'Selecione um Bairro:', 
    sorted(bairros),
    index=0
)

# Filtros Avançados
st.sidebar.header('Filtros Avançados')

# Filtro por Faixa de Preço
min_price = int(df['SalePrice'].min())
max_price = int(df['SalePrice'].max())
price_range = st.sidebar.slider(
    'Faixa de Preço:',
    min_value=min_price,
    max_value=max_price,
    value=(min_price, max_price)
)

# Filtro por Ano de Construção
min_year = int(df['YearBuilt'].min())
max_year = int(df['YearBuilt'].max())
year_range = st.sidebar.slider(
    'Ano de Construção:',
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

# Filtro por Qualidade Geral
qualidade = st.sidebar.select_slider(
    'Qualidade Geral (1-10):',
    options=sorted(df['OverallQual'].unique()),
    value=(1, 10)
)

# Filtro por Área Habitável
min_area = int(df['GrLivArea'].min())
max_area = int(df['GrLivArea'].max())
area_range = st.sidebar.slider(
    'Área Habitável (sqft):',
    min_value=min_area,
    max_value=max_area,
    value=(min_area, max_area)
)

# Aplicar filtros ao dataframe
df_filtrado = df[
    (df['SalePrice'] >= price_range[0]) & 
    (df['SalePrice'] <= price_range[1]) &
    (df['YearBuilt'] >= year_range[0]) & 
    (df['YearBuilt'] <= year_range[1]) &
    (df['OverallQual'] >= qualidade[0]) & 
    (df['OverallQual'] <= qualidade[1]) &
    (df['GrLivArea'] >= area_range[0]) & 
    (df['GrLivArea'] <= area_range[1])
]

# Atualizar o dataframe do bairro com os filtros
df_bairro = df_filtrado[df_filtrado['Neighborhood'] == bairro_escolhido]

# Layout com abas
tab1, tab2, tab3 = st.tabs(["📊 Visualizações Principais", "🗺️ Mapa Interativo", "📈 Análises Avançadas"])

with tab1:
    # Container para os primeiros gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        # 1. Distribuição de Preço no Bairro
        st.subheader(f'Distribuição de Preços em {bairro_escolhido}')
        fig1 = px.histogram(
            df_bairro, 
            x='SalePrice', 
            nbins=30,
            marginal='box',
            title='Distribuição dos Preços de Venda',
            hover_data=['GrLivArea', 'BedroomAbvGr', 'YearBuilt']
        )
        fig1.update_traces(
            marker_line_width=1, 
            marker_line_color="white",
            opacity=0.7
        )
        fig1.update_layout(
            hovermode='x unified',
            showlegend=False
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # 2. Boxplot por Qualidade Geral
        st.subheader('Preço por Qualidade Geral')
        fig2 = px.box(
            df_filtrado, 
            x='OverallQual', 
            y='SalePrice', 
            points='all',
            hover_data=['Neighborhood', 'GrLivArea', 'YearBuilt'],
            title='Relação entre Qualidade e Preço'
        )
        fig2.update_traces(
            boxmean=True,
            jitter=0.3,
            marker=dict(size=4, opacity=0.5)
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # 3. Scatter Plot Interativo
    st.subheader('Relação Área x Preço')
    
    # Controles interativos para o scatter plot
    col3, col4, col5 = st.columns(3)
    with col3:
        color_by = st.selectbox(
            'Colorir por:', 
            ['OverallQual', 'OverallCond', 'BedroomAbvGr', 'FullBath', 'Neighborhood']
        )
    with col4:
        size_by = st.selectbox(
            'Tamanho por:', 
            [None, 'TotalBsmtSF', 'GarageArea', 'LotArea', 'GrLivArea']
        )
    with col5:
        trendline = st.selectbox(
            'Linha de tendência:', 
            [None, 'ols', 'lowess']
        )
    
    fig3 = px.scatter(
        df_filtrado, 
        x='GrLivArea', 
        y='SalePrice', 
        color=color_by,
        size=size_by,
        trendline=trendline,
        hover_data=['Neighborhood', 'YearBuilt', 'SaleCondition', 'OverallQual'],
        title='Relação Área Habitável x Preço de Venda',
        labels={'GrLivArea': 'Área Habitável (sqft)', 'SalePrice': 'Preço de Venda ($)'}
    )
    
    if trendline:
        fig3.update_traces(
            selector=dict(type='scatter'),
            marker=dict(opacity=0.5)
        )
    
    st.plotly_chart(fig3, use_container_width=True)

with tab2:
    # Mapa Interativo (se houver coordenadas)
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        st.subheader('Mapa de Distribuição de Preços')
        
        # Opções de visualização do mapa
        map_type = st.radio(
            "Tipo de Mapa:",
            ["Mapa de Calor", "Pontos Interativos"],
            horizontal=True
        )
        
        if map_type == "Mapa de Calor":
            fig_map = px.density_mapbox(
                df_filtrado, 
                lat='Latitude', 
                lon='Longitude', 
                z='SalePrice', 
                radius=12,
                center=dict(
                    lat=df_filtrado['Latitude'].mean(), 
                    lon=df_filtrado['Longitude'].mean()
                ),
                zoom=11,
                mapbox_style="open-street-map",
                hover_data=['Neighborhood', 'GrLivArea', 'YearBuilt'],
                title='Mapa de Calor de Preços por Localização'
            )
        else:
            fig_map = px.scatter_mapbox(
                df_filtrado,
                lat='Latitude',
                lon='Longitude',
                color='SalePrice',
                size='GrLivArea',
                hover_name='Neighborhood',
                hover_data=['YearBuilt', 'OverallQual', 'SaleCondition'],
                zoom=11,
                mapbox_style="open-street-map",
                color_continuous_scale=px.colors.cyclical.IceFire,
                title='Distribuição Geográfica dos Imóveis'
            )
        
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning("Dados de latitude/longitude não encontrados no dataset.")

with tab3:
    # Análises Avançadas
    st.subheader('Análises Avançadas e Comparativas')
    
    # Gráfico com seleção cruzada
    st.markdown("**Seleção Cruzada entre Gráficos**")
    
    fig_combined = make_subplots(rows=1, cols=2)
    
    # Primeiro gráfico - Scatter plot
    fig_combined.add_trace(
        go.Scatter(
            x=df_filtrado['GrLivArea'], 
            y=df_filtrado['SalePrice'],
            mode='markers',
            name='Área x Preço',
            marker=dict(
                color=df_filtrado['OverallQual'],
                colorscale='Viridis',
                showscale=True,
                size=8,
                opacity=0.7
            ),
            customdata=df_filtrado[['Neighborhood', 'YearBuilt', 'OverallQual']],
            hovertemplate=(
                "<b>Área:</b> %{x} sqft<br>"
                "<b>Preço:</b> %{y:$,.0f}<br>"
                "<b>Bairro:</b> %{customdata[0]}<br>"
                "<b>Ano:</b> %{customdata[1]}<br>"
                "<b>Qualidade:</b> %{customdata[2]}"
            )
        ),
        row=1, col=1
    )
    
    # Segundo gráfico - Boxplot
    fig_combined.add_trace(
        go.Box(
            x=df_filtrado['OverallQual'], 
            y=df_filtrado['SalePrice'],
            name='Qualidade x Preço',
            boxpoints='all',
            jitter=0.3,
            marker=dict(size=4, opacity=0.5)
        ),
        row=1, col=2
    )
    
    # Configurações de layout
    fig_combined.update_layout(
        title_text="Análise Combinada - Seleção Cruzada",
        hovermode='closest',
        height=500,
        showlegend=False
    )
    
    # Configurar eixos
    fig_combined.update_xaxes(title_text="Área Habitável (sqft)", row=1, col=1)
    fig_combined.update_yaxes(title_text="Preço de Venda ($)", row=1, col=1)
    fig_combined.update_xaxes(title_text="Qualidade Geral", row=1, col=2)
    fig_combined.update_yaxes(title_text="Preço de Venda ($)", row=1, col=2)
    
    st.plotly_chart(fig_combined, use_container_width=True)
    
    # Tabela interativa com dados selecionados
    st.subheader('Explorar Dados Detalhados')
    
    col6, col7 = st.columns(2)
    with col6:
        selected_neighborhoods = st.multiselect(
            'Selecione bairros:',
            options=df_filtrado['Neighborhood'].unique()
        )
    with col7:
        min_bedrooms = st.slider(
            'Mínimo de quartos:',
            min_value=0,
            max_value=int(df['BedroomAbvGr'].max()),
            value=0
        )
    
    if selected_neighborhoods or min_bedrooms > 0:
        filtered_data = df_filtrado.copy()
        if selected_neighborhoods:
            filtered_data = filtered_data[filtered_data['Neighborhood'].isin(selected_neighborhoods)]
        if min_bedrooms > 0:
            filtered_data = filtered_data[filtered_data['BedroomAbvGr'] >= min_bedrooms]
        
        st.dataframe(
            filtered_data.sort_values('SalePrice', ascending=False)[
                ['Neighborhood', 'SalePrice', 'GrLivArea', 'BedroomAbvGr', 
                 'FullBath', 'YearBuilt', 'OverallQual']
            ].style.format({
                'SalePrice': '${:,.0f}',
                'GrLivArea': '{:,.0f} sqft'
            }).highlight_max(subset=['SalePrice'], color='lightgreen')
            .highlight_min(subset=['SalePrice'], color='#ffcccb'),
            use_container_width=True,
            height=400
        )
    else:
        st.info("Selecione bairros ou número mínimo de quartos para visualizar os dados detalhados.")

# Rodapé
st.markdown("---")
st.markdown("**Dashboard desenvolvido com Streamlit e Plotly** | Dados: Ames Housing Dataset")
