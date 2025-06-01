# streamlit_dashboard_anova.py
import streamlit as st
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import shapiro, levene, kruskal, anderson
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  # Adicionado para lidar com potenciais issues numéricas


# --- Funções de Análise (Adaptadas do script anterior) ---

@st.cache_data  # Cache para otimizar o carregamento de dados
def load_data():
    """Carrega o Ames Housing Dataset de uma URL e faz uma limpeza básica."""
    urls_tentativas = [
        "https://raw.githubusercontent.com/Viniciusalgueiro/Ameshousing/refs/heads/main/AmesHousing.csv"
    ]
    df = None
    url_carregada = ""
    for url in urls_tentativas:
        try:
            df = pd.read_csv(url)
            url_carregada = url
            break
        except Exception:
            continue  # Tenta a próxima URL

    if df is None:
        st.error("Não foi possível carregar o dataset de nenhuma das URLs conhecidas.")
        return None, None, [], []

    st.success(f"Dataset carregado com sucesso de: {url_carregada}")
    df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True).str.lower()

    coluna_preco_nome = None
    if 'saleprice' in df.columns:
        coluna_preco_nome = 'saleprice'
    elif 'sale_price' in df.columns:
        df.rename(columns={'sale_price': 'saleprice'}, inplace=True)
        coluna_preco_nome = 'saleprice'
    # Adicionar mais heurísticas se necessário

    if coluna_preco_nome:
        df[coluna_preco_nome] = pd.to_numeric(df[coluna_preco_nome], errors='coerce')
        df.dropna(subset=[coluna_preco_nome], inplace=True)

    colunas_categoricas_potenciais = df.select_dtypes(include=['object']).columns.tolist()
    colunas_numericas_discretas = [col for col in df.select_dtypes(include=np.number).columns
                                   if df[col].nunique() < 20 and col != coluna_preco_nome]  # Exemplo de heurística
    colunas_categoricas_potenciais.extend(colunas_numericas_discretas)

    # Remover duplicatas e garantir que a coluna de preço não está na lista
    colunas_categoricas_potenciais = sorted(
        list(set(col for col in colunas_categoricas_potenciais if col != coluna_preco_nome)))

    return df, coluna_preco_nome, colunas_categoricas_potenciais, df.columns.tolist()


def perform_anova_for_variable(df_analysis, var_cat, col_preco):
    """Executa ANOVA e testes de pressupostos para uma variável."""
    results = {"var_cat": var_cat, "plots": {}}

    df_var = df_analysis[[var_cat, col_preco]].copy()

    # Converter para categoria se não for e garantir que tem pelo menos 2 níveis
    if df_var[var_cat].dtype != 'object' and not pd.api.types.is_categorical_dtype(df_var[var_cat]):
        df_var[var_cat] = df_var[var_cat].astype('category')

    df_var.dropna(inplace=True)  # Remove NaNs especificamente para este par

    if df_var[var_cat].nunique() < 2 or len(df_var) < 10:  # Mínimo de observações e níveis
        results["error"] = "Dados insuficientes ou poucos níveis para análise após limpeza."
        return results

    formula = f'{col_preco} ~ C({var_cat})'
    try:
        modelo = ols(formula, data=df_var).fit()
        results["anova_table"] = sm.stats.anova_lm(modelo, typ=2)

        p_valor_anova = None
        if f'C({var_cat})' in results["anova_table"].index:
            p_valor_anova = results["anova_table"].loc[f'C({var_cat})', 'PR(>F)']
        elif not results["anova_table"].empty:
            p_valor_anova = results["anova_table"]['PR(>F)'].iloc[0]
        results["p_valor_anova"] = p_valor_anova

        residuos = modelo.resid
        results["residuos_count"] = len(residuos)

        # 1. Normalidade dos resíduos
        normalidade_ok = False
        if len(residuos) >= 3:
            if len(residuos) <= 5000:
                stat_shapiro, p_shapiro = shapiro(residuos)
                results["shapiro_test"] = (stat_shapiro, p_shapiro)
                if p_shapiro >= 0.05: normalidade_ok = True
            else:
                ad_result = anderson(residuos)
                results["anderson_test"] = ad_result
                # Verifica se a estatística é menor que o valor crítico para 5%
                sig_level_idx = ad_result.significance_level.tolist().index(5.0)
                if ad_result.statistic < ad_result.critical_values[sig_level_idx]:
                    normalidade_ok = True
        results["normalidade_ok"] = normalidade_ok

        # Plots de Normalidade
        fig_norm, ax_norm = plt.subplots(1, 2, figsize=(10, 4))
        if len(residuos) > 1:
            sns.histplot(residuos, kde=True, ax=ax_norm[0], stat="density", bins=30)
            ax_norm[0].set_title(f'Histograma Resíduos ({var_cat})', fontsize=10)
            sm.qqplot(residuos, line='s', ax=ax_norm[1], markerfacecolor="skyblue", markeredgecolor="dodgerblue",
                      alpha=0.7)
            ax_norm[1].set_title(f'Q-Q Plot Resíduos ({var_cat})', fontsize=10)
        else:
            ax_norm[0].text(0.5, 0.5, "Poucos dados", ha='center', va='center')
            ax_norm[1].text(0.5, 0.5, "Poucos dados", ha='center', va='center')
        plt.tight_layout()
        results["plots"]["normalidade"] = fig_norm

        # 2. Homocedasticidade (Teste de Levene)
        homocedasticidade_ok = False
        grupos = [df_var[col_preco][df_var[var_cat] == categoria].dropna() for categoria in df_var[var_cat].unique()]
        grupos_validos = [g for g in grupos if len(g) >= 2]  # Levene precisa de grupos com pelo menos 2 obs
        if len(grupos_validos) >= 2:
            stat_levene, p_levene = levene(*grupos_validos)
            results["levene_test"] = (stat_levene, p_levene)
            if p_levene >= 0.05: homocedasticidade_ok = True
        results["homocedasticidade_ok"] = homocedasticidade_ok

        # 3. Kruskal-Wallis (se necessário)
        if not normalidade_ok or not homocedasticidade_ok:
            if len(grupos_validos) >= 2:
                stat_kruskal, p_kruskal = kruskal(*grupos_validos)
                results["kruskal_test"] = (stat_kruskal, p_kruskal)

        # Boxplot
        fig_box, ax_box = plt.subplots(figsize=(10, 5))
        unique_cats = df_var[var_cat].nunique()
        order_boxplot = None
        if unique_cats > 5 and unique_cats < 50:  # Evitar ordenar muitas categorias
            try:
                order_boxplot = df_var.groupby(var_cat)[col_preco].median().sort_values().index
            except Exception:
                order_boxplot = df_var[var_cat].unique()  # Fallback

        sns.boxplot(x=var_cat, y=col_preco, data=df_var, order=order_boxplot, ax=ax_box, palette="viridis")
        ax_box.set_title(f'Distribuição de {col_preco} por {var_cat}', fontsize=12)
        if unique_cats > 10:
            plt.setp(ax_box.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        else:
            plt.setp(ax_box.get_xticklabels(), fontsize=9)

        plt.tight_layout()
        results["plots"]["boxplot"] = fig_box

    except Exception as e:
        results["error"] = str(e)
    return results


# --- Interface do Streamlit ---
st.set_page_config(layout="wide", page_title="Dashboard de Análise Imobiliária ANOVA")

st.title("🏠 Dashboard de Análise Imobiliária com ANOVA")
st.markdown("""
Esta ferramenta interativa permite realizar Análises de Variância (ANOVA) no Ames Housing Dataset
para investigar como diferentes características categóricas impactam o preço de venda dos imóveis.
""")

# Carregar Dados
df, coluna_preco, colunas_categoricas_selecionaveis, todas_colunas = load_data()

if df is not None and coluna_preco is not None:
    st.header("1. Visão Geral dos Dados")
    if st.checkbox("Mostrar amostra dos dados"):
        st.dataframe(df.head())
    st.write(f"Total de registros carregados (após limpeza inicial na coluna '{coluna_preco}'): {len(df)}")
    st.write(f"Coluna alvo (preço): `{coluna_preco}`")

    st.sidebar.header("⚙️ Configurações da Análise")
    # Seleção de variáveis
    variaveis_selecionadas = st.sidebar.multiselect(
        "Escolha 1 a 3 variáveis categóricas para análise ANOVA:",
        options=colunas_categoricas_selecionaveis,
        max_selections=3
    )

    if variaveis_selecionadas:
        st.header("2. Resultados da Análise ANOVA")
        st.markdown(f"Analisando o impacto de **{', '.join(variaveis_selecionadas)}** sobre **{coluna_preco}**.")

        for var_analisada in variaveis_selecionadas:
            st.subheader(f"Análise para: `{var_analisada}`")

            # Prepara dados específicos para a variável (remove NaNs apenas para as colunas envolvidas)
            df_analise_var = df[[var_analisada, coluna_preco]].copy()
            df_analise_var.dropna(subset=[var_analisada, coluna_preco], inplace=True)

            if df_analise_var.empty or df_analise_var[var_analisada].nunique() < 2:
                st.warning(f"Não há dados suficientes ou níveis para '{var_analisada}' após limpeza. Pulando.")
                continue

            resultados_var = perform_anova_for_variable(df_analise_var, var_analisada, coluna_preco)

            if "error" in resultados_var:
                st.error(f"Erro ao analisar '{var_analisada}': {resultados_var['error']}")
                continue

            # Exibir Tabela ANOVA
            if "anova_table" in resultados_var:
                st.markdown("**Tabela ANOVA:**")
                st.dataframe(resultados_var["anova_table"])
                p_anova = resultados_var.get("p_valor_anova")
                if p_anova is not None:
                    if p_anova < 0.05:
                        st.success(
                            f"✅ ANOVA: Há uma diferença estatisticamente significativa nos preços (p-valor: {p_anova:.4e}).")
                    else:
                        st.info(
                            f"ℹ️ ANOVA: Não há uma diferença estatisticamente significativa nos preços (p-valor: {p_anova:.4e}).")

            # Pressupostos e Testes Alternativos
            with st.expander("Verificar Pressupostos da ANOVA e Testes Alternativos"):
                st.markdown("**Normalidade dos Resíduos:**")
                if "shapiro_test" in resultados_var:
                    stat, p_val = resultados_var["shapiro_test"]
                    st.write(f"Shapiro-Wilk: Estatística={stat:.4f}, P-valor={p_val:.4e}")
                elif "anderson_test" in resultados_var:
                    ad_res = resultados_var["anderson_test"]
                    st.write(f"Anderson-Darling: Estatística={ad_res.statistic:.4f}")
                    # st.write(f"  Valores Críticos: {ad_res.critical_values}")
                    # st.write(f"  Níveis de Significância: {ad_res.significance_level}")

                if resultados_var.get("normalidade_ok"):
                    st.success("✅ Resíduos parecem ser normalmente distribuídos.")
                else:
                    st.warning("⚠️ Resíduos NÃO parecem ser normalmente distribuídos.")

                if "normalidade" in resultados_var["plots"]:
                    st.pyplot(resultados_var["plots"]["normalidade"])

                st.markdown("**Homogeneidade das Variâncias (Homocedasticidade):**")
                if "levene_test" in resultados_var:
                    stat_l, p_l = resultados_var["levene_test"]
                    st.write(f"Teste de Levene: Estatística={stat_l:.4f}, P-valor={p_l:.4e}")
                    if resultados_var.get("homocedasticidade_ok"):
                        st.success("✅ Variâncias parecem ser homogêneas.")
                    else:
                        st.warning("⚠️ Variâncias NÃO parecem ser homogêneas.")
                else:
                    st.write("Teste de Levene não pôde ser realizado (dados insuficientes).")

                if "kruskal_test" in resultados_var:
                    st.markdown("**Teste de Kruskal-Wallis (Alternativa Não Paramétrica):**")
                    stat_k, p_k = resultados_var["kruskal_test"]
                    st.write(f"Kruskal-Wallis: Estatística={stat_k:.4f}, P-valor={p_k:.4e}")
                    if p_k < 0.05:
                        st.success(f"✅ Kruskal-Wallis: Diferença significativa nas medianas dos preços.")
                    else:
                        st.info(f"ℹ️ Kruskal-Wallis: Sem diferença significativa nas medianas dos preços.")

            # Boxplot
            if "boxplot" in resultados_var["plots"]:
                st.markdown("**Distribuição de Preços por Categoria:**")
                st.pyplot(resultados_var["plots"]["boxplot"])

            st.markdown("---")  # Separador entre variáveis

    elif not variaveis_selecionadas and st.sidebar.button("Analisar", type="primary",
                                                          help="Clique para iniciar após selecionar as variáveis.",
                                                          use_container_width=True, disabled=True):
        # Botão fica desabilitado até selecionar algo, apenas para feedback visual
        pass

    st.sidebar.markdown("---")
    st.sidebar.markdown("Desenvolvido como parte de uma análise de dados imobiliários.")

elif df is None and coluna_preco is None:  # Falha no carregamento
    st.warning("Aguardando carregamento dos dados ou verifique os erros acima.")
else:  # Carregou mas não achou coluna de preço ou não há categóricas
    if coluna_preco is None:
        st.error(
            f"A coluna de preço de venda ('saleprice' ou similar) não foi encontrada no dataset. Verifique as colunas disponíveis: {todas_colunas}")
    if not colunas_categoricas_selecionaveis:
        st.error("Nenhuma coluna categórica adequada para análise foi identificada.")

# No final do script Streamlit, após o loop de análise das variáveis

if variaveis_selecionadas: # Somente mostrar se alguma análise foi feita
    st.header("3. Insights Gerais e Recomendações")
    with st.expander("Ver Análise Detalhada e Recomendações"):
        st.markdown("""
        ### Como Interpretar os Resultados para Tomada de Decisão:

        A análise ANOVA nos ajuda a entender se uma característica específica da casa (como estilo da casa,
        ano da venda, ou estilo do telhado) tem uma associação estatisticamente significativa com o preço
        médio de venda.

        **Para as variáveis analisadas (`HouseStyle`, `YrSold`, `RoofStyle`):**

        #### `HouseStyle` (Estilo da Moradia):
        * **Impacto Geral:** Geralmente significativo. Estilos diferentes (Térrea, Dois Andares, Níveis Divididos)
            atraem diferentes compradores e têm diferentes custos e áreas construídas.
        * **Orientação para Corretores:** Utilize o estilo para segmentar o marketing e justificar faixas de preço.
        * **Orientação para Investidores:** Analise a popularidade e o potencial de valorização de diferentes estilos
            na sua área de interesse.

        #### `YrSold` (Ano da Venda):
        * **Impacto Geral:** Pode ser significativo se o mercado passou por mudanças (altas ou baixas) durante
            os anos analisados (ex: 2006-2010). Reflete tendências macroeconômicas.
        * **Orientação para Corretores:** Fornece contexto histórico para a precificação atual e ajuda a gerenciar
            expectativas.
        * **Orientação para Investidores:** Sublinha a importância de entender os ciclos de mercado, embora os dados
            históricos de `YrSold` não prevejam o futuro diretamente.

        #### `RoofStyle` (Estilo do Telhado):
        * **Impacto Geral:** Pode ser significativo. Certos estilos (ex: Quatro Águas vs. Duas Águas) podem estar
            associados a diferentes níveis de custo, durabilidade e estética.
        * **Orientação para Corretores:** Um detalhe que pode agregar valor, especialmente se o telhado for novo
            ou de um estilo particularmente desejável ou durável.
        * **Orientação para Investidores:** O custo de manutenção e substituição pode variar com o estilo do telhado.
            A condição do telhado é mais crítica que o estilo em si, mas o estilo influencia o custo.

        **Recomendações Gerais:**
        * **Corretores:** Usem esses insights para refinar suas estratégias de precificação, marketing e aconselhamento
            aos clientes. Uma casa não é apenas um conjunto de quartos, mas um conjunto de características que, juntas,
            determinam seu valor.
        * **Investidores:** Considerem como essas características (e outras) se alinham com seus objetivos de
            investimento, seja para renda, "flipping" ou valorização a longo prazo. Focar em características
            que têm um impacto positivo e duradouro no valor é fundamental.

        *Lembre-se que a ANOVA univariada mostra a relação de uma variável por vez com o preço.
        Para uma análise mais completa do impacto combinado de múltiplas variáveis, a Regressão Linear (Parte II da sua tarefa original) seria o próximo passo.*
        """)
