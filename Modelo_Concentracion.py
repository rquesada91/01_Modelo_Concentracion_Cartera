######################################### A. INTRODUCCIÓN #########################################
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.optimize import minimize
import math
from datetime import date


st.markdown("### A. Análisis de Concentración y Optimización de Portafolios")
st.markdown("""El modelo de Análisis de Concentración y Optimización de Portafolios permite evaluar cómo se distribuye el capital entre distintos activos y determinar combinaciones óptimas que equilibren rendimiento y riesgo. Su propósito es ofrecer una visión cuantitativa y estructurada del comportamiento de un portafolio, identificando tanto posibles excesos de exposición como oportunidades para mejorar la eficiencia financiera.
            
A través de métricas como la volatilidad, los retornos esperados, la correlación entre activos y los índices de concentración, el modelo ayuda a comprender qué tan diversificado está un portafolio y qué tan vulnerable podría ser ante movimientos adversos del mercado. 
            
Además, mediante técnicas de optimización —como la frontera eficiente o la asignación basada en riesgo— es posible construir portafolios que maximicen el rendimiento esperado para un nivel de riesgo dado, o que minimicen el riesgo manteniendo un retorno objetivo.
            
En conjunto, este enfoque permite tomar decisiones de inversión más informadas, transparentes y alineadas con los objetivos del inversionista, ya sea mejorar la diversificación, reducir la exposición a activos dominantes o identificar configuraciones más eficientes del capital.
""")

## 02. Incluir imagenes y acomodarlas por columnas. ##
col1, col2, col3 = st.columns(3, gap="small")

with col1:
        st.image(image = "imagen/1.Ilustrativa.png", width=300, caption = "imagen/Imagen Ilustrativa", output_format = "auto")




######################################### B. DESCARGA DE DATOS #########################################
st.divider()
# 1. Configuración de la página
st.set_page_config(page_title="Descargador de Acciones", page_icon="📈")
st.markdown("### B. Descargador de Datos Financieros")

# 2. Barra lateral para los parámetros
st.sidebar.header("Configuración")

# === NUEVOS CONTROLES PARA MONTECARLO Y TASA LIBRE DE RIESGO ===
st.sidebar.subheader("⚙️ Parámetros de Optimización")

# Control para iteraciones de Montecarlo
iteraciones_montecarlo = st.sidebar.number_input(
    "Iteraciones de Montecarlo:",
    min_value=100,
    max_value=10000,
    value=1500,
    step=100,
    help="Número de simulaciones aleatorias para la frontera eficiente"
)
st.session_state['iteraciones_montecarlo'] = iteraciones_montecarlo

# Control para tasa libre de riesgo
tasa_rf = st.sidebar.number_input(
    "Tasa Libre de Riesgo (%):",
    min_value=0.0,
    max_value=20.0,
    value=3.3,
    step=0.1,
    help="Tasa de referencia anual (ej: bonos del tesoro)"
)
st.session_state['tasa_rf'] = tasa_rf / 100  # Convertir a decimal

st.sidebar.divider()
# === FIN DE NUEVOS CONTROLES ===

# 3. Entrada de texto para las empresas seleccionadas
ticker_input = st.sidebar.text_input(
    "Símbolos de las acciones (separados por coma):", 
    value="MSFT, KO"
)

# 4. Selección de fechas
col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Fecha de inicio", value=date(2021, 1, 1))
end_date = col2.date_input("Fecha de fin", value=date(2023, 6, 30))

# Guardar fechas en session_state
st.session_state['start_date'] = start_date
st.session_state['end_date'] = end_date

# 5. Definimos las opciones disponibles en yfinance
opciones_metricas = ['Close', 'High', 'Low', 'Open', 'Volume']
metrica_seleccionada = st.sidebar.selectbox(
    "¿Qué métrica deseas analizar?",
    options=opciones_metricas,
    index=0  # Cambiado a Close (índice 0)
)
# Guardar métrica en session_state
st.session_state['metrica_seleccionada'] = metrica_seleccionada

# 6. Lógica de descarga
if st.sidebar.button("Descargar Datos"):
    empresas = [x.strip() for x in ticker_input.split(',')]
    
    if not empresas or ticker_input.strip() == "":
        st.error("Por favor, ingresa al menos un símbolo de acción.")
    else:
        try:
            with st.spinner(f'Descargando datos...'):
                datos_raw = yf.download(empresas, start=start_date, end=end_date)
            if not datos_raw.empty:
                datos_filtrados = datos_raw[metrica_seleccionada]
                st.success(f"¡Descarga exitosa! Mostrando datos de: **{metrica_seleccionada}**")
                
                # Visualización
                st.markdown(f"Tabla de datos ({metrica_seleccionada})")
                st.dataframe(datos_filtrados.tail())
                
                st.markdown(f"Gráfico de evolución ({metrica_seleccionada})")
                st.line_chart(datos_filtrados)
            else:
                st.warning("No se encontraron datos.")
        except Exception as e:
            st.error(f"Ocurrió un error: {e}")
            st.write("Detalle:", e)
else:
    st.info("Configura los parámetros y la métrica deseada en la barra lateral.")






######################################### C. CALCULOS INICIALES #########################################
st.divider()
st.markdown("### C. Calculos Variaciones")
st.markdown("Antes de trabajar en los indicadores, debemos realizar los siguientes cálculos base:")
st.markdown("Rendimientos diarios (Ri)")
col1, col2, col3 = st.columns(3, gap="small")
with col1:
        st.image(image = "imagen/2.Formula Rendimientos Log.png", width=300, caption = "Formula Rendimientos Logaritmicos", output_format = "auto")

# Paso 1: Capturar los datos del bloque anterior
if 'datos_filtrados' in locals():
    st.session_state['datos_para_analisis'] = datos_filtrados

if 'datos_para_analisis' in st.session_state:
    datos = st.session_state['datos_para_analisis']
    
    if st.button("Calcular Variaciones Logarítmicas y Métricas Anualizadas"):
        # 1. Cálculo de retornos diarios
        retornos_diarios = np.log(datos / datos.shift(1)).dropna()
        
        # 2. Cálculos Anualizados
        rendimiento_anualizado = (1 + retornos_diarios.mean())**252 - 1
        volatilidad_anualizada = retornos_diarios.std() * np.sqrt(252)
        
        # --- Visualización de Resultados ---
        st.subheader("Métricas Anualizadas")
        col_rend, col_vol = st.columns(2)
        
        with col_rend:
            st.write("**Rendimiento Anualizado**")
            st.dataframe(rendimiento_anualizado.rename("Rendimiento %") * 100)
            
        with col_vol:
            st.write("**Volatilidad Anualizada (Riesgo)**")
            st.dataframe(volatilidad_anualizada.rename("Volatilidad %") * 100)

        st.markdown("---")
        
        st.subheader("Gráfico de Retornos Diarios")
        st.line_chart(retornos_diarios)
        
        with st.expander("Ver tabla de retornos diarios"):
            st.dataframe(retornos_diarios)
            
        st.success("Análisis completado exitosamente.")
else:
    st.info("⚠️ Para activar este botón, primero debes realizar la descarga de datos exitosamente arriba.")



######################################### D . MATRIZ DE COVARIANZA Y RENDIMIENTOS #########################################
st.divider()

if 'retornos_diarios' in locals():
    st.session_state['retornos_calculados'] = retornos_diarios

st.markdown("### D. Análisis de Covarianzas y Rendimientos Anualizados")
if 'retornos_calculados' in st.session_state:
    retornos = st.session_state['retornos_calculados']
    
    if st.button("Calcular Matrices de Riesgo y Rendimiento"):
        covarianzas_diarias = retornos.cov()
        rendimiento_anualizado = retornos.mean() * 252
        covarianzas_anualizada = covarianzas_diarias * 252
        
        st.markdown("### Rendimiento Esperado Anualizado")
        st.bar_chart(rendimiento_anualizado)
        st.dataframe(rendimiento_anualizado.rename("Retorno Anualizado"))

        st.markdown("### Matriz de Covarianzas Anualizada")
        st.write("Esta matriz mide la relación de movimiento entre los activos:")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(covarianzas_anualizada, annot=True, cmap="YlGnBu", ax=ax)
        st.pyplot(fig)
        
        with st.expander("Ver tabla de Covarianzas Anualizadas"):
            st.dataframe(covarianzas_anualizada)

        with st.expander("Ver Covarianzas Diarias"):
            st.dataframe(covarianzas_diarias)
            
        st.success("Cálculos de matrices completados.")
else:
    st.info("💡 Para activar esta sección, primero debes ejecutar el 'Análisis de Variaciones' en la sección de arriba.")



######################################### E. CALCULO DE TODOS LOS INDICADORES #####################################
st.divider()
st.markdown("### E. Calculo de Modelos de Optimización de Portafolios")
st.info(f" Usando **{st.session_state.get('iteraciones_montecarlo', 1500)} iteraciones** y tasa libre de riesgo de **{st.session_state.get('tasa_rf', 0.033)*100:.1f}%**")


st.markdown("##### 🎯 1. Mínima Varianza")
st.markdown("""Es una estrategia de inversión que busca construir el portafolio con el MENOR riesgo posible,
            sin importar tanto el rendimiento, se utiliza cuando el perfil de inversión es concervador""")
col1, col2, col3 = st.columns(3, gap="small")
with col1:
        st.image(image = "imagen/6.Varianza_Minima.png", width=300, caption = "Varianza Minima", output_format = "auto")


st.markdown("##### 🎯 2. Asimetría (Skewness) y Curtosis (Kurtosis)")
st.markdown("""Son medidas estadísticas que nos dicen cómo se distribuyen los retornos de una inversión,
            más allá del simple promedio y volatilidad.""")
col1, col2, col3 = st.columns(3, gap="small")
with col1:
        st.image(image = "imagen/7.Asimetria y Curtosis.png", width=300, caption = "Asimetria y Curtosis", output_format = "auto")


st.markdown("##### 🎯 3. Ratio Sharpe")
st.markdown("""¿Cuánto rendimiento extra estoy obteniendo por cada unidad de riesgo que tomo?""")
col1, col2, col3 = st.columns(3, gap="small")
with col1:
        st.image(image = "imagen/8. Ratio_Sharpe.png", width=300, caption = "Ratio Sharpe", output_format = "auto")


st.markdown("##### 🎯 4. Ratio Sortino")
st.markdown("""Es una versión mejorada del Sharpe que solo penaliza la volatilidad mala (caídas), ignorando la volatilidad buena (subidas)""")
col1, col2, col3 = st.columns(3, gap="small")
with col1:
        st.image(image = "imagen/9. Ratio_Sortino.png", width=300, caption = "Ratio Sortino", output_format = "auto")


st.markdown("##### 🎯 5. Ratio Treynor")
st.markdown("""EEvalúa el rendimiento por unidad de riesgo de mercado. Útil para portafolios bien diversificados.""")
col1, col2, col3 = st.columns(3, gap="small")
with col1:
        st.image(image = "imagen/10. Ratio_Treynor.png", width=300, caption = "Ratio Sortino", output_format = "auto")

if 'retornos_calculados' in st.session_state:
    ret_d = st.session_state['retornos_calculados']
    activos = ret_d.columns
    n_activos = len(activos)
    rend_anual = ret_d.mean() * 252
    cov_anual = ret_d.cov() * 252
    tasa_rf = st.session_state.get('tasa_rf', 0.033)
    iteraciones = st.session_state.get('iteraciones_montecarlo', 1500)
    f_inicio = st.session_state.get('start_date', '2021-01-01')
    f_fin = st.session_state.get('end_date', '2023-12-31')

    if st.button("🔥 Ejecutar Todas las Estrategias"):
        barra_progreso = st.progress(0)
        status_text = st.empty()

        w0 = np.array([1/n_activos] * n_activos)
        bounds = [(0, 1) for _ in range(n_activos)]
        cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

        # --- 1. Volatilidad Minima o Varianza minima ---
        status_text.text("Calculando Mínima Varianza...") ## Muestra mensaje al usuario mientras calcula
        res_mv = minimize(lambda w, c: np.dot(w.T, np.dot(c, w)), w0, args=(cov_anual,), method='SLSQP', bounds=bounds, constraints=cons)        
        if res_mv.success: st.session_state['pesos_minima_varianza'] = res_mv.x
        barra_progreso.progress(20)

        # --- 2. ASIMETRÍA/CURTOSIS ---
        status_text.text("Calculando Asimetría y Curtosis...")
        l_asim = st.session_state.get('l_asimetria', 1.0)
        l_curt = st.session_state.get('l_curtosis', 1.0)
        def obj_asim(w, y):
            p_ret = np.dot(y, w)
            return -l_asim * (np.mean((p_ret - p_ret.mean())**3)/np.std(p_ret)**3) + l_curt * (np.mean((p_ret - p_ret.mean())**4)/np.std(p_ret)**4)
        res_as = minimize(obj_asim, w0, args=(ret_d.values,), method='SLSQP', bounds=bounds, constraints=cons)
        if res_as.success: st.session_state['pesos_asimetria'] = res_as.x
        barra_progreso.progress(40)

        # --- 3. SHARPE ---
        status_text.text("Calculando Ratio Sharpe...")
        res_sh = minimize(lambda w, r, c, f: -(np.dot(w, r) - f) / np.sqrt(np.dot(w.T, np.dot(c, w))), w0, args=(rend_anual, cov_anual, tasa_rf), method='SLSQP', bounds=bounds, constraints=cons)
        if res_sh.success: st.session_state['pesos_optimos_sharpe'] = res_sh.x
        barra_progreso.progress(60)

        # --- 4. SORTINO ---
        status_text.text("Calculando Ratio Sortino...")
        def obj_sort(w, rd, f):
            p_rend = np.dot(w, rd.mean()) * 252
            p_returns = np.dot(rd, w)
            downside = np.std(p_returns[p_returns < 0]) * np.sqrt(252)
            return -(p_rend - f) / downside if downside > 0 else 1e10
        res_so = minimize(obj_sort, w0, args=(ret_d, tasa_rf), method='SLSQP', bounds=bounds, constraints=cons)
        if res_so.success: st.session_state['pesos_optimos_sortino'] = res_so.x
        barra_progreso.progress(80)

        # --- 5. TREYNOR ---
        status_text.text("Calculando Ratio Treynor (Sincronizando Mercado)...")
        merc_data = yf.download('^GSPC', start=f_inicio, end=f_fin, progress=False)
        metrica_usuario = st.session_state.get('metrica_seleccionada', 'Close')
        merc_ret = merc_data[metrica_usuario].pct_change().dropna()
        def obj_tr(w, rd, f, m):
            p_ret = (1 + rd.dot(w).mean()) ** 252 - 1
            df_s = pd.concat([rd.dot(w), m], axis=1).dropna()
            beta = sm.OLS(df_s.iloc[:,0], sm.add_constant(df_s.iloc[:,1])).fit().params.iloc[1]
            return -(p_ret - f) / beta if beta > 0 else 1e10
        res_tr = minimize(obj_tr, w0, args=(ret_d, tasa_rf, merc_ret), method='SLSQP', bounds=bounds, constraints=cons)
        if res_tr.success: st.session_state['pesos_optimos_treynor'] = res_tr.x
        barra_progreso.progress(100)
        status_text.success("🎉 ¡Todos los modelos calculados y listos para comparar!")

        # --- GRÁFICO MAESTRO CON MONTECARLO DINÁMICO ---
        fig, ax = plt.subplots(figsize=(10, 6))
        p_vols, p_rets = [], []
        for _ in range(iteraciones):  # Usa el valor del slider
            w = np.random.random(n_activos); w /= np.sum(w)
            p_rets.append(np.dot(w, rend_anual)); p_vols.append(np.sqrt(np.dot(w.T, np.dot(cov_anual, w))))
        ax.scatter(p_vols, p_rets, c='gray', s=2, alpha=0.1)

        modelos = {
            'pesos_minima_varianza': ('Mín. Varianza', 'red', 'o'),
            'pesos_asimetria': ('Asimetría', 'green', '*'),
            'pesos_optimos_sharpe': ('Sharpe', 'blue', '*'),
            'pesos_optimos_sortino': ('Sortino', 'purple', '*'),
            'pesos_optimos_treynor': ('Treynor', 'black', '*')
        }

        for clave, (nom, col, mar) in modelos.items():
            if clave in st.session_state:
                p = st.session_state[clave]
                ax.scatter(np.sqrt(np.dot(p.T, np.dot(cov_anual, p))), np.dot(p, rend_anual), 
                           color=col, marker=mar, s=250, label=nom, edgecolors='white', zorder=10)

        ax.set_title("Comparativa de Estrategias: Espacio Riesgo-Retorno")
        ax.set_xlabel("Riesgo (Volatilidad Anualizada)")
        ax.set_ylabel("Rendimiento Anualizado")
        ax.legend(); ax.grid(True, alpha=0.2)
        st.pyplot(fig)

        st.subheader("📊 Comparativo de Pesos por Modelo")
        resumen_pesos = pd.DataFrame(index=activos)
        for clave, (nom, _, _) in modelos.items():
            if clave in st.session_state:
                resumen_pesos[nom] = (st.session_state[clave] * 100).round(2)
        st.dataframe(resumen_pesos.T.style.highlight_max(axis=0, color='lightgreen'))

else:
    st.warning("Debe descargar los datos primero.")


######################################### F. BACKTESTING Y COMPARATIVA FINAL #########################################
st.divider()
st.markdown("### F. Backtesting: Comparación de Rendimientos Reales")

if 'pesos_minima_varianza' in locals(): st.session_state['pesos_minima_varianza'] = pesos_minima_varianza
if 'pesos_asimetria' in locals(): st.session_state['pesos_asimetria'] = pesos_asimetria
if 'pesos_optimos_sharpe' in locals(): st.session_state['pesos_optimos_sharpe'] = pesos_optimos_sharpe
if 'pesos_optimos_sortino' in locals(): st.session_state['pesos_optimos_sortino'] = pesos_optimos_sortino
if 'pesos_optimos_treynor' in locals(): st.session_state['pesos_optimos_treynor'] = pesos_optimos_treynor

modelos_a_evaluar = {
    'Mínima Varianza': 'pesos_minima_varianza',
    'Asimetría/Curtosis': 'pesos_asimetria',
    'Ratio Sharpe': 'pesos_optimos_sharpe',
    'Ratio Sortino': 'pesos_optimos_sortino',
    'Ratio Treynor': 'pesos_optimos_treynor'
}

disponibles = [nombre for nombre, var in modelos_a_evaluar.items() if var in st.session_state]

if disponibles:
    st.success(f"✅ Listos para comparar: {', '.join(disponibles)}")
    
    if st.button("🚀 Ejecutar Comparativa de Rendimientos"):
        with st.spinner('Analizando datos históricos...'):
            
            f_inicio = start_date 
            f_fin = end_date
            
            empresas_lista = [x.strip() for x in ticker_input.split(',')]
            metrica_usuario = st.session_state.get('metrica_seleccionada', 'Close')
            datos_backtest = yf.download(empresas_lista, start=f_inicio, end=f_fin, progress=False)[metrica_usuario]
            
            if not datos_backtest.empty:
                retornos_test = datos_backtest.pct_change().dropna()
                test_df = pd.DataFrame(index=retornos_test.index)

                for nombre, var_name in modelos_a_evaluar.items():
                    if var_name in st.session_state:
                        pesos = st.session_state[var_name]
                        test_df[nombre] = retornos_test.dot(pesos)
                
                test_acumulado = (1 + test_df).cumprod()

                st.subheader(f"Evolución de la Inversión ({f_inicio} a {f_fin})")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                for columna in test_acumulado.columns:
                    ax.plot(test_acumulado.index, test_acumulado[columna], label=columna, linewidth=2)
                
                ax.set_title("Crecimiento de $1 USD por Estrategia", fontsize=12)
                ax.set_ylabel("Valor del Portafolio ($)")
                ax.set_xlabel("Fecha")
                ax.legend(loc='upper left', frameon=True)
                ax.grid(True, alpha=0.2)
                plt.xticks(rotation=45)
                
                st.pyplot(fig)

                st.subheader("📊 Resumen de Desempeño")
                
                resumen = pd.DataFrame({
                    "Retorno Total (%)": (test_acumulado.iloc[-1] - 1) * 100,
                    "Volatilidad Anual (%)": test_df.std() * np.sqrt(252) * 100,
                    "Valor Final ($)": test_acumulado.iloc[-1]
                }).sort_values(by="Retorno Total (%)", ascending=False)
                
                st.dataframe(resumen.style.format({
                    "Retorno Total (%)": "{:.2f}%",
                    "Volatilidad Anual (%)": "{:.2f}%",
                    "Valor Final ($)": "${:.2f}"
                }))
                
                st.balloons()
            else:
                st.error(f"No se encontraron datos para los tickers en el periodo {f_inicio} a {f_fin}.")
else:
    st.warning("⚠️ Primero debes ejecutar al menos una optimización (Sharpe, Treynor, etc.) para comparar.")
