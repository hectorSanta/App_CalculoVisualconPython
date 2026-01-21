import streamlit as st
import numpy as np
import sympy as sp
import plotly.graph_objects as go
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor
)

# =============================
# CONFIGURACIÓN GENERAL
# =============================
st.set_page_config(
    page_title="Cálculo Visual con Python",
    layout="wide",
    page_icon="📘"
)

st.markdown(
    "<h1 style='text-align:center;'>📘 Simulador Visual Para el Aprendizaje de Funciones</h1>"
    "<p style='text-align:center;'>Simulador interactivo para aprender Deerivadas y Integrales</p>",
    unsafe_allow_html=True
)

# =============================
# PARSER ROBUSTO
# =============================
X = sp.Symbol("x")

TRANSFORMATIONS = (
    standard_transformations +
    (implicit_multiplication_application, convert_xor)
)

# Funciones permitidas
SAFE_FUNCTIONS = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "ln": sp.log,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
    "pi": sp.pi,
    "e": sp.E
}

def limpiar_entrada(expr: str) -> str:
    """
    Normaliza la entrada del usuario
    """
    expr = expr.lower().strip()
    reemplazos = {
        "sen": "sin",
        "π": "pi",
        "^": "**",
        "|x|": "abs(x)"
    }
    for k, v in reemplazos.items():
        expr = expr.replace(k, v)
    return expr

def parsear_funcion(expr_str: str):
    """
    Intenta convertir texto a expresión simbólica segura
    """
    try:
        expr = parse_expr(
            expr_str,
            local_dict=SAFE_FUNCTIONS | {"x": X},
            transformations=TRANSFORMATIONS,
            evaluate=True
        )
        if not expr.has(X):
            raise ValueError("La función debe depender de x")
        return expr, None
    except Exception as e:
        return None, str(e)

def lambdify_seguro(expr):
    """
    Evalúa sin romper la app
    """
    try:
        f = sp.lambdify(X, expr, modules=["numpy"])
        def wrapper(x):
            try:
                y = f(x)
                y = np.array(y, dtype=float)
                y[~np.isfinite(y)] = np.nan
                return y
            except:
                return np.full_like(x, np.nan)
        return wrapper
    except:
        return None

# =============================
# SIDEBAR
# =============================
with st.sidebar:
    st.header("⚙ Configuración")

    raw_input = st.text_input(
        "Ingresa f(x):",
        value="x^2 + 5",
        help="Ejemplos: x^2, sin(x), 2(x+1), ln(x), sqrt(x)"
    )

    xmin = st.number_input(
        "x mínimo",
        min_value=-35.0,
        max_value=0.0,
        value=-10.0,
        step=0.5
    )

    xmax = st.number_input(
        "x máximo",
        min_value=0.0,
        max_value=35.0,
        value=10.0,
        step=0.5
    )

    st.markdown("---")
    show_f = st.checkbox("Mostrar f(x)", True)
    show_d = st.checkbox("Mostrar derivada", True)
    show_i = st.checkbox("Mostrar integral", False)
    show_area = st.checkbox("Área bajo la curva", False)
    
    if show_area:
        st.subheader("Rango de Integración")
        a_b = st.slider("Intervalo [a, b]", float(xmin), float(xmax), (0.0, 3.0))
        a_int, b_int = a_b


    st.markdown("---")
    x0 = st.slider("Punto x₀", xmin, xmax, (xmin + xmax) / 4)

resolution = 1200
# =============================
# PROCESAMIENTO
# =============================
expr_limpia = limpiar_entrada(raw_input)
f_sym, error = parsear_funcion(expr_limpia)

if error:
    st.error("❌ No se pudo interpretar la función.")
    st.info("Ejemplos válidos: x^2, sin(x), 5x+3, ln(x), sqrt(x)")
    st.stop()

f = lambdify_seguro(f_sym)

try:
    d_sym = sp.diff(f_sym, X)
    df = lambdify_seguro(d_sym)
except:
    d_sym, df = None, None

try:
    i_sym = sp.integrate(f_sym, X)
    Fi = lambdify_seguro(i_sym)
except:
    i_sym, Fi = None, None

xs = np.linspace(xmin, xmax, resolution)
ys = f(xs)

# =============================
# GRÁFICA
# =============================
fig = go.Figure()

if show_f:
    fig.add_trace(go.Scatter(x=xs, y=ys, name="f(x)", line=dict(width=3)))

# Área bajo la curva
if show_area:
    x_fill = np.linspace(a_int, b_int, 400)
    y_fill = f(x_fill)
    fig.add_trace(go.Scatter(
        x=x_fill, y=y_fill,
        fill='tozeroy',
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(0, 150, 255, 0.3)',
        name='Área Definida',
        hoverinfo='skip'
    ))


if show_d and df:
    fig.add_trace(go.Scatter(x=xs, y=df(xs), name="f'(x)", line=dict(color="red", dash="dash")))

if show_i and Fi:
    fig.add_trace(go.Scatter(x=xs, y=Fi(xs), name="∫f(x)dx", line=dict(color="green", dash="dot")))

# Punto y tangente
if df:
    y0 = f(np.array([x0]))[0]
    slope = df(np.array([x0]))[0]
    # Dibujar una línea corta de tangente
    t_range = (xmax - xmin) * 0.1
    xt = np.linspace(x0 - t_range, x0 + t_range, 100)
    yt = slope * (xt - x0) + y0
    fig.add_trace(go.Scatter(x=xt, y=yt, name="Tangente", line=dict(color="orange", width=3)))
    fig.add_trace(go.Scatter(x=[x0], y=[y0], mode="markers", marker=dict(size=12, color="orange"), name="Punto x₀"))

fig.update_layout(
    height=650,
    template="plotly_white",
    legend=dict(orientation="h"),
    xaxis=dict(
        tickmode="linear",
        dtick=2  # 👈 separación entre números del eje X
    )
)

st.plotly_chart(fig, use_container_width=True)

# =============================
# EVALUACIÓN EN x₀
# =============================
fx0 = f(np.array([x0]))[0]

dfx0 = None
if df:
    dfx0 = df(np.array([x0]))[0]

Fx0 = None
if Fi:
    Fx0 = Fi(np.array([x0]))[0]


with st.expander("📐 Expresiones matemáticas"):
    col_math, col_res = st.columns([1, 1])

    # -------------------------
    # COLUMNA 1: Análisis simbólico
    # -------------------------
    with col_math:
        st.latex("f(x) = " + sp.latex(f_sym))
        st.latex(rf"f({x0:.2f}) = {fx0:.2f}")

        if d_sym:
            st.latex("f'(x) = " + sp.latex(d_sym))
            st.latex(rf"f'({x0:.2f}) = {dfx0:.2f}")

        if i_sym:
            st.latex(r"\int f(x)\,dx = " + sp.latex(i_sym))

    # -------------------------
    # COLUMNA 2: Área definida
    # -------------------------
    with col_res:
        if show_area:
            try:
                area_val = sp.integrate(f_sym, (X, a_int, b_int))
                st.latex(
                    r"\int_{" + f"{a_int:.2f}" + r"}^{" + f"{b_int:.2f}" + r"} f(x)\,dx"
                )
                st.metric("Resultado del área", f"{float(area_val):.2f}")
            except:
                st.warning("No se pudo calcular la integral exacta.")
