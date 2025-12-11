import streamlit as st
import numpy as np
import sympy as sp
import plotly.graph_objects as go
import plotly.io as pio
import math
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
)

# ---------------------------
# Configuración de página
# ---------------------------
st.set_page_config(page_title="Cálculo Visual con Python", layout="wide", page_icon="📘")
st.markdown("<h1 style='text-align:center;'>📘 Cálculo Visual con Python — Simulador Interactivo de Cálculo</h1>", unsafe_allow_html=True)

# ---------------------------
# Helpers
# ---------------------------
TRANS = standard_transformations + (implicit_multiplication_application, convert_xor)
X = sp.Symbol("x")

def limpiar(expr):
    return expr.replace("^", "**").replace("sen", "sin").replace("π", "pi").strip()

def safe_lamb(sym_expr):
    try:
        f = sp.lambdify(X, sym_expr, modules=["numpy"])
    except:
        return None

    def env(arr):
        try:
            y = f(arr)
            y = np.array(y, dtype=float)
            y[~np.isfinite(y)] = np.nan
            return y
        except:
            out = []
            for xi in np.atleast_1d(arr):
                try:
                    out.append(float(f(xi)))
                except:
                    out.append(np.nan)
            return np.array(out, dtype=float)

    return env

def puntos_criticos(expr, xmin, xmax):
    crit = []
    try:
        d = sp.diff(expr, X)
        sol = sp.solve(sp.Eq(d, 0), X)
        for s in sol:
            try:
                v = float(s.evalf())
                if xmin <= v <= xmax:
                    crit.append(v)
            except:
                pass
    except:
        pass
    return sorted(set(crit))

def clasificar(expr, x0):
    try:
        d2 = sp.diff(expr, X, 2)
        val = float(d2.subs(X, x0))
        if val > 0: return "mínimo local"
        if val < 0: return "máximo local"
        return "inconcluso"
    except:
        return "inconcluso"

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("⚙️ Configuración")
    raw = limpiar(st.text_input("Ingresa f(x):", "sin(x) + x**2"))
    xmin = st.number_input("x mínimo", value=-6.0)
    xmax = st.number_input("x máximo", value=6.0)
    resolution = st.slider("Resolución", 200, 3000, 1200)

    st.markdown("---")
    st.subheader("Capas")
    show_f = st.checkbox("Mostrar f(x)", True)
    show_d = st.checkbox("Mostrar f'(x)", True)
    show_p = st.checkbox("Mostrar F(x) (primitiva)", False)
    show_area = st.checkbox("Mostrar área", False)

    st.markdown("---")
    st.subheader("Punto")
    x0 = st.slider("x₀", xmin, xmax, float((xmin + xmax)/4))

# ---------------------------
# Parsing simbólico
# ---------------------------
try:
    f_sym = parse_expr(raw, transformations=TRANS, evaluate=True)
except Exception as e:
    st.error("La función es inválida.")
    st.stop()

# Derivada e integral
try:
    d_sym = sp.diff(f_sym, X)
except:
    d_sym = None

try:
    p_sym = sp.integrate(f_sym, X)
except:
    p_sym = None

# lambdas
f = safe_lamb(f_sym)
df = safe_lamb(d_sym) if d_sym is not None else None
Fp = safe_lamb(p_sym) if p_sym is not None else None

# Dominio
xs = np.linspace(xmin, xmax, resolution)
ys = f(xs)

# Puntos críticos
crit = puntos_criticos(f_sym, xmin, xmax)
crit_info = []
for c in crit:
    try:
        yc = float(f_sym.subs(X, c))
    except:
        yc = None
    crit_info.append((c, yc, clasificar(f_sym, c)))

# ---------------------------
# Evaluación en x0
# ---------------------------
try:
    y0 = float(f_sym.subs(X, x0))
except:
    y0 = float(f(np.array([x0]))[0])

try:
    slope = float(d_sym.subs(X, x0))
except:
    slope = float(df(np.array([x0]))[0]) if df else float("nan")

# ---------------------------
# Figura
# ---------------------------
fig = go.Figure()

# f(x)
if show_f:
    fig.add_trace(go.Scatter(x=xs, y=ys, name="f(x)", line=dict(width=3)))

# f'(x)
if show_d and df is not None:
    fig.add_trace(go.Scatter(x=xs, y=df(xs), name="f'(x)", line=dict(color="red", dash="dash")))

# F(x)
if show_p:
    if p_sym is not None:
        fig.add_trace(go.Scatter(x=xs, y=Fp(xs), name="F(x)", line=dict(color="green", dash="dot")))
    else:
        fig.add_trace(go.Scatter(x=xs, y=np.cumsum(ys), name="F(x)≈"))

# Área
if show_area:
    with st.sidebar:
        a = st.number_input("a", value=-2.0)
        b = st.number_input("b", value=2.0)
    if a < b:
        mask = (xs >= a) & (xs <= b)
        fig.add_trace(go.Scatter(
            x=np.concatenate([[a], xs[mask], [b]]),
            y=np.concatenate([[0], ys[mask], [0]]),
            fill="toself",
            fillcolor="rgba(0,255,0,0.2)",
            name="Área"
        ))
        try:
            area = float(sp.integrate(f_sym, (X, a, b)))
        except:
            area = float(np.trapz(ys[mask], xs[mask]))
        st.success(f"Área = {area}")

# Punto
fig.add_trace(go.Scatter(
    x=[x0], y=[y0], mode="markers",
    marker=dict(size=10, color="orange"),
    name="x₀"
))

# Tangente
xt = np.linspace(x0 - (xmax - xmin) * 0.15, x0 + (xmax - xmin) * 0.15, 200)
yt = slope * (xt - x0) + y0
fig.add_trace(go.Scatter(
    x=xt, y=yt, name="Tangente",
    line=dict(color="orange", dash="dot")
))

# Puntos críticos
for c, yc, tipo in crit_info:
    fig.add_trace(go.Scatter(
        x=[c], y=[yc],
        mode="markers+text",
        marker=dict(size=10, color="#8e44ad"),
        text=[tipo],
        textposition="top center",
        name=f"Crítico {tipo}"
    ))

fig.update_layout(
    height=650,
    template="plotly_white",
    legend=dict(orientation="h", y=1.05)
)

# Mostrar gráfica
st.plotly_chart(fig, use_container_width=True)

# Simbología
with st.expander("Expresiones simbólicas"):
    st.latex("f(x)=" + sp.latex(f_sym))
    if d_sym is not None:
        st.latex("f'(x)=" + sp.latex(d_sym))
    if p_sym is not None:
        st.latex("F(x)=" + sp.latex(p_sym))

# Exportar PNG
try:
    img = pio.to_image(fig, format="png")
    st.download_button("Descargar gráfica en PNG", img, "grafica.png", "image/png")
except:
    st.info("Instala 'kaleido' para exportar imágenes.")
