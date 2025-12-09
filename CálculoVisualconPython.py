import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as go
from sympy.parsing.sympy_parser import parse_expr
import random


def limpiar(expr):
    return (
        expr.replace("^", "**")
            .replace("sen", "sin")
            .replace("π", "pi")
    )

x = sp.Symbol("x")

# ---------------------------
# Título
# ---------------------------
st.title("Cálculo Visual con Python")

# ---------------------------
# Entrada de función
# ---------------------------
entrada = st.text_input("Ingresa f(x):", "sin(x) + x**2")
expr = limpiar(entrada)

xmin = st.slider("x mínimo", -10, 0, -5)
xmax = st.slider("x máximo", 0, 10, 5)

try:
    # ---------------------------
    # Crear función simbólica
    # ---------------------------
    f = parse_expr(expr)
    st.latex("f(x)=" + sp.latex(f))

    f_lamb = sp.lambdify(x, f, "numpy")

    # Derivada
    der = sp.diff(f, x)
    st.latex("f'(x)=" + sp.latex(der))
    der_lamb = sp.lambdify(x, der, "numpy")

    # Integral
    F = sp.integrate(f, x)
    st.latex("F(x)=" + sp.latex(F))
    F_lamb = sp.lambdify(x, F, "numpy")

    # ---------------------------
    # Rango
    # ---------------------------
    xs = np.linspace(xmin, xmax, 600)
    ys = f_lamb(xs)
    ys_der = der_lamb(xs)
    ys_F = F_lamb(xs)

    # ---------------------------
    # Figura
    # ---------------------------
    fig = go.Figure()

    # f(x)
    fig.add_trace(go.Scatter(
        x=xs, y=ys, name="f(x)", line=dict(width=3)
    ))

    # f'(x)
    fig.add_trace(go.Scatter(
        x=xs, y=ys_der, name="f'(x)",
        line=dict(color="red", dash="dash")
    ))

    # F(x)
    fig.add_trace(go.Scatter(
        x=xs, y=ys_F, name="F(x)",
        line=dict(color="green", dash="dot")
    ))

    # ---------------------------
    # Punto móvil
    # ---------------------------
    if "x0" not in st.session_state:
        st.session_state.x0 = 0.0

    st.subheader("Control del punto")

    x0 = st.slider(
        "Posición del punto:",
        float(xmin), float(xmax), st.session_state.x0
    )

    st.session_state.x0 = x0

    y0 = float(f_lamb(x0))
    dy0 = float(der_lamb(x0))

    # punto
    fig.add_trace(go.Scatter(
        x=[x0], y=[y0], mode="markers+text",
        name="Punto",
        marker=dict(size=10, color="orange"),
        text=["x0"],
        textposition="top center"
    ))

    # ---------------------------
    # Tangente
    # ---------------------------
    xt = np.linspace(x0 - 2, x0 + 2, 200)
    yt = dy0 * (xt - x0) + y0

    fig.add_trace(go.Scatter(
        x=xt, y=yt, name="Tangente",
        line=dict(color="orange")
    ))

    # ---------------------------
    # Área bajo la curva
    # ---------------------------
    st.subheader("Área bajo la curva")

    a = st.number_input("Límite a", value=-1.0)
    b = st.number_input("Límite b", value=1.0)

    if a < b:
        mask = (xs >= a) & (xs <= b)
        fig.add_trace(go.Scatter(
            x=np.concatenate([[a], xs[mask], [b]]),
            y=np.concatenate([[0], ys[mask], [0]]),
            fill="toself",
            fillcolor="rgba(0,255,0,0.2)",
            name="Área"
        ))

        area_val = float(sp.integrate(f, (x, a, b)))
        st.success(f"Área entre {a} y {b}: {area_val:.4f}")

    # ---------------------------
    # Mostrar gráfica
    # ---------------------------
    fig.update_layout(
        template="plotly_white",
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # Animación del punto
    # ---------------------------
    st.subheader("Animación del punto")

    if "anim" not in st.session_state:
        st.session_state.anim = False

    if st.button("Iniciar animación"):
        st.session_state.anim = True

    if st.button("Detener animación"):
        st.session_state.anim = False

    if st.session_state.anim:
        st.session_state.x0 += 0.1
        if st.session_state.x0 > xmax:
            st.session_state.x0 = xmin
        st.experimental_rerun()

except Exception as e:
    st.error("La función no es válida.")
    st.exception(e)
