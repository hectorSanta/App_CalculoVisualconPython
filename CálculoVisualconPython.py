import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as go
from sympy.parsing.sympy_parser import parse_expr

def limpiar(expr):
    return expr.replace("^", "**").replace("sen", "sin").replace("π", "pi")

x = sp.Symbol("x")
st.title("Cálculo Visual con Python")

entrada = st.text_input("Ingresa f(x):", "x^2")
expr = limpiar(entrada)

xmin = st.slider("x mínimo", -10, 0, -5)
xmax = st.slider("x máximo", 0, 10, 5)

try:
    f = parse_expr(expr)
    f_lamb = sp.lambdify(x, f, "numpy")

    xs = np.linspace(xmin, xmax, 500)
    ys = f_lamb(xs)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="f(x)"))

    st.plotly_chart(fig)
except:
    st.error("Función inválida.")