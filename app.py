import streamlit as st
from matplotlib import pyplot as plt

st.set_page_config(
   page_title="Statistics Tools",
   page_icon="🧰",
   layout="centered",
   initial_sidebar_state="expanded",
)

# Hide MainMenu

hide_menu = """
<style>
#MainMenu {
    visibility:hidden
}
header {
    visibility:hidden
}
footer {
    visibility:hidden
}
footer:after{
    content:"Dev Ver 0.1.0 by Xiaodong Yi";
    display:block;
    position:relative;
    color:brown;
}

</style>
"""
# <p>Developed with ❤ by <a style='display: block; text-align: center;' href="https://www.heflin.dev/" target="_blank">Xiaodong Yi</a></p>
st.markdown(hide_menu, unsafe_allow_html=True)

# Webpage Start

# st.title("Fundamntal of Statistics")

from sympy import *
from sympy.stats import Exponential, Poisson, Gamma, Beta, Normal, Bernoulli, Binomial,Geometric, Uniform
from sympy.stats import density, E, variance, cdf, quantile, Expectation

st.subheader("Basic Properties of Distribution")

x, a, b = symbols("x a b", real=True)
p = Symbol("p", positive=True)
n = Symbol("n", integer=True, positive=True)
l = Symbol("lambda", positive=True)
k = Symbol("k", positive=True)
theta = Symbol("theta", positive=True)
alpha = Symbol("alpha", positive=True)
beta = Symbol("beta", positive=True)
mu = Symbol("mu")
sigma = Symbol("sigma", positive=True)

dist_fml = {
    "Discrete": {
        "Bernoulli":{"num_p":1, "p1":{"var":p, "min":0., "max":1., "default":.5}, "x":{"var":x, "min":0, "max":1}},
        "Geometric":{"num_p":1, "p1":{"var":p, "min":0., "max":1., "default":.5}, "x":{"var":x, "min":0, "max":10}},
        # "Binomial":{"num_p":2, "p1":{"var":n, "min":1., "max":10, "default":3}, 
        #     "p2":{"var":p, "min":0., "max":1., "default":.5}, "x":{"var":x, "min":1, "max":10}},
        "Poisson":{"num_p":1, "p1":{"var":l, "min":0., "max":1., "default":.5},"x":{"var":x, "min":1, "max":10}},
    },
    "Continuous":{ 
        "Exponential":{"num_p":1, "p1":{"var":l, "min":0., "max":5., "default":1.},"x":{"var":x, "min":0, "max":5}}, 
        "Uniform":{"num_p":2, "p1":{"var":a, "min":-5., "max":5., "default":.0}, 
            "p2":{"var":b, "min":0., "max":10., "default":5.}, "x":{"var":x, "min":-5, "max":10}},
        "Normal":{"num_p":2, "p1":{"var":mu, "min":-5., "max":5., "default":.0}, 
            "p2":{"var":sigma, "min":-5., "max":5., "default":1.}, "x":{"var":x, "min":-5, "max":5}},
        "Gamma":{"num_p":2, "p1":{"var":k, "min":0., "max":5., "default":1.}, 
            "p2":{"var":theta, "min":0., "max":5., "default":2.}, "x":{"var":x, "min":0, "max":10}},
        "Beta":{"num_p":2, "p1":{"var":alpha, "min":0., "max":5., "default":2.}, 
            "p2":{"var":beta, "min":0., "max":5., "default":2.}, "x":{"var":x, "min":0, "max":1}},
    }
}

col_dist_1, col_dist_2, col_dist_3 = st.columns(3)
dist_type = col_dist_1.selectbox("", dist_fml.keys(), 1,)
dist_name = col_dist_2.selectbox("", dist_fml.get(dist_type).keys(),)
dist = globals()[dist_name]
discrete = dist_type== "Discrete"
col_dist_3.selectbox("", ["Plotting", "Plotting Disabled"], int(discrete), disabled=True)
with st.expander("Expand to check out pdf, expectation, variance, plotting ..."):
    col_prop, col_fig = st.columns((2,3))


num_p = dist_fml.get(dist_type).get(dist_name).get("num_p")
if num_p == 1:
    p1 = dist_fml.get(dist_type).get(dist_name).get("p1")
    X = dist("X", p1.get("var"))
elif num_p == 2:
    p1 = dist_fml.get(dist_type).get(dist_name).get("p1")
    p2 = dist_fml.get(dist_type).get(dist_name).get("p2")
    X = dist("X", p1.get("var"), p2.get("var"))

if dist_type == "Discrete":
    col_prop.error("Plotting disabled")
else:
    p1_value = col_prop.slider(str(p1["var"]), p1["min"], p1["max"], p1["default"], 0.1)
    if num_p == 2:
        p2_value = col_prop.slider(str(p2["var"]), p2["min"], p2["max"], p2["default"], 0.1)

pdf = density(X).dict if dist_name == "Binomial" else density(X)(x)
prop_latex = """
\\begin{align}
\\text{PDF}  & : \\  {pdf} \\\\
\\text{Mean} & : \\quad {mu} \\\\
\\text{Var}(x) & : \\quad {var} \\\\
\\end{align}
""" .replace("{pdf}", latex(pdf))\
    .replace("{mu}", latex(E(X)))\
    .replace("{var}", latex(factor(variance(X))))

col_prop.latex(prop_latex)

with col_fig:
    if dist_type == "Continuous":
        if num_p == 1:
            plt.style.use("seaborn-whitegrid")
            fig_pdf = plot(
                pdf.subs(p1["var"], p1_value), 
                (x,dist_fml[dist_type][dist_name]["x"]["min"],dist_fml[dist_type][dist_name]["x"]["max"]), 
                show=False, title="Probability Density Function", xlabel="random variable - x", ylabel="P(x)",
                )
        else:
            plt.style.use("seaborn-whitegrid")
            fig_pdf = plot(
                pdf.subs({p1["var"]: p1_value, p2["var"]:p2_value}), 
                (x,dist_fml[dist_type][dist_name]["x"]["min"],dist_fml[dist_type][dist_name]["x"]["max"]), 
                show=False, title="Probability Density Function", xlabel="random variable - x", ylabel="P(x)",
                )

    else:
        plt.style.use("seaborn-whitegrid")
        fig_pdf, ax = plt.subplots()
        ax.text(.2,.5, "pdf plotting is only valid for continuous distribution")
    st.pyplot(fig_pdf.show())

from scipy.stats import t, norm, chi2, kstwo, ksone
import numpy as np

st.subheader("Pivotal Table for Hypothesis Testing")

# "before updating"
# st.session_state

############# Test Statistic Set-up #####################

col_setup_1, col_setup_2, col_setup_3 = st.columns(3)

test_fml = {
    "norm":"Wald (or Z)",
    "chi2":"Chi Square",
    "t":"Student T",
    "ks":"Kolmogorov-Smirnov"
}
# Select Distribution
test_name = col_setup_1.selectbox("Test Name".upper(), test_fml.keys(), format_func=test_fml.get)
# determine requirement of degree of freedom / sample size
df_map = {
    "norm": False,
    "chi2": True,
    "t": True,
    "ks": True
}
if df_map[test_name]:
    df_default, df_min = 5, 1
else:
    df_default, df_min = 0, 0

alpha = col_setup_2.selectbox("test signifcant level".upper(), (0.1, .05,.01), 1)
test_type = {0:"two-sided (|Tn| > c)", 1:"one-sided (Tn > c)", 2:"one-sided (Tn < c)"}
one_side = col_setup_3.selectbox("test type".upper(), test_type.keys(), format_func=test_type.get)

# determine distribution variables
if test_name == "ks":
    test = ksone if one_side else kstwo
else:
    test = globals()[test_name]
# determine level of quantile
if not (one_side or test_name == "ks"):
    alpha = alpha/2

################ Test Statistic Calculation #######################
with st.expander("Expand to check out pivotal table, p-value, test result..."):
    col_setting, col_img = st.columns((1,2))
    col_result_1,col_result_2,col_result_3,col_result_4,col_result_5,col_result_6 = st.columns((3,4,1,4,1,5))

def update_pvalue():
    st.session_state.pvalue = getattr(test, "ppf")(y1, df).round(2)

def update_cdf():
    st.session_state.cdf = getattr(test, "cdf")(x1, df).round(4)

with col_setting:
    df_label = "num of trials, n" if test_name == "ks" else "degree of freedom, df"
    df=st.slider(df_label, df_min,20,df_default,1,disabled=not df_default)
    q_min = getattr(test, "ppf")(0.001, df).round(2)
    q_max = getattr(test, "ppf")(0.999, df).round(2)
    if one_side == 2:
        q_a = getattr(test, "ppf")(alpha, df).round(2)
    else:    
        q_a = getattr(test, "ppf")(1-alpha, df).round(2)
    # set up default p-value and cdf
    # if "pvalue" not in st.session_state:
    #     st.session_state["pvalue"] = getattr(test, "ppf")(q_a, df).round(2)
    # if "cdf" not in st.session_state:
    #     st.session_state["cdf"] = getattr(test, "cdf")(q_a, df).round(4)
    
    x1 = st.number_input(
        'P-Value', q_min, q_max, q_a, 0.01,
        key="pvalue", on_change=update_cdf
        )
    y1 = st.number_input(
        'CDF', 0., 1., getattr(test, "cdf")(q_a, df).round(4), 0.001, format='%f',
        key="cdf", on_change=update_pvalue)

with col_img:
    fig_dist, ax = plt.subplots(1, 1)
    fig_x = np.linspace(q_min, q_max, 100,)
    fig_y = getattr(test, "pdf")(fig_x, df)
    ax.plot(fig_x, fig_y, 'k-', lw=1, alpha=0.6, label='PDF Curve')
    if not one_side:
        ax.fill_between(fig_x, fig_y, 0, np.abs(fig_x)>q_a, color="r", alpha=0.3, label="Reject Region")
    elif one_side == 1:
        ax.fill_between(fig_x, fig_y, 0, fig_x>q_a, color="r", alpha=0.3, label="Reject Region")
    elif one_side == 2:
        ax.fill_between(fig_x, fig_y, 0, fig_x<q_a, color="r", alpha=0.3, label="Reject Region")
    
    ax.vlines(x1, 0, getattr(test, "pdf")(x1, df), "b","solid", "p-value", lw=3, alpha=0.5)
    ax.scatter(x1, getattr(test, "pdf")(x1, df))
    ax.legend(loc="upper right")
    ax.set_title(f"Hypothesis Testing - {test_fml[test_name]} Distribution")
    st.pyplot(fig_dist)


col_result_1.markdown("# $\displaystyle{\Phi \mathbb{1} \{ }$")
if one_side:
    col_result_2.metric("value of the test", round(x1,2), (x1-q_a).round(2))
else:
    col_result_2.metric("value of the test", round(abs(x1),2), (abs(x1)-q_a).round(2))
if one_side == 2:
    col_result_3.markdown("#  < ")
else:
    col_result_3.markdown("#  > ")
col_result_4.metric("quantile at alpha", q_a.round(2))
col_result_5.markdown("# $ \} $ ")
if one_side == 2 and (x1<q_a):
    col_result_6.markdown("# $\color{red}{Reject} \  H_0$ ")
elif x1>q_a:
    col_result_6.markdown("# $\color{red}{ Reject } \ H_0$ ")
else:
    col_result_6.markdown("### Failed to reject $H_0$")

footer_markdown = """
> look-up tool (Beta Ver 0.1.0) for [Fundamental of Statistics](https://learning.edx.org/course/course-v1:MITx+18.6501x+1T2022/home)
"""
st.markdown(footer_markdown)
# "after updating"
# st.session_state