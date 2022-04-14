import streamlit as st
import numpy as np

from scipy.stats import t, norm, chi2, kstwo, ksone
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
footer {
    visibility:visible
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
st.title("Fundamntal of Statistics")
st.header("Pivotal Tool for Hypothesis Testing")

# "before updating"
# st.session_state

############# Test Statistic Set-up #####################
col_setup_1, col_setup_2, col_setup_3 = st.columns(3)
dist_fml = {
    "norm":"Standard Normal",
    "chi2":"Chi Square",
    "t":"Student T",
    "ks":"Kolmogorov-Smirnov"
}
# Select Distribution
dist_name = col_setup_1.selectbox("distribution".upper(), dist_fml.keys(), format_func=dist_fml.get)
# determine requirement of degree of freedom / sample size
df_map = {
    "norm": False,
    "chi2": True,
    "t": True,
    "ks": True
}
if df_map[dist_name]:
    df_default, df_min = 5, 1
else:
    df_default, df_min = 0, 0

alpha = col_setup_2.selectbox("test signifcant level".upper(), (0.1, .05,.01), 1)
test_type = {0:"two-sided (|Tn| > c)", 1:"one-sided (Tn > c)", 2:"one-sided (Tn < c)"}
one_side = col_setup_3.selectbox("test type".upper(), test_type.keys(), format_func=test_type.get)

# determine distribution variables
if dist_name == "ks":
    dist = ksone if one_side else kstwo
else:
    dist = globals()[dist_name]
# determine level of quantile
if not (one_side or dist_name == "ks"):
    alpha = alpha/2

################ Test Statistic Calculation #######################

col_setting, col_img = st.columns((1,2))

def update_pvalue():
    st.session_state.pvalue = getattr(dist, "ppf")(y1, df).round(2)

def update_cdf():
    st.session_state.cdf = getattr(dist, "cdf")(x1, df).round(4)

with col_setting:
    df_label = "num of trials, n" if dist_name == "ks" else "degree of freedom, df"
    df=st.slider(df_label, df_min,20,df_default,1,disabled=not df_default)
    q_min = getattr(dist, "ppf")(0.001, df).round(2)
    q_max = getattr(dist, "ppf")(0.999, df).round(2)
    if one_side == 2:
        q_a = getattr(dist, "ppf")(alpha, df).round(2)
    else:    
        q_a = getattr(dist, "ppf")(1-alpha, df).round(2)
    # set up default p-value and cdf
    # if "pvalue" not in st.session_state:
    #     st.session_state["pvalue"] = getattr(dist, "ppf")(q_a, df).round(2)
    # if "cdf" not in st.session_state:
    #     st.session_state["cdf"] = getattr(dist, "cdf")(q_a, df).round(4)
    
    x1 = st.number_input(
        'P-Value', q_min, q_max, q_a, 0.01,
        key="pvalue", on_change=update_cdf
        )
    y1 = st.number_input(
        'CDF', 0., 1., getattr(dist, "cdf")(q_a, df).round(4), 0.001, format='%f',
        key="cdf", on_change=update_pvalue)
    # st.metric("pdf", t.pdf(x1, df))
    # st.metric("cdf", f"{t.cdf(x1, df):.2%}" )
    # st.metric("ppf", f"{t.ppf(t.cdf(x1, df), df):.2f}" )

with col_img:
    fig_dist, ax = plt.subplots(1, 1)
    fig_x = np.linspace(q_min, q_max, 100,)
    fig_y = getattr(dist, "pdf")(fig_x, df)
    ax.plot(fig_x, fig_y, 'k-', lw=2, alpha=0.6, label='PDF Curve')
    if not one_side:
        ax.fill_between(fig_x, fig_y, 0, np.abs(fig_x)>q_a, color="r", alpha=0.3, label="Reject Region")
    elif one_side == 1:
        ax.fill_between(fig_x, fig_y, 0, fig_x>q_a, color="r", alpha=0.3, label="Reject Region")
    elif one_side == 2:
        ax.fill_between(fig_x, fig_y, 0, fig_x<q_a, color="r", alpha=0.3, label="Reject Region")
    
    ax.vlines(x1, 0, getattr(dist, "pdf")(x1, df), "b","solid", "p-value", lw=3, alpha=0.5)
    ax.scatter(x1, getattr(dist, "pdf")(x1, df))
    ax.legend(loc="upper right")
    ax.set_title(f"Hypothesis Testing - {dist_fml[dist_name]} Distribution")
    st.pyplot(fig_dist)

# test_stat_string = f"{x1:.2f} \geq {t.ppf(0.95,df):.2f}"
# test_bool = str(bool(x1>=t.ppf(0.95,df)))
col_result_1,col_result_2,col_result_3,col_result_4,col_result_5,col_result_6 = st.columns((3,4,1,4,1,4))
col_result_1.markdown("# $\displaystyle{\Phi \mathbb{1} \{ }$")
col_result_2.metric("value of the test", round(x1,2), (x1-q_a).round(2))
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

# st.latex(
#     "\displaystyle{ \Phi\{T_n>q_a\} \quad = \quad \Phi\ \{"
#      + test_stat_string + "\} \quad = \quad " + test_bool +" }"
# )

# "after updating"
# st.session_state