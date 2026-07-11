import streamlit as st
import pandas as pd

# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Retail Demand Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# Load CSS
# ==========================================

def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ==========================================
# Load Dataset
# ==========================================

@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\Admin\OneDrive\Desktop\Retail_demand_streamlit\retail_dataset_v2.csv")
    return df

df = load_data()

# ==========================================
# Sidebar
# ==========================================

st.sidebar.markdown("<h1 style='text-align:center;'>🏪 Retail BI</h1>", unsafe_allow_html=True)

st.sidebar.markdown("---")

store = st.sidebar.selectbox(
    "🏬 Select Store",
    ["All"] + sorted(df["Store"].unique())
)

department = st.sidebar.selectbox(
    "🏢 Select Department",
    ["All"] + sorted(df["Department"].unique())
)

# Dynamic Category Filter

if department == "All":
    categories = ["All"] + sorted(df["Category"].unique())
else:
    categories = ["All"] + sorted(
        df[df["Department"] == department]["Category"].unique()
    )

category = st.sidebar.selectbox(
    "📦 Select Category",
    categories
)

# Product Search

search = st.sidebar.text_input(
    "🔍 Search Product"
)

st.sidebar.markdown("---")

st.sidebar.success("Dashboard Ready")

# ==========================================
# Apply Filters
# ==========================================

filtered_df = df.copy()

if store != "All":
    filtered_df = filtered_df[
        filtered_df["Store"] == store
    ]

if department != "All":
    filtered_df = filtered_df[
        filtered_df["Department"] == department
    ]

if category != "All":
    filtered_df = filtered_df[
        filtered_df["Category"] == category
    ]

if search:
    filtered_df = filtered_df[
        filtered_df["Product_Name"].str.contains(
            search,
            case=False
        )
    ]

# ==========================================
# Header
# ==========================================

st.markdown("""

<div class="header">

<h1>📊 Retail Demand Forecasting & Inventory Optimization</h1>

<p>AI Powered Business Intelligence Dashboard</p>

</div>

""", unsafe_allow_html=True)

# ==========================================
# KPI Calculations
# ==========================================

total_sales = filtered_df["Sales"].sum()

inventory = filtered_df["Inventory"].sum()

avg_price = filtered_df["Price"].mean()

forecast = filtered_df["Demand_Forecast"].mean()

products = filtered_df["Product_ID"].nunique()

stores = filtered_df["Store"].nunique()

# ==========================================
# KPI Cards
# ==========================================

c1,c2,c3 = st.columns(3)

with c1:

    st.markdown(f"""

<div class="card">

<h4>💰 Total Sales</h4>

<h2>{total_sales:,.0f}</h2>

</div>

""",unsafe_allow_html=True)

with c2:

    st.markdown(f"""

<div class="card">

<h4>📦 Inventory</h4>

<h2>{inventory:,.0f}</h2>

</div>

""",unsafe_allow_html=True)

with c3:

    st.markdown(f"""

<div class="card">

<h4>💲 Average Price</h4>

<h2>₹ {avg_price:.2f}</h2>

</div>

""",unsafe_allow_html=True)

st.write("")

c4,c5,c6 = st.columns(3)

with c4:

    st.markdown(f"""

<div class="card2">

<h4>📈 Forecast</h4>

<h2>{forecast:.0f}</h2>

</div>

""",unsafe_allow_html=True)

with c5:

    st.markdown(f"""

<div class="card2">

<h4>🛍 Products</h4>

<h2>{products}</h2>

</div>

""",unsafe_allow_html=True)

with c6:

    st.markdown(f"""

<div class="card2">

<h4>🏪 Stores</h4>

<h2>{stores}</h2>

</div>

""",unsafe_allow_html=True)

st.divider()

# ==========================================
# Quick Summary
# ==========================================

left,right = st.columns([2,1])

with left:

    st.subheader("📋 Retail Dataset")

    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=500
    )

with right:

    st.subheader("📌 Business Summary")

    st.info(f"""
Total Products : {products}

Stores : {stores}

Average Price : ₹{avg_price:.2f}

Average Forecast : {forecast:.0f}

Current Department : {department}

Current Category : {category}
""")

    st.success("✔ Dashboard Updated")

# ==========================================
# Download
# ==========================================

st.download_button(

"📥 Download Filtered Report",

filtered_df.to_csv(index=False),

file_name="Retail_Report.csv",

mime="text/csv"

)

st.markdown("---")

st.markdown("""

<center>

Developed by <b>Sohel Shaikh</b>

<br>

Retail Demand Forecasting & Inventory Optimization

</center>

""",unsafe_allow_html=True)






