import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import base64
import time

APP_TITLE = "E-commerce Price Optimization Dashboard"  # No emoji in title
PROJECT_FOOTER = "© 2025 Price Optimization Project | Author: Dhansuh"

THEME = {
    "bg_color": "#FFFFFF",
    "text_color": "#222222",
    "card_bg": "#FFFFFF",
    "card_shadow": "rgba(0, 0, 0, 0.08)",
    "hover_shadow": "rgba(0, 0, 0, 0.15)",
    "sidebar_bg": "linear-gradient(180deg, #6a11cb 0%, #2575fc 100%)",
    "sidebar_text": "#FFFFFF",
    "sidebar_hover_bg": "rgba(255,255,255,0.2)",
}


@st.cache_data
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [col.strip() for col in df.columns]
    return df


def filter_data(df, product_ids, warehouses, product_importances, genders):
    if product_ids:
        df = df[df['ID'].isin(product_ids)]
    if warehouses:
        df = df[df['Warehouse_block'].isin(warehouses)]
    if product_importances:
        df = df[df['Product_importance'].isin(product_importances)]
    if genders:
        df = df[df['Gender'].isin(genders)]
    return df


def calculate_optimal_price(df, discount_percentage):
    df_opt = df.groupby('ID').agg(
        avg_cost=('Cost_of_the_Product', 'mean'),
        avg_prior_purchases=('Prior_purchases', 'mean'),
        avg_discount=('Discount_offered', 'mean')
    ).reset_index()

    df_opt['Optimal_Price'] = df_opt['avg_cost'] * (1 - (discount_percentage / 100))
    df_opt['Potential_Revenue'] = df_opt['Optimal_Price'] * df_opt['avg_prior_purchases']
    df_opt = df_opt.sort_values(by='Potential_Revenue', ascending=False)
    return df_opt


def highlight_top5(row):
    if row.name < 5:
        return ['background: linear-gradient(90deg, #FFD700, #FFA500)']*len(row)
    return ['']*len(row)


def to_csv_download_link(df, filename, label):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{label}</a>'
    return href


def animated_counter(id_str, value, label, duration=1500):
    place = st.empty()
    step = max(1, int(value/50))
    interval = duration / 50 / 1000
    for val in range(0, int(value)+step, step):
        if val > value:
            val = value
        place.metric(label, f"{val:,}")
        time.sleep(interval)
    if value == 0:
        place.metric(label, f"0")
    return place


def apply_bg_and_sidebar_style(theme):
    style = f"""
    <style>
        .stApp {{
            background: {theme['bg_color']} !important;
            color: {theme['text_color']} !important;
        }}

        /* Sidebar style */
        .sidebar .sidebar-content {{
            background: {theme['sidebar_bg']} !important;
            color: {theme['sidebar_text']} !important;
            font-weight: 600;
        }}

        .sidebar-logo-container {{
            background: #4F46E5;
            padding: 15px 5px;
            border-radius: 12px;
            margin-bottom: 20px;
            text-align: center;
            color: white;
            font-weight: 700;
            font-size: 18px;
        }}

        .stButton>button {{
            background: linear-gradient(to right, #6a11cb, #2575fc);
            color: white;
            font-weight: 600;
            border-radius: 10px;
            padding: 0.6rem 1.2rem;
            border: none;
            transition: 0.3s ease;
        }}
        .stButton>button:hover {{
            background: linear-gradient(to right, #2575fc, #6a11cb);
            cursor: pointer;
        }}

        /* --- YouTube-style underline tabs --- */
        .stTabs [data-baseweb="tab-list"] {{
            position: relative;
            display: flex;
            border-bottom: 2px solid #e0e0e0;
            margin-bottom: 10px;
        }}

        .stTabs [data-baseweb="tab"] {{
            position: relative;
            padding: 8px 16px;
            font-weight: 500;
            transition: color 0.3s ease;
        }}

        .stTabs [data-baseweb="tab-list"]::after {{
            content: "";
            position: absolute;
            bottom: -2px;
            left: 0;
            height: 3px;
            width: var(--underline-width, 0);
            transform: translateX(var(--underline-x, 0));
            background: linear-gradient(90deg, #4CAF50, #2196F3);
            border-radius: 2px;
            transition: transform 0.3s ease, width 0.3s ease;
        }}
    </style>
    """
    st.markdown(style, unsafe_allow_html=True)

    # Inject JavaScript for underline animation
    st.markdown("""
    <script>
    const waitForTabs = setInterval(() => {
        const tabList = document.querySelector('[data-baseweb="tab-list"]');
        if (tabList) {
            clearInterval(waitForTabs);
            const tabs = tabList.querySelectorAll('[data-baseweb="tab"]');

            function updateUnderline() {
                const active = tabList.querySelector('[aria-selected="true"]');
                if (active) {
                    const rect = active.getBoundingClientRect();
                    const parentRect = tabList.getBoundingClientRect();
                    tabList.style.setProperty('--underline-width', rect.width + 'px');
                    tabList.style.setProperty('--underline-x', (rect.left - parentRect.left) + 'px');
                }
            }
            tabs.forEach(tab => tab.addEventListener('click', () => {
                setTimeout(updateUnderline, 200);
            }));
            updateUnderline();
        }
    }, 300);
    </script>
    """, unsafe_allow_html=True)


def render_logo():
    logo_html = """
    <div class="sidebar-logo-container">
        <img src="https://cdn-icons-png.flaticon.com/512/34/34568.png" alt="E-commerce Logo" style="width:60px;height:60px;"/>
        <div>E-commerce Price Optimization</div>
        <small style="font-weight:400; font-size:0.9rem; color:#D1D5DB;">
          Make smart pricing decisions 💡
        </small>
    </div>
    """
    st.sidebar.markdown(logo_html, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🛒",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={'About': "Interactive Price Optimization dashboard for E-commerce products."}
    )

    apply_bg_and_sidebar_style(THEME)
    render_logo()

    # Sidebar
    with st.sidebar.expander("Upload Dataset & Filters", expanded=True):
        uploaded_file = st.file_uploader(
            "Upload your CSV file with required columns 👇",
            type=["csv"],
        )
        if not uploaded_file:
            st.info("Please upload a CSV file to start analyzing.")
            st.stop()
        df = load_data(uploaded_file)

        product_ids = st.multiselect("Filter by Product ID", options=sorted(df['ID'].unique()))
        warehouses = st.multiselect("Filter by Warehouse Block", options=sorted(df['Warehouse_block'].unique()))
        product_importances = st.multiselect("Filter by Product Importance", options=sorted(df['Product_importance'].unique()))
        genders = st.multiselect("Filter by Gender", options=sorted(df['Gender'].unique()))
        discount_percent = st.slider("Adjust Discount Percentage (%)", 0, 50, 10, 1)

    filtered_df = filter_data(df, product_ids, warehouses, product_importances, genders).copy()

    total_sales = filtered_df['Prior_purchases'].sum()
    total_revenue = (filtered_df['Cost_of_the_Product'] * filtered_df['Prior_purchases']).sum()
    total_discount = filtered_df['Discount_offered'].sum()
    avg_revenue = total_revenue / max(len(filtered_df['ID'].unique()), 1)

    # Tabs
    tabs = st.tabs(["Dataset Preview", "Dashboard", "Optimal Pricing", "Heatmaps", "About"])

    # --- Dataset Preview ---
    with tabs[0]:
        st.markdown("### Dataset Preview")
        st.dataframe(filtered_df.reset_index(drop=True), height=400)

    # --- Dashboard ---
    with tabs[1]:
        st.markdown("## Performance KPIs")
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            animated_counter("total_sales", total_sales if total_sales > 0 else 0, "Total Sales (Units)")
        with k2:
            animated_counter("total_revenue", int(total_revenue), f"Total Revenue (₹)")
        with k3:
            animated_counter("total_discount", int(total_discount), "Total Discount (%)")
        with k4:
            animated_counter("avg_revenue", int(avg_revenue), f"Avg Revenue per Product (₹)")

    # --- Optimal Pricing ---
    with tabs[2]:
        st.markdown("## Optimal Pricing Suggestions")
        df_opt = calculate_optimal_price(filtered_df, discount_percent)
        st.dataframe(df_opt.style.apply(highlight_top5, axis=1), height=400)

        st.markdown(to_csv_download_link(df_opt, "optimal_pricing.csv", "📥 Download Optimal Pricing CSV"), unsafe_allow_html=True)

    # --- Heatmaps ---
    with tabs[3]:
        st.markdown("## Heatmaps")
        if not filtered_df.empty:
            fig1 = px.density_heatmap(filtered_df, x="Discount_offered", y="Cost_of_the_Product",
                                      title="Discount vs Cost Heatmap", nbinsx=20, nbinsy=20, color_continuous_scale="Blues")
            st.plotly_chart(fig1, use_container_width=True)

            fig2 = px.density_heatmap(filtered_df, x="Product_importance", y="Prior_purchases",
                                      title="Importance vs Purchases Heatmap", color_continuous_scale="Viridis")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No data available to display heatmaps.")

    # --- About ---
    with tabs[4]:
        st.markdown("## About This App")
        st.markdown(
            """
            Welcome to the 🛒 **E-commerce Price Optimization Dashboard**! This app helps explore sales trends, prices, customer ratings, and optimal pricing strategies for your products.

            Features include:
            - Clean, stylish UI with vibrant sidebar and Indian Rupee currency symbol ₹.
            - Interactive charts with Plotly.
            - Animated KPIs for instant insights.
            - Download filtered data and recommendations.
            - Responsive and user-friendly design.

        Developed by **Dhansuh  and team...(VIT AP)**  
        This dashboard helps in analyzing product sales, revenue, and suggesting optimal prices using discount strategies.
        """)

    st.markdown("---")
    st.markdown(f"<p style='text-align:center;color:#888;font-size:0.9rem;'>{PROJECT_FOOTER}</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
