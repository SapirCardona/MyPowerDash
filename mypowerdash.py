import streamlit as st
import pandas as pd
import io
import csv
import numpy as np
import plotly.express as px
import requests
from bs4 import BeautifulSoup
import re

st.set_page_config(
    page_title="⚡ MyPowerDash",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
    .highlight-box {
        background-color: #f5f5f5;
        border-left: 6px solid #ff9900;
        padding: 1rem;
        margin: 2rem 0;
        border-radius: 10px;
    }
    .highlight-box h4 {
        margin: 0 0 0.5rem 0;
        color: #ff9900;
        text-align: center;
        font-size: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    [data-baseweb="tab-list"] > div > button {
        font-size: 1.5rem !important;
        font-weight: bold !important;
        padding: 0.75rem 1.5rem !important;
    }

    [data-baseweb="tab-list"] > div > button span {
        font-size: 2rem !important;
        margin-right: 0.5rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    div[data-testid="stNumberInput"] input {
        font-size: 1.5rem !important;
        font-weight: bold !important;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .block-container h3 {
        text-align: center;
        font-size: 1.8rem !important;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Page setup
st.title("⚡ MyPowerDash – Electricity Consumption Insights")

# Step 1: Upload file
uploaded_file = st.file_uploader("Upload your electricity usage CSV file", type=["csv"])
st.caption("🔒 Your file is processed locally and not stored on our servers. No personal data is saved.")

if uploaded_file is not None:
    # Step 2: Read file content and extract metadata
    content = uploaded_file.read().decode("utf-8")
    lines = content.splitlines()

    name_line = lines[3]
    meter_line = lines[7]
    name_values = next(csv.reader([name_line]))
    meter_values = next(csv.reader([meter_line]))

    def clean(val):
        return val.strip().replace('"', '').replace('\u200f', '').replace(',', '')

    name = clean(name_values[0])
    address = clean(name_values[1])
    meter_code = clean(meter_values[0])
    meter_number = clean(meter_values[1])

    df_start = next(i for i, line in enumerate(lines) if "תאריך" in line)
    df = pd.read_csv(io.StringIO("\n".join(lines[df_start:])))
    df.columns = ['Date', 'Time', 'kWh']
    df['kWh'] = df['kWh'].astype(str).str.replace('"', '').str.strip().replace('', np.nan).astype(float)
    df = df.dropna(subset=['Date', 'Time'])
    df = df[~df['Date'].astype(str).str.strip().eq('')]
    df = df[~df['Time'].astype(str).str.strip().eq('')]
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
    df = df[['Datetime', 'kWh']].sort_values('Datetime')

    # Enrich dataframe with time-based columns
    df['Month'] = df['Datetime'].dt.month
    df['Weekday'] = df['Datetime'].dt.day_name()
    df['Hour'] = df['Datetime'].dt.hour
    df['WeekdayNum'] = df['Datetime'].dt.weekday

    from streamlit_option_menu import option_menu

    selected_tab = option_menu(
        menu_title=None,
        options=["Data Overview", "Smart Insights", "Cost & Comparison"],
        icons=["bar-chart-line", "lightbulb", "cash"],
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important"},
            "nav-link": {"font-size": "1.5rem", "font-weight": "bold"},
            "icon": {"font-size": "2rem"},
        }
    )
    # ---------------- TAB 1 ----------------
    if selected_tab == "Data Overview":
        with st.container():
            st.subheader("User Information")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Name", name)
            col2.metric("Address", address)
            col3.metric("Meter Code", meter_code)
            col4.metric("Meter Number", meter_number)

            st.markdown("---")

            st.subheader("Consumption Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Consumption (kWh)", f"{df['kWh'].sum():,.0f}")
            col2.metric("Avg Monthly (kWh)", f"{df.resample('M', on='Datetime')['kWh'].sum().mean():,.0f}")
            col3.metric("Avg Daily (kWh)", f"{df.resample('D', on='Datetime')['kWh'].sum().mean():,.1f}")

            st.markdown("---")

            st.subheader("Monthly Consumption")
            monthly = df.resample('M', on='Datetime')['kWh'].sum().reset_index()
            monthly['Month'] = monthly['Datetime'].dt.strftime('%b %Y')
            fig_month = px.line(monthly, x='Month', y='kWh', markers=True)
            fig_month.update_layout(title='Monthly Electricity Consumption', xaxis_tickangle=-45)
            st.plotly_chart(fig_month, use_container_width=True, key="plot_monthly_line")

            st.markdown("---")

            st.subheader("Weekday Consumption")
            weekdays_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
            weekday_avg = df.groupby('Weekday')['kWh'].mean().reindex(weekdays_order).reset_index()
            fig_week = px.bar(weekday_avg, x='Weekday', y='kWh')
            fig_week.update_layout(title='Average Daily Consumption by Weekday', xaxis_tickangle=-45)
            st.plotly_chart(fig_week, use_container_width=True, key="plot_weekday_bar")

    # ---------------- TAB 2 ----------------
    elif selected_tab == "Smart Insights":

        st.subheader("Smart Tips")

        morning = df[df['Hour'].between(6, 11)]['kWh'].mean()
        noon = df[df['Hour'].between(12, 16)]['kWh'].mean()
        evening = df[df['Hour'].between(17, 21)]['kWh'].mean()
        night = df[(df['Hour'] >= 22) | (df['Hour'] <= 5)]['kWh'].mean()

        max_block = max(morning, noon, evening, night)
        if max_block == morning:
            st.write("Tip: Your highest usage is in the morning. Consider a plan optimized for daytime use.")
        elif max_block == noon:
            st.write("Tip: Your highest usage is around noon. You might benefit from a midday rate plan.")
        elif max_block == evening:
            st.write("Tip: Your highest usage is in the evening. Consider checking peak-hour plans.")
        elif max_block == night:
            st.write("Tip: Your highest usage is at night. A night-focused plan may reduce your costs.")

        weekdays_avg = df[df['WeekdayNum'].between(0, 4)]['kWh'].mean()
        weekend_avg = df[df['WeekdayNum'] >= 5]['kWh'].mean()

        if weekend_avg > weekdays_avg:
            st.write("Tip: Weekend usage is higher. Check weekend-optimized plans with your provider.")
        else:
            st.write("Tip: Weekday usage is higher. A business-day plan might be more cost-effective.")

        summer_avg = df[df['Month'].isin([6, 7, 8])]['kWh'].mean()
        winter_avg = df[df['Month'].isin([12, 1, 2])]['kWh'].mean()
        overall_avg = df['kWh'].mean()

        if summer_avg > overall_avg * 1.2:
            st.write(
                "Tip: Summer usage is significantly higher. Clean your AC filters and avoid setting it on very low temperatures.")

        if winter_avg > overall_avg * 1.2:
            st.write(
                "Tip: Winter usage is high. Consider using gas or solar heating and reduce electric heater use. Heat water during off-peak times if using electric boilers.")

        st.markdown("---")
        with st.container():
            st.subheader("Heatmap: Weekday × Month")
            heat_df = df.copy()
            heat_df['MonthDate'] = heat_df['Datetime'].dt.to_period('M').dt.to_timestamp()
            heat_df['MonthLabel'] = heat_df['MonthDate'].dt.strftime('%b %Y')
            data = heat_df.groupby(['Weekday', 'MonthDate', 'MonthLabel'])['kWh'].mean().reset_index()
            left_col, right_col = st.columns([1, 4])

            with left_col:
                with st.expander("🗓️ Select Months", expanded=False):
                    all_months_sorted = data[['MonthDate', 'MonthLabel']].drop_duplicates().sort_values('MonthDate')
                    available_months = all_months_sorted['MonthLabel'].tolist()
                    selected_months = st.multiselect("Months:", available_months, default=available_months,
                                                     label_visibility="collapsed")

            data = data[data['MonthLabel'].isin(selected_months)]
            month_order = all_months_sorted[all_months_sorted['MonthLabel'].isin(selected_months)][
                'MonthLabel'].tolist()
            weekday_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
            data['MonthLabel'] = pd.Categorical(data['MonthLabel'], categories=month_order, ordered=True)
            data['Weekday'] = pd.Categorical(data['Weekday'], categories=weekday_order, ordered=True)
            data = data.sort_values(['MonthDate', 'Weekday'])
            pivot = data.pivot(index='Weekday', columns='MonthLabel', values='kWh')
            pivot = pivot.astype(float)

            with right_col:
                fig = px.imshow(pivot, labels=dict(x="Month", y="Weekday", color="kWh"), aspect="auto",
                                color_continuous_scale='YlOrRd')
                fig.update_layout(title="Avg. Electricity Consumption by Weekday and Month")
                st.plotly_chart(fig, use_container_width=True, key="plot_weekday_month")

            st.markdown("---")
            st.subheader("Heatmap: Weekday × Hour Block")
            heat_df['Hour Block'] = df['Hour'].apply(lambda h: f"{(h // 2) * 2:02d}:00–{((h // 2) * 2) + 2:02d}:00")
            data = heat_df.groupby(['Weekday', 'Hour Block'])['kWh'].mean().reset_index()

            left_col, right_col = st.columns([1, 4])
            with left_col:
                with st.expander("⚙️ Filter Heatmap", expanded=False):
                    selected_days = st.multiselect("Weekdays:", weekday_order, default=weekday_order,
                                                   label_visibility="collapsed")
                    selected_blocks = st.multiselect("Hour blocks:",
                                                     [f"{h:02d}:00–{h + 2:02d}:00" for h in range(0, 24, 2)],
                                                     default=[f"{h:02d}:00–{h + 2:02d}:00" for h in range(0, 24, 2)],
                                                     label_visibility="collapsed")

            data = data[data['Weekday'].isin(selected_days)]
            data = data[data['Hour Block'].isin(selected_blocks)]
            data['Weekday'] = pd.Categorical(data['Weekday'], categories=weekday_order, ordered=True)
            data['Hour Block'] = pd.Categorical(data['Hour Block'],
                                                categories=[f"{h:02d}:00–{h + 2:02d}:00" for h in range(0, 24, 2)],
                                                ordered=True)
            data = data.sort_values(['Weekday', 'Hour Block'])
            pivot = data.pivot(index='Weekday', columns='Hour Block', values='kWh').astype(float)

            with right_col:
                fig = px.imshow(pivot, labels=dict(x="Hour Block", y="Weekday", color="kWh"), aspect="auto",
                                color_continuous_scale='YlOrRd')
                fig.update_layout(title="Avg. Electricity Consumption by Weekday and 2-Hour Block")
                st.plotly_chart(fig, use_container_width=True, key="plot_weekday_hourblock")


    # ---------------- TAB 3 ----------------
    elif selected_tab == "Cost & Comparison":
        with st.container():
            st.subheader("Electricity Cost Estimation")

            # Try to fetch the actual electricity rate from IEC site
            try:
                url = "https://www.iec.co.il/content/tariffs/contentpages/homeelectricitytariff"
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                text = soup.get_text()
                match = re.search(r"כולל מע\"מ.*?(\d+\.\d+)", text)
                if match:
                    rate = float(match.group(1)) / 100  # Convert agorot to shekels
                else:
                    rate = 0.6402
            except:
                rate = 0.6402


            col_spacer_left, col_center, col_spacer_right = st.columns([1, 2, 1])

            with col_center:
                st.markdown("### Tariff per kWh (₪)", unsafe_allow_html=True)
                user_tariff = st.number_input(
                    label="",
                    value=rate,
                    step=0.01,
                    key="tariff_input_centered"
                )
                st.caption(
                    "Default tariff based on the current IEC residential rate (including VAT). You can adjust it manually."
                )

            st.markdown("---")

            # Monthly cost calculation
            monthly_costs = df.resample('M', on='Datetime')['kWh'].sum().reset_index()
            monthly_costs['Month'] = monthly_costs['Datetime'].dt.strftime('%b %Y')
            monthly_costs['Estimated Cost (₪)'] = monthly_costs['kWh'] * user_tariff

            st.subheader("Monthly Estimated Costs")
            fig_cost = px.bar(
                monthly_costs,
                x='Month',
                y='Estimated Cost (₪)',
                text_auto='.2f'
            )
            fig_cost.update_layout(
                title='Estimated Monthly Electricity Cost',
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_cost, use_container_width=True)

            total_kwh = monthly_costs['kWh'].sum()
            total_cost = monthly_costs['Estimated Cost (₪)'].sum()

            col1, col2 = st.columns(2)
            col1.metric("Total Annual Consumption (kWh)", f"{total_kwh:,.0f}")
            col2.metric("Estimated Annual Cost (₪)", f"₪{total_cost:,.2f}")

            st.caption(
                "Note: The cost calculation does not include the fixed distribution and supply charges, "
                "which may add ~₪30/month depending on your connection type."
            )

            st.markdown("---")

            st.markdown(
                """
                <div class="highlight-box">
                    <h4>🏷️ Better Deal? Personalized Recommendations</h4>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Step 1: Define time slot masks
            masks = {
                "Weekend (Fri-Sat) (%)": df['Weekday'].isin(['Friday', 'Saturday']),
                "Daily 17:00–23:00 (%)": df['Hour'].between(17, 22),
                "Daily 14:00–20:00 (%)": df['Hour'].between(14, 20),
                "Weekday 23:00–07:00 (%)": (
                                                   (df['Hour'] >= 23) | (df['Hour'] <= 6)
                                           ) & df['Weekday'].isin(['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday']),
                "Weekday 07:00–17:00 (%)": (
                                               df['Hour'].between(7, 16)
                                           ) & df['Weekday'].isin(['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday'])
            }
            block_kwh = {name: df.loc[mask, 'kWh'].sum() for name, mask in masks.items()}
            total_iec_cost = df['kWh'].sum() * user_tariff

            # Step 2: Discount matrix for suppliers
            discounts_data = {
                "Supplier": ["Heshmal Yashir", "Electra", "Cellcom Energy", "Power", "Bezeq Energy", "HOT Energy"],
                "Weekend (Fri-Sat) (%)": [0, 0, 0, 0, 0, 0],
                "Daily 17:00–23:00 (%)": [0, 10, 0, 0, 0, 0],
                "Daily 14:00–20:00 (%)": [0, 0, 18, 0, 0, 0],
                "Weekday 23:00–07:00 (%)": [0, 20, 20, 20, 20, 20],
                "Weekday 07:00–17:00 (%)": [0, 0, 15, 15, 15, 15],
                "Constant Discount (%)": [7, 7, 5, 5, 6, 6]
            }
            df_discounts = pd.DataFrame(discounts_data)

            # Step 3: Calculate cheapest plan for each supplier
            results = []
            for _, row in df_discounts.iterrows():
                supplier = row['Supplier']
                cost_options = []
                for discount_type in row.index[1:]:
                    discount = row[discount_type]
                    if discount == 0:
                        continue
                    if discount_type == "Constant Discount (%)":
                        est_cost = df['kWh'].sum() * user_tariff * (1 - discount / 100)
                    else:
                        est_cost = 0
                        for block, kwh in block_kwh.items():
                            block_discount = discount if block == discount_type else 0
                            rate_disc = user_tariff * (1 - block_discount / 100)
                            est_cost += kwh * rate_disc
                    cost_options.append((discount_type, discount, round(est_cost, 2)))

                if cost_options:
                    best_plan = min(cost_options, key=lambda x: x[2])
                    results.append({
                        "Supplier": supplier,
                        "Best Plan": best_plan[0],
                        "Discount (%)": best_plan[1],
                        "Estimated Yearly Cost (₪)": best_plan[2]
                    })

            # Step 4: Create DataFrame and calculate savings
            df_compare = pd.DataFrame(results)
            df_compare["Savings (₪)"] = total_iec_cost - df_compare["Estimated Yearly Cost (₪)"]
            df_compare["Status"] = df_compare["Savings (₪)"].apply(
                lambda x: "🔻 Cheaper" if x > 0 else ("➖ Same" if x == 0 else "🔺 More Expensive")
            )

            # Display top 3 cheaper suppliers sorted by actual savings
            st.subheader("Top 3 Personalized Suggestions")
            top_cheaper = (
                df_compare[df_compare["Savings (₪)"] > 0]
                .sort_values(by="Savings (₪)", ascending=False)
                .head(3)
            )
            if not top_cheaper.empty:
                for _, row in top_cheaper.iterrows():
                    saving = row["Savings (₪)"]
                    st.markdown(
                        f"✅ **{row['Supplier']}** ({row['Best Plan']} of {row['Discount (%)']}) "
                        f"might save you approximately **₪{saving:,.0f}** this year."
                    )
            else:
                st.info(
                    "No suppliers offer a better deal than your current IEC plan based on your consumption profile."
                )

            # Full comparison table
            with st.expander("See Full Supplier Comparison"):
                st.dataframe(
                    df_compare.style.format({
                        "Estimated Yearly Cost (₪)": "₪{:.2f}",
                        "Savings (₪)": "₪{:.2f}"
                    }),
                    use_container_width=True
                )
                st.caption(
                    "⚠️ Note: The discounts shown are calculated as if they apply to the regulated IEC base rate (₪0.6402). "
                    "However, alternative electricity suppliers may apply these discounts to their own (non-public) base rates. "
                    "As a result, actual savings may vary and could be lower than estimated."
                )

else:
    st.info("Please upload a CSV file to begin.")

    # Step 0: Link to external electricity provider
    # ——— Footer: Data Source ———
    st.markdown("---")

    col_left, col_mid, col_right = st.columns([1, 2, 1])

    with col_mid:
        with st.expander("Where do I get my electricity data?"):
            st.markdown(
                """
                You'll need to download your electricity consumption report from the Israel Electric Company (IEC).

                👉 [Click here to log in and download your data](https://www.iec.co.il/login?returnPath=%2Fconsumption-info-menu)
                """,
                unsafe_allow_html=True
            )
