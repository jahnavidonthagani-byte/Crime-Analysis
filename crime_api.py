import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
#import os
import random
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import textwrap
import base64
import io
import matplotlib

# Use Agg backend for matplotlib on headless servers
matplotlib.use('Agg')

# Paths
script_dir = Path(__file__).parent.resolve()
#logo_path = script_dir / "images" / "Logo_TechOptima.png"
raw_path = script_dir / "crime_dataset_india.csv"
clean_path = script_dir / "crime_dataset_india_clean.csv"

def load_image_base64(img_path):
    img = Image.open(img_path)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f'<img src="data:image/png;base64,{img_str}" width="200">'

# Load and clean data
if clean_path.exists():
    df = pd.read_csv(clean_path)
    if 'YearMonth' in df.columns:
        df['YearMonth'] = pd.to_datetime(df['YearMonth'], errors='coerce')
else:
    df = pd.read_csv(raw_path)
    df.columns = df.columns.str.strip()
    date_cols = ['Date Reported', 'Date of Occurrence', 'Time of Occurrence', 'Date Case Closed']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    df['Weapon Used'] = df['Weapon Used'].fillna('Unknown')
    df['Victim Gender'] = df['Victim Gender'].replace('X', 'Unknown')
    numeric_cols = ['Victim Age', 'Police Deployed', 'Crime Code']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())
    df = df.drop_duplicates()
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()
    if 'Date Reported' in df.columns:
        df['Year'] = df['Date Reported'].dt.year
        df['Month'] = df['Date Reported'].dt.month
        df['YearMonth'] = df['Date Reported'].dt.to_period('M').dt.to_timestamp()
    df.to_csv(clean_path, index=False)

if 'Date Reported' in df.columns:
    df['Date Reported'] = pd.to_datetime(df['Date Reported'], errors='coerce')

st.set_page_config(page_title="CrimeAnalysis", layout="wide")

with st.sidebar:
    #st.title("Crime Analytics")
    # try:
    #     if os.path.exists(str(logo_path)):
    #         st.markdown(load_image_base64(logo_path), unsafe_allow_html=True)
    #     else:
    #         st.warning(f"Logo file not found at {logo_path}")
    # except Exception as e:
    #     st.warning(f"Could not load logo: {e}")

    page = st.radio("Select Section:", ["Dashboard", "Knowledge", "Crime Records"])

city_summary = {}
for city in df['City'].dropna().unique():
    city_data = df[df['City'] == city]
    city_summary[city.lower()] = {
        'total_crimes': len(city_data),
        'top_crimes': city_data['Crime Description'].value_counts().head(5).index.tolist() if 'Crime Description' in city_data else [],
        'top_weapons': city_data['Weapon Used'].value_counts().head(3).index.tolist() if 'Weapon Used' in city_data else [],
        'gender_ratio': city_data['Victim Gender'].value_counts().to_dict() if 'Victim Gender' in city_data else {},
        'avg_age': city_data['Victim Age'].mean() if 'Victim Age' in city_data else None
    }

def compose_city_overview(city_name, data, full=False):
    response = f" **City Overview — {city_name.title()}:**\n"
    if full:
        response += f"- Total reported crimes: {data['total_crimes']}\n"
        if data['top_crimes']:
            response += f"- Common crimes: {', '.join(data['top_crimes'])}\n"
        if data['top_weapons']:
            response += f"- Frequently used weapons: {', '.join(data['top_weapons'])}\n"
        if data['gender_ratio']:
            response += f"- Victim gender distribution: {data['gender_ratio']}\n"
        if data['avg_age'] is not None:
            response += f"- Average victim age: {data['avg_age']:.1f} years\n"
        avg_crime = np.mean([c['total_crimes'] for c in city_summary.values()])
        if data['total_crimes'] > avg_crime * 1.2:
            risk_level = "High "
        elif data['total_crimes'] > avg_crime * 0.8:
            risk_level = "Medium "
        else:
            risk_level = "Low "
        response += f"- Crime risk level: {risk_level}\n"
        response += random.choice([
            "Citizens should stay alert and follow safety guidelines.",
            "Authorities may prioritize high-risk areas for better safety."
        ])
    else:
        response += f"- Total reported crimes: {data['total_crimes']}\n"
        if data['top_crimes']:
            response += f"- Common crimes: {', '.join(data['top_crimes'][:3])}\n"
    return response

def handle_single_query(query):
    query_lower = query.lower().strip()
    found_city = None
    found_year = None
    for city in city_summary.keys():
        if city in query_lower:
            found_city = city
            break
    year_match = re.search(r"\b(20\d{2})\b", query_lower)
    if year_match:
        found_year = int(year_match.group(1))   
    if "highest crime city" in query_lower:
        city_counts = df['City'].value_counts()
        return f" **{city_counts.idxmax()}** has the highest number of reported crimes: {city_counts.max()}."
    if "lowest crime city" in query_lower:
        city_counts = df['City'].value_counts()
        return f" **{city_counts.idxmin()}** has the lowest number of reported crimes: {city_counts.min()}."
    if found_city and "weapon" in query_lower:
        city_data = df[df['City'].str.lower() == found_city]
        if 'Weapon Used' in city_data.columns:
            most_weapon = city_data['Weapon Used'].value_counts().idxmax()
            count = city_data['Weapon Used'].value_counts().max()
            return f" The most used weapon in **{found_city.title()}** is **{most_weapon}** ({count} cases)."
        else:
            return " Weapon data not available."
    crime_match = re.search(r"how many (.*?) cases in (.+)", query_lower)
    if crime_match:
        crime_type = crime_match.group(1).strip()
        city_name = crime_match.group(2).strip()
        city_data = df[df['City'].str.lower().str.contains(city_name.lower())]
        count = city_data[city_data['Crime Description'].str.lower().str.contains(crime_type.lower(), na=False)].shape[0]
        return f" There are **{count} {crime_type.title()} cases** reported in **{city_name.title()}**."
    if found_city and found_year:
        if 'Date Reported' in df.columns:
            city_year_data = df[(df['City'].str.lower() == found_city) & (df['Date Reported'].dt.year == found_year)]
            count = len(city_year_data)
            if count > 0:
                top_crimes = city_year_data['Crime Description'].value_counts().head(5)
                response = f" **Crimes in {found_city.title()} for {found_year}: {count} cases.**\n\n**Top Crimes:**\n"
                for crime, val in top_crimes.items():
                    response += f"- {crime}: {val}\n"
                return response
            else:
                return f"No recorded crimes found in **{found_city.title()}** for **{found_year}**."
        else:
            return "Date information not available in dataset."
    if found_city:
        city_data = df[df['City'].str.lower() == found_city]
        total = len(city_data)
        top_crimes = city_data['Crime Description'].value_counts().head(5).index.tolist() if 'Crime Description' in city_data else []
        return compose_city_overview(found_city, {
            'total_crimes': total,
            'top_crimes': top_crimes,
            'top_weapons': city_data['Weapon Used'].value_counts().head(3).index.tolist() if 'Weapon Used' in city_data else [],
            'gender_ratio': city_data['Victim Gender'].value_counts().to_dict() if 'Victim Gender' in city_data else {},
            'avg_age': city_data['Victim Age'].mean() if 'Victim Age' in city_data else None
        }, full="details" in query_lower or "overview" in query_lower)
    return " I couldn't detect a specific question. Try asking:\n- 'Give details of Delhi'\n- 'Highest crime city'\n- 'Show crimes in Delhi 2023'"

def generate_response(user_input):
    queries = re.split(r'\band\b|\bthen\b|,', user_input, flags=re.IGNORECASE)
    responses = []
    for q in queries:
        q = q.strip()
        if q:
            responses.append(handle_single_query(q))
    return "\n\n".join(responses)

def autopct_format(pct, allvals):
    return "{:.1f}%".format(pct)

def render_matplotlib_figure(fig):
    """Render matplotlib figure with fallback to avoid Streamlit image width errors."""
    try:
        plt.tight_layout()
        fig.canvas.draw()
        st.pyplot(fig)
    except Exception:
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        st.image(buf, use_column_width=True)

if page == "Dashboard":
    # Centered Main Title
    st.markdown("<h1 style='text-align: center;'>CRIME ANALYSIS</h1>", unsafe_allow_html=True)
    st.subheader("Intelligent Insights & City-Wise Crime Trends")
    st.write("Empowering Safer Cities Through Data-Driven Crime Insights")

    # Data Processing
    total_cities = df['City'].nunique()
    total_crimes = len(df)
    avg_crime_count = df.groupby('City').size().mean()
    high_risk_cities = len(df.groupby('City').filter(lambda x: len(x) > avg_crime_count)['City'].unique())
    safe_cities = total_cities - high_risk_cities

    # Metric Cards Layout with dark text for clear visibility
    st.markdown(f"""
    <div style='display:flex; gap:20px; flex-wrap:wrap; margin-bottom: 30px;'>
        <div style='background:#e3f2fd; padding:15px; border-radius:10px; flex:1; text-align:center;'>
            <h4 style='color: #1e1e1e; margin: 0;'>Total Cities</h4>
            <p style='color: #1e1e1e; font-size: 24px; font-weight: bold; margin: 5px 0 0 0;'>{total_cities}</p>
        </div>
        <div style='background:#ffcdd2; padding:15px; border-radius:10px; flex:1; text-align:center;'>
            <h4 style='color: #1e1e1e; margin: 0;'>High Risk Cities</h4>
            <p style='color: #1e1e1e; font-size: 24px; font-weight: bold; margin: 5px 0 0 0;'>{high_risk_cities}</p>
        </div>
        <div style='background:#c8e6c9; padding:15px; border-radius:10px; flex:1; text-align:center;'>
            <h4 style='color: #1e1e1e; margin: 0;'>Total Crimes</h4>
            <p style='color: #1e1e1e; font-size: 24px; font-weight: bold; margin: 5px 0 0 0;'>{total_crimes}</p>
        </div>
        <div style='background:#dcedc8; padding:15px; border-radius:10px; flex:1; text-align:center;'>
            <h4 style='color: #1e1e1e; margin: 0;'>Safe Cities</h4>
            <p style='color: #1e1e1e; font-size: 24px; font-weight: bold; margin: 5px 0 0 0;'>{safe_cities}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h4 style='text-align:center; margin-bottom:15px;'>Crime Type Distribution</h4>", unsafe_allow_html=True)
        if 'Crime Description' in df.columns:
            crime_counts = df['Crime Description'].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(6, 6))
            labels = [textwrap.fill(label, 14) for label in crime_counts.index]

            wedges, texts, autotexts = ax.pie(
                crime_counts.values,
                labels=labels,
                autopct=lambda pct: autopct_format(pct, crime_counts.values),
                startangle=130,
                pctdistance=0.6,
                labeldistance=1.12,
                wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
                textprops={'fontsize': 9},
                rotatelabels=True
            )

            for i, text in enumerate(texts):
                x, y = text.get_position()
                offset = 0.02 if i % 2 == 0 else -0.02
                text.set_position((x, y + offset))

            ax.axis('equal')
            render_matplotlib_figure(fig)

        st.markdown("<br>", unsafe_allow_html=True)

    with col2:
        st.markdown("<h4 style='text-align:center; margin-bottom:15px;'>Victim Gender Distribution</h4>", unsafe_allow_html=True)
        if 'Victim Gender' in df.columns:
            gender_counts = df['Victim Gender'].value_counts()
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bar(gender_counts.index, gender_counts.values, color=['#42a5f5', '#ff8a65', '#9e9e9e'])
            ax.set_xlabel("Gender")
            ax.set_ylabel("Count")
            ax.tick_params(axis='x', labelrotation=0)
            render_matplotlib_figure(fig)
        st.markdown("<br>", unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("<h4 style='text-align:center; margin-bottom:15px;'>Top 5 Cities by Crime</h4>", unsafe_allow_html=True)
        top_cities = df['City'].value_counts().head(5)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.barh(top_cities.index[::-1], top_cities.values[::-1], color='#ff7043')
        ax.set_xlabel("Number of Crimes")
        ax.set_ylabel("City")
        render_matplotlib_figure(fig)
        st.markdown("<br>", unsafe_allow_html=True)

    with col4:
        st.markdown("<h4 style='text-align:center; margin-bottom:15px;'>Top 5 Cities with Lowest Crimes</h4>", unsafe_allow_html=True)
        low_cities = df['City'].value_counts().tail(5)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.barh(low_cities.index, low_cities.values, color='#66bb6a')
        ax.set_xlabel("Number of Crimes")
        ax.set_ylabel("City")
        render_matplotlib_figure(fig)
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<h4 style='text-align:center; margin-bottom:15px;'>Weapon Usage Frequency</h4>", unsafe_allow_html=True)
    if 'Weapon Used' in df.columns:
        weapon_counts = df['Weapon Used'].value_counts().head(8)
        fig, ax = plt.subplots(figsize=(5, 3))
        bars = ax.bar(weapon_counts.index, weapon_counts.values, color='#26a69a')

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.3, f'{int(height)}',
                    ha='center', va='bottom', fontsize=6)

        ax.set_xlabel("Weapon", fontsize=6)
        ax.set_ylabel("Count", fontsize=6)
        plt.xticks(rotation=40, fontsize=6)
        plt.yticks(fontsize=7)
        render_matplotlib_figure(fig)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<h4 style='text-align:center; margin-bottom:15px;'>Monthly Crime Trend</h4>", unsafe_allow_html=True)
    if 'YearMonth' in df.columns:
        monthly_crime = df.groupby('YearMonth').size().reset_index(name='Crime Count')
        monthly_crime['MonthLabel'] = monthly_crime['YearMonth'].dt.strftime('%b-%Y')
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(monthly_crime['MonthLabel'], monthly_crime['Crime Count'], color='#1E88E5')
        plt.xticks(rotation=75, fontsize=9)
        plt.yticks(fontsize=8)
        ax.set_xlabel("Month")
        ax.set_ylabel("Number of Crimes")
        render_matplotlib_figure(fig)
    st.markdown("<br>", unsafe_allow_html=True)

elif page == "Knowledge":
    main_col, right_col = st.columns([3, 1])
    with main_col:
        st.title(" CrimeAnalysis")
        st.subheader("Intelligent Insights & City-Wise Crime Trends")
        st.write("Empowering Safer Cities Through Data-Driven Crime Insights")
        if 'City' in df.columns and 'Victim Gender' in df.columns:
            X = pd.get_dummies(df[["City", "Victim Gender"]], drop_first=True)
            y = pd.Series([1 if city_summary[c.lower()]['total_crimes'] > np.mean(
                [city_summary[x]['total_crimes'] for x in city_summary]) else 0 for c in df['City']])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            acc = accuracy_score(y_test, model.predict(X_test))
            st.markdown("###  AI & ML Integration")
            st.write("**Model Used:** Random Forest Classifier")
            st.write(f"**Model Accuracy:** {acc:.2f}")
        if "answer" not in st.session_state:
            st.session_state.answer = ""
        def submit_query():
            st.session_state.answer = generate_response(st.session_state.query_input)
            st.session_state.query_input = ""
        query = st.text_input("Type your question here:", key="query_input", on_change=submit_query)

        if st.session_state.answer:
            st.markdown("###  Answer to Your Question")
            st.write(st.session_state.answer)
    with right_col:
        st.markdown("### About the Platform")
        st.markdown("Provides comprehensive crime analytics for risk assessment and planning.")
        st.markdown("### Target Users")
        st.markdown("Police, city planners, researchers, and citizens.")
        st.markdown("### Sample Cities")
        st.markdown(", ".join(df['City'].dropna().unique()))
        st.markdown("### Sample Questions")
        st.markdown("- Give details of Delhi")
        st.markdown("- Highest crime city")
        st.markdown("- Lowest crime city")
        st.markdown("- Most used weapon in Hyderabad")
        st.markdown("- How many fraud cases in Mumbai")
        st.markdown("- Show crimes in Delhi 2023")
        st.markdown("- Compare Delhi and Mumbai 2023")

elif page == "Crime Records":
    st.title(" Crime Records Database")
    main_columns = ["City", "Crime Description", "Victim Age", "Victim Gender", "Weapon Used", "Date Reported"]
    df_display = df[main_columns] if all(col in df.columns for col in main_columns) else df
    st.dataframe(df_display, use_container_width=True)
    csv = df_display.to_csv(index=False).encode('utf-8')
    st.download_button(" Download CSV", data=csv, file_name="crime_records_clean.csv")
