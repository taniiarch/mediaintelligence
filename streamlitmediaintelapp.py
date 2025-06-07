import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import google.generativeai as genai
import json
import io

# --- Konfigurasi Gemini API ---
# Anda bisa menyimpan kunci API Anda di .streamlit/secrets.toml
# Contoh secrets.toml:
# GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
# Atau masukkan secara manual di sidebar (untuk demo)
try:
    # Coba memuat dari Streamlit secrets
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except KeyError:
    # Jika tidak ada di secrets, berikan input manual (opsional)
    st.warning("Tambahkan GEMINI_API_KEY Anda ke file secrets.toml (di folder .streamlit) untuk menghasilkan insight. Atau masukkan di bawah.")
    with st.sidebar:
        st.session_state["GEMINI_API_KEY_ENV"] = st.text_input("Masukkan Kunci API Gemini Anda (Opsional untuk Insight):", type="password")
        if st.session_state["GEMINI_API_KEY_ENV"]:
            genai.configure(api_key=st.session_state["GEMINI_API_KEY_ENV"])


# --- Fungsi Pembersihan dan Pembuatan Grafik/Insight ---

def generate_mock_data():
    """Menghasilkan DataFrame tiruan untuk demonstrasi."""
    data = {
        'Date': pd.to_datetime(['2023-01-01', '2023-01-05', '2023-01-10', '2023-01-15', '2023-01-20', '2023-01-25', '2023-01-30']),
        'Platform': ['Twitter', 'Facebook', 'Instagram', 'Twitter', 'LinkedIn', 'TikTok', 'Twitter'],
        'Sentiment': ['Positive', 'Negative', 'Neutral', 'Positive', 'Positive', 'Negative', 'Neutral'],
        'Location': ['New York', 'London', 'Paris', 'New York', 'Tokyo', 'Sydney', 'London'],
        'Engagements': [120, 80, 50, 150, None, 100, 70], # Mensimulasikan data yang hilang
        'Media Type': ['Text', 'Image', 'Video', 'Text', 'Link', 'Video', 'Image']
    }
    df = pd.DataFrame(data)
    return df

def clean_data(df):
    """
    Melakukan pembersihan data pada DataFrame:
    - Mengonversi 'Date' ke datetime.
    - Mengisi 'Engagements' yang hilang dengan 0.
    - Normalisasi nama kolom (semua lowercase dan spasi diganti underscore).
    """
    df['Date'] = pd.to_datetime(df['Date'])
    df['Engagements'] = df['Engagements'].fillna(0).astype(int)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df

def generate_charts_and_insights(df):
    """
    Menghasilkan grafik Plotly dan insight menggunakan Gemini API (jika kunci tersedia).
    """
    dashboard_data = {}

    # --- 1. Pie chart: Sentiment Breakdown ---
    sentiment_counts = df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    fig_sentiment = px.pie(sentiment_counts, values='Count', names='Sentiment',
                           title='Sentiment Breakdown', hole=.4,
                           color_discrete_sequence=['#F472B6', '#EF4444', '#FCD34D']) # Pink, Red, Yellow
    fig_sentiment.update_layout(height=350, margin=dict(t=50, b=50, l=50, r=50),
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(family='Inter', color='#374151'), showlegend=True,
                                legend=dict(orientation='h', yanchor='bottom', y=-0.2))
    dashboard_data['sentiment_breakdown'] = {'title': 'Sentiment Breakdown', 'chart': fig_sentiment, 'insights': []}

    # --- 2. Line chart: Engagement Trend over time ---
    # Mengelompokkan berdasarkan minggu untuk tren
    engagement_trend = df.groupby(df['date'].dt.to_period('W'))['engagements'].sum().reset_index()
    engagement_trend['date'] = engagement_trend['date'].dt.to_timestamp() # Konversi Period kembali ke Timestamp untuk Plotly
    fig_engagement_trend = px.line(engagement_trend, x='date', y='engagements',
                                   title='Engagement Trend Over Time',
                                   line_shape='linear', markers=True,
                                   color_discrete_sequence=['#EC4899']) # Pink gelap
    fig_engagement_trend.update_layout(height=350, margin=dict(t=50, b=50, l=50, r=50),
                                      xaxis_title='Date', yaxis_title='Engagements',
                                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                      font=dict(family='Inter', color='#374151'))
    dashboard_data['engagement_trend'] = {'title': 'Engagement Trend over Time', 'chart': fig_engagement_trend, 'insights': []}

    # --- 3. Bar chart: Platform Engagements ---
    platform_engagements = df.groupby('platform')['engagements'].sum().reset_index()
    fig_platform_engagements = px.bar(platform_engagements, x='platform', y='engagements',
                                      title='Platform Engagements',
                                      color='platform',
                                      color_discrete_map={
                                          'Twitter': '#DB2777',
                                          'Facebook': '#F0ABFC',
                                          'Instagram': '#FB7185',
                                          'LinkedIn': '#F87171',
                                          'TikTok': '#C084FC'
                                      }) # Skema warna pink
    fig_platform_engagements.update_layout(height=350, margin=dict(t=50, b=50, l=50, r=50),
                                          xaxis_title='Platform', yaxis_title='Total Engagements',
                                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                          font=dict(family='Inter', color='#374151'))
    dashboard_data['platform_engagements'] = {'title': 'Platform Engagements', 'chart': fig_platform_engagements, 'insights': []}

    # --- 4. Pie chart: Media Type Mix ---
    media_type_counts = df['media_type'].value_counts().reset_index()
    media_type_counts.columns = ['Media Type', 'Count']
    fig_media_type = px.pie(media_type_counts, values='Count', names='Media Type',
                           title='Media Type Mix', hole=.4,
                           color_discrete_sequence=['#F9A8D4', '#FBCFE8', '#FCE7F6', '#FDA4AF']) # Warna pink yang lebih terang
    fig_media_type.update_layout(height=350, margin=dict(t=50, b=50, l=50, r=50),
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(family='Inter', color='#374151'), showlegend=True,
                                legend=dict(orientation='h', yanchor='bottom', y=-0.2))
    dashboard_data['media_type_mix'] = {'title': 'Media Type Mix', 'chart': fig_media_type, 'insights': []}

    # --- 5. Bar chart: Top 5 Locations ---
    location_engagements = df.groupby('location')['engagements'].sum().nlargest(5).reset_index()
    fig_top_locations = px.bar(location_engagements, x='location', y='engagements',
                               title='Top 5 Locations by Engagements',
                               color_discrete_sequence=['#FBCFE8']) # Warna pink
    fig_top_locations.update_layout(height=350, margin=dict(t=50, b=50, l=50, r=50),
                                   xaxis_title='Location', yaxis_title='Total Engagements',
                                   paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                   font=dict(family='Inter', color='#374151'))
    dashboard_data['top_locations'] = {'title': 'Top 5 Locations by Engagements', 'chart': fig_top_locations, 'insights': []}

    # --- Menghasilkan insight menggunakan Gemini API (jika kunci tersedia) ---
    # Cek apakah API key sudah dikonfigurasi
    api_key_configured = False
    try:
        # Check if genai is configured (this might raise an error if not configured)
        _ = genai.get_model('gemini-2.0-flash')
        api_key_configured = True
    except Exception:
        api_key_configured = False

    if api_key_configured:
        model = genai.GenerativeModel('gemini-2.0-flash')
        for key, value in dashboard_data.items():
            chart_data_desc = ""
            chart_title = value['title']

            # Siapkan deskripsi data untuk prompt LLM
            if key == 'sentiment_breakdown':
                chart_data_desc = f"Sentiment counts: {sentiment_counts.to_dict('records')}"
            elif key == 'engagement_trend':
                chart_data_desc = f"Engagement trend data: {engagement_trend.to_dict('records')}"
            elif key == 'platform_engagements':
                chart_data_desc = f"Platform engagements: {platform_engagements.to_dict('records')}"
            elif key == 'media_type_mix':
                chart_data_desc = f"Media type mix: {media_type_counts.to_dict('records')}"
            elif key == 'top_locations':
                chart_data_desc = f"Top locations by engagements: {location_engagements.to_dict('records')}"

            try:
                prompt = f"Berdasarkan data berikut untuk \"{chart_title}\", berikan 3 insight ringkas teratas. Jadilah spesifik dan dapat ditindaklanjuti:\n\n{chart_data_desc}"
                response = model.generate_content(
                    [{"role": "user", "parts": [{"text": prompt}]}],
                    generation_config={
                        "response_mime_type": "application/json",
                        "response_schema": {"type": "ARRAY", "items": {"type": "STRING"}}
                    }
                )
                insights = json.loads(response.text)
                dashboard_data[key]['insights'] = insights
            except Exception as e:
                st.error(f"Gagal menghasilkan insight untuk '{chart_title}': {e}. Pastikan Anda memiliki koneksi internet dan kuota API yang cukup.")
                dashboard_data[key]['insights'] = ["Gagal menghasilkan insight."]
    else:
        st.info("Kunci API Gemini tidak ditemukan atau tidak valid. Insight tidak akan dihasilkan.")

    return dashboard_data

# --- Aplikasi Streamlit ---

# Konfigurasi tata letak halaman
st.set_page_config(layout="wide", page_title="Interactive Media Intelligence Dashboard")

# --- CSS Kustom untuk tema pink ---
st.markdown("""
    <style>
    /* Mengatur latar belakang aplikasi menjadi gradien pink */
    .stApp {
        background: linear-gradient(to bottom right, #FCE7F6, #FCE7F6, #FCE7F6);
    }
    /* Gaya untuk kontainer utama konten */
    .main {
        background-color: white;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 2rem;
    }
    /* Gaya untuk judul dan subjudul */
    h1, h2, h3, h4, h5, h6 {
        color: #DB2777; /* Rose 700 */
        font-family: 'Inter', sans-serif;
    }
    /* Gaya untuk label file uploader */
    .css-1d3z3hw > div > label { /* Target spesifik label uploader Streamlit */
        color: white !important;
        background-color: #EC4899; /* Pink 600 */
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        cursor: pointer;
    }
    .css-1d3z3hw > div > label:hover {
        background-color: #DB2777; /* Pink 700 */
    }
    /* Gaya untuk tombol umum Streamlit */
    .stButton>button {
        background-color: #8B5CF6; /* Purple 600 */
        color: white;
        padding: 0.75rem 2rem;
        border-radius: 0.5rem;
        font-weight: bold;
        font-size: 1.125rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #7C3AED; /* Purple 700 */
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    /* Gaya untuk peringatan (misalnya, kunci API) */
    .stAlert {
        border-radius: 0.5rem;
    }
    /* Gaya untuk teks informasi */
    .st-emotion-cache-16txt4v p {
        color: #B91C1C; /* Merah untuk pesan error */
    }
    </style>
    """, unsafe_allow_html=True)

# --- Header Aplikasi ---
st.title("Interactive Media Intelligence Dashboard")
st.subheader("by Tania Putri Rachmadani")
st.markdown("Unggah CSV Anda untuk memvisualisasikan dan mendapatkan insight dari data media Anda.")

st.markdown("---")

# --- Bagian Unggah CSV ---
st.header("1. Unggah File CSV Anda")
st.markdown("Kolom yang diperlukan: **Tanggal, Platform, Sentimen, Lokasi, Engagements, Jenis Media**")

uploaded_file = st.file_uploader("Pilih File CSV", type="csv")

# Inisialisasi session state untuk menyimpan output dashboard
if 'dashboard_output' not in st.session_state:
    st.session_state['dashboard_output'] = None
if 'cleaned_df' not in st.session_state:
    st.session_state['cleaned_df'] = None

if uploaded_file:
    st.success(f"File terpilih: {uploaded_file.name}")
    try:
        # Membaca file CSV
        df_uploaded = pd.read_csv(uploaded_file)

        # Validasi dasar untuk kolom yang diperlukan
        required_cols = ['Date', 'Platform', 'Sentiment', 'Location', 'Engagements', 'Media Type']
        if not all(col in df_uploaded.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df_uploaded.columns]
            st.error(f"CSV yang diunggah tidak memiliki kolom yang diperlukan: {', '.join(missing_cols)}. Harap periksa formatnya.")
            st.session_state['dashboard_output'] = None # Reset dashboard jika error
            st.stop() # Hentikan eksekusi jika kolom hilang

        if st.button("Proses Data"):
            with st.spinner('Memproses Data dan Menghasilkan Grafik & Insight...'):
                st.session_state['cleaned_df'] = clean_data(df_uploaded.copy())
                st.session_state['dashboard_output'] = generate_charts_and_insights(st.session_state['cleaned_df'])
            st.success("Data berhasil diproses!")

    except Exception as e:
        st.error(f"Gagal memproses CSV: {e}. Harap pastikan format file sudah benar.")
        st.session_state['dashboard_output'] = None
        st.session_state['cleaned_df'] = None # Reset data yang dibersihkan juga

else:
    # Jika tidak ada file yang diunggah, tawarkan untuk menampilkan contoh dashboard
    st.info("Atau, Anda dapat melihat contoh dashboard tanpa mengunggah file CSV.")
    if st.button("Tampilkan Contoh Dashboard"):
        with st.spinner('Memuat Contoh Dashboard...'):
            mock_df = generate_mock_data()
            st.session_state['cleaned_df'] = clean_data(mock_df.copy())
            st.session_state['dashboard_output'] = generate_charts_and_insights(st.session_state['cleaned_df'])
        st.success("Contoh Dashboard berhasil dimuat!")

st.markdown("---")

# --- Bagian Dashboard Interaktif ---
if st.session_state.get('dashboard_output'):
    st.header("Dashboard Interaktif")
    for key, chart_info in st.session_state['dashboard_output'].items():
        st.subheader(f"{chart_info['title']}")
        st.plotly_chart(chart_info['chart'], use_container_width=True)
        st.markdown("#### 3 Insight Teratas:")
        # Tampilkan insights
        if chart_info['insights']:
            for i, insight in enumerate(chart_info['insights']):
                st.write(f"- {insight}")
        else:
            st.write("Tidak ada insight yang dihasilkan.")
        st.markdown("---") # Garis pemisah antar chart

    # --- Opsi Unduh ---
    st.markdown("### Unduh Laporan")
    st.info("Fungsionalitas unduh PDF dari seluruh dashboard (seperti di React) lebih kompleks di Streamlit. Anda dapat mengunduh setiap grafik sebagai HTML interaktif atau data yang dibersihkan sebagai CSV.")

    # Unduh data yang dibersihkan sebagai CSV
    if st.session_state['cleaned_df'] is not None:
        csv_data = st.session_state['cleaned_df'].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Unduh Data yang Dibersihkan (CSV)",
            data=csv_data,
            file_name="cleaned_media_intelligence_data.csv",
            mime="text/csv",
        )

    # Unduh setiap grafik sebagai HTML (interaktif)
    for key, chart_info in st.session_state['dashboard_output'].items():
        chart_html = chart_info['chart'].to_html(full_html=False, include_plotlyjs='cdn')
        st.download_button(
            label=f"Unduh '{chart_info['title']}' (HTML Interaktif)",
            data=chart_html,
            file_name=f"{key}_chart.html",
            mime="text/html",
        )

# --- Footer Aplikasi ---
st.markdown("---")
st.caption(f"&copy; {pd.Timestamp.now().year} Media Intelligence Dashboard. Hak Cipta Dilindungi.")

