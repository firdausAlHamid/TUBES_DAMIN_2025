import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, r2_score, mean_squared_error, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
import os

# Set page config
st.set_page_config(
    page_title="Salary Analysis & Prediction Dashboard",
    page_icon="",
    layout="wide"
)

# Title
st.title("Salary Analysis & Prediction Dashboard")
st.write("Analisis gaji dan perbandingan model prediksi (Linear Regression & Random Forest)")

# Load data
@st.cache_data
def load_data():
    # Construct the absolute path to the CSV file
    # Assuming ds_salaries.csv is in the same directory as app.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "ds_salaries.csv")
    df = pd.read_csv(csv_path)
    df['job_title'] = df['job_title'].str.strip()
    return df

df_original = load_data()
min_historic_year = df_original['work_year'].min() # Define min_historic_year globally

# --- Mappings for full descriptions (used for display) ---
experience_level_map = {
    'EN': 'Entry-level / Junior',
    'MI': 'Mid-level / Intermediate',
    'SE': 'Senior-level / Expert',
    'EX': 'Executive-level / Director'
}

employment_type_map = {
    'FT': 'Full-time',
    'PT': 'Part-time',
    'CT': 'Contract',
    'FL': 'Freelance'
}

company_size_map = {
    'S': 'Small (<50 employees)',
    'M': 'Medium (50-250 employees)',
    'L': 'Large (>250 employees)'
}
#end

# Sidebar untuk filter visualisasi umum
st.sidebar.header("Filter Visualisasi Umum")
selected_years_viz = st.sidebar.multiselect(
    "Pilih Tahun",
    options=sorted(df_original['work_year'].unique()),
    default=sorted(df_original['work_year'].unique()),
    key='viz_year_filter'
)
st.sidebar.markdown("")
df_viz = df_original[df_original['work_year'].isin(selected_years_viz)]

# --- Model 1: Linear Regression (Data Scientist Specific) ---
st.header("Model 1: Linear Regression (Spesifik 'Data Scientist')")

# Data Filtering for Linear Model
df_linear_src = df_original[df_original['job_title'] == 'Data Scientist'].copy()

# Outlier Visualization Data (Before for Linear Model)
st.subheader("Visualisasi Outlier Gaji 'Data Scientist' (untuk Model Linear)")
col_box1, col_box2 = st.columns(2)
with col_box1:
    st.write("Sebelum Pembersihan Outlier:")
    if not df_linear_src.empty:
        fig_before_outlier = px.box(df_linear_src, y='salary_in_usd', title="Distribusi Gaji 'Data Scientist' (Sebelum)")
        st.plotly_chart(fig_before_outlier, use_container_width=True)
    else:
        st.write("Tidak ada data 'Data Scientist' untuk ditampilkan sebelum outlier.")

# Outlier Removal for Linear Model
Q1_linear = df_linear_src['salary_in_usd'].quantile(0.25)
Q3_linear = df_linear_src['salary_in_usd'].quantile(0.75)
IQR_linear = Q3_linear - Q1_linear
batas_bawah_linear = Q1_linear - 1.5 * IQR_linear   
batas_atas_linear = Q3_linear + 1.5 * IQR_linear
df_linear_clean = df_linear_src[~((df_linear_src['salary_in_usd'] < batas_bawah_linear) | (df_linear_src['salary_in_usd'] > batas_atas_linear))]

with col_box2:
    st.write("Setelah Pembersihan Outlier:")
    if not df_linear_clean.empty:
        fig_after_outlier = px.box(df_linear_clean, y='salary_in_usd', title="Distribusi Gaji 'Data Scientist' (Sesudah)")
        st.plotly_chart(fig_after_outlier, use_container_width=True)
    else:
        st.write("Tidak ada data 'Data Scientist' tersisa setelah pembersihan outlier.")

if df_linear_clean.empty:
    st.error("Model Linear: Tidak ada data 'Data Scientist' tersisa setelah pembersihan outlier. Model tidak dapat dilatih.")
else:
    categorical_features_linear = ['experience_level', 'employment_type', 'company_size']
    numerical_features_linear = ['work_year', 'remote_ratio']
    X_linear = df_linear_clean[categorical_features_linear + numerical_features_linear]
    y_linear = df_linear_clean['salary_in_usd']

    preprocessor_linear = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features_linear),
            ('num', StandardScaler(), numerical_features_linear)
        ], remainder='passthrough')

    linear_model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_linear),
        ('regressor', LinearRegression())
    ])
    X_linear_train, X_linear_test, y_linear_train, y_linear_test = train_test_split(X_linear, y_linear, test_size=0.2, random_state=42)
    
    if not X_linear_train.empty:
        linear_model_pipeline.fit(X_linear_train, y_linear_train)
        y_pred_linear_eval = linear_model_pipeline.predict(X_linear_test)
        r2_linear = r2_score(y_linear_test, y_pred_linear_eval)
        mse_linear = mean_squared_error(y_linear_test, y_pred_linear_eval)
        mae_linear = mean_absolute_error(y_linear_test, y_pred_linear_eval)
        st.subheader("Evaluasi Model Linear Regression")
        st.caption(f"Model Linear dilatih dengan {len(X_linear_train)} sampel data 'Data Scientist' setelah pembersihan.")
        col_lin_m1, col_lin_m2, col_lin_m3 = st.columns(3)
        col_lin_m1.metric("R-squared (R2)", f"{r2_linear:.3f}")
        col_lin_m2.metric("Mean Squared Error (MSE)", f"USD {mse_linear:,.2f}")
        col_lin_m3.metric("Mean Absolute Error (MAE)", f"USD {mae_linear:,.2f}")
    else:
        st.error("Model Linear: Data training kosong.")

# --- Model 2: Random Forest (General) ---
st.header("Model 2: Random Forest Regressor (Semua Jabatan)")

# Data Preparation for Random Forest Model (using all job titles)
df_rf_src = df_original.copy()

# Outlier Removal for RF Model (general robustness, not visualized as per request for linear only)
Q1_rf = df_rf_src['salary_in_usd'].quantile(0.25)
Q3_rf = df_rf_src['salary_in_usd'].quantile(0.75)
IQR_rf = Q3_rf - Q1_rf
batas_bawah_rf = Q1_rf - 1.5 * IQR_rf
batas_atas_rf = Q3_rf + 1.5 * IQR_rf
df_rf_clean = df_rf_src[~((df_rf_src['salary_in_usd'] < batas_bawah_rf) | (df_rf_src['salary_in_usd'] > batas_atas_rf))]

if df_rf_clean.empty:
    st.error("Model Random Forest: Tidak ada data tersisa setelah pembersihan outlier. Model tidak dapat dilatih.")
else:
    categorical_features_rf = ['experience_level', 'employment_type', 'job_title', 'company_size']
    numerical_features_rf = ['work_year', 'remote_ratio']
    X_rf = df_rf_clean[categorical_features_rf + numerical_features_rf]
    y_rf = df_rf_clean['salary_in_usd']

    preprocessor_rf = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features_rf),
            ('num', StandardScaler(), numerical_features_rf)
        ], remainder='passthrough')

    # Pipeline dasar untuk Random Forest (sebelum GridSearchCV)
    base_rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_rf),
        ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))
    ])

    # Definisikan parameter grid untuk GridSearchCV
    # Opsi 1: Grid yang sedikit lebih luas dari sebelumnya
    param_grid_rf = {
        'regressor__n_estimators': [100, 150, 200], # Lebih banyak opsi estimator
        'regressor__max_depth': [None, 10, 20],    # Lebih banyak opsi kedalaman
        'regressor__min_samples_split': [2, 5, 10], # Lebih banyak opsi min_samples_split
        'regressor__min_samples_leaf': [1, 2, 4],   # Lebih banyak opsi min_samples_leaf
        'regressor__max_features': ['sqrt', 'log2'] # Menambahkan max_features
    }

    # Jika Anda ingin mencoba grid yang lebih agresif (PERINGATAN: AKAN SANGAT LAMA TRAININGNYA):
    # param_grid_rf_aggressive = {
    #     'regressor__n_estimators': [100, 200, 300, 400],
    #     'regressor__max_depth': [None, 10, 20, 30, 40],
    #     'regressor__min_samples_split': [2, 5, 10],
    #     'regressor__min_samples_leaf': [1, 2, 4],
    #     'regressor__max_features': ['sqrt', 'log2', None],
    #     # 'regressor__bootstrap': [True, False] # Contoh hyperparameter lain
    # }
    # Jika memilih ini, ganti param_grid_rf di bawah dengan param_grid_rf_aggressive

    # Inisialisasi GridSearchCV
    # Menggunakan cv=3 untuk cross-validation 3-fold agar tidak terlalu lama saat running
    # scoring='neg_mean_squared_error' karena GridSearchCV memaksimalkan skor, dan kita ingin meminimalkan MSE
    # verbose=2 agar kita bisa lihat lebih banyak log dari GridSearchCV
    grid_search_rf = GridSearchCV(estimator=base_rf_pipeline, param_grid=param_grid_rf, 
                                  cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
    
    X_rf_train, X_rf_test, y_rf_train, y_rf_test = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)
    
    if not X_rf_train.empty:
        with st.spinner("Melakukan tuning hyperparameter untuk Random Forest, ini mungkin memakan waktu..."):
            grid_search_rf.fit(X_rf_train, y_rf_train)
        
        rf_model_pipeline = grid_search_rf.best_estimator_ # Gunakan model terbaik
        st.success("Tuning hyperparameter Random Forest selesai.")
        st.write(f"Parameter terbaik untuk Random Forest: {grid_search_rf.best_params_}")

        y_pred_rf_eval = rf_model_pipeline.predict(X_rf_test)
        r2_rf = r2_score(y_rf_test, y_pred_rf_eval)
        mse_rf = mean_squared_error(y_rf_test, y_pred_rf_eval)
        mae_rf = mean_absolute_error(y_rf_test, y_pred_rf_eval)
        st.subheader("Evaluasi Model Random Forest")
        col_rf_m1, col_rf_m2, col_rf_m3 = st.columns(3)
        col_rf_m1.metric("R-squared (R2)", f"{r2_rf:.3f}")
        col_rf_m2.metric("Mean Squared Error (MSE)", f"USD {mse_rf:,.2f}")
        col_rf_m3.metric("Mean Absolute Error (MAE)", f"USD {mae_rf:,.2f}")
    else:
        st.error("Model Random Forest: Data training kosong.")

# --- Visualizations (using df_viz from sidebar filter) ---
st.markdown("--- ")
st.header("Visualisasi Data Umum (berdasarkan filter sidebar)")
plot_option_viz = st.selectbox(
    "Pilih visualisasi untuk ditampilkan:",
    ["Pilih", "Trend Gaji", "Distribusi Gaji per Tipe Pekerjaan", "Analisis Cluster", "Distribusi Gaji per Level Pengalaman"],
    key='viz_select'
)

if plot_option_viz == "Trend Gaji":
    st.subheader("Trend Gaji Berdasarkan Tahun (diwarnai Level Pengalaman)")
    fig_viz = px.scatter(df_viz, x='work_year', y='salary_in_usd',
                    color='experience_level', 
                    trendline="ols",
                    labels={'experience_level': 'Level Pengalaman', 'work_year': 'Tahun', 'salary_in_usd': 'Gaji (USD)'},
                    title="Trend Gaji")
    st.plotly_chart(fig_viz, use_container_width=True)

elif plot_option_viz == "Distribusi Gaji per Tipe Pekerjaan":
    st.subheader("Distribusi Gaji Berdasarkan Tipe Pekerjaan")
    fig_viz = px.box(df_viz, y="salary_in_usd", x="employment_type", 
                 color="employment_type",
                 labels={'employment_type': 'Tipe Pekerjaan', 'salary_in_usd': 'Gaji (USD)'},
                 title="Distribusi Gaji per Tipe Pekerjaan")
    fig_viz.update_xaxes(categoryorder='array', categoryarray=list(employment_type_map.keys()), tickvals=list(employment_type_map.keys()), ticktext=[employment_type_map[k] for k in employment_type_map.keys()])
    st.plotly_chart(fig_viz, use_container_width=True)

elif plot_option_viz == "Analisis Cluster":
    st.subheader("Analisis Cluster Gaji (berdasarkan Tahun dan Gaji)")
    if not df_viz.empty and len(df_viz) >=2:
        n_clusters_viz = st.slider("Jumlah Cluster", 2, min(6, len(df_viz)-1 if len(df_viz)-1 >=2 else 2), 3, key='cluster_slider_viz') # Ensure min value for slider is 2
        X_cluster_data_viz = df_viz[['work_year', 'salary_in_usd']].copy()
        scaler_cluster_viz = StandardScaler()
        X_scaled_cluster_viz = scaler_cluster_viz.fit_transform(X_cluster_data_viz)
        
        # K-Means for the selected number of clusters
        kmeans_viz = KMeans(n_clusters=n_clusters_viz, random_state=42, n_init='auto')
        cluster_labels_viz = kmeans_viz.fit_predict(X_scaled_cluster_viz)
        X_cluster_data_viz['cluster'] = cluster_labels_viz
        
        fig_viz = px.scatter(X_cluster_data_viz, x='work_year', y='salary_in_usd',
                        color='cluster', color_continuous_scale=px.colors.qualitative.Plotly,
                        labels={'work_year': 'Tahun', 'salary_in_usd': 'Gaji (USD)'},
                        title=f"Cluster Gaji (k={n_clusters_viz})")
        st.plotly_chart(fig_viz, use_container_width=True)

        # Display Silhouette Score and Inertia
        if n_clusters_viz >= 2 and len(np.unique(cluster_labels_viz)) > 1: # Silhouette score requires at least 2 clusters and more than 1 unique label
            silhouette_avg = silhouette_score(X_scaled_cluster_viz, cluster_labels_viz)
            st.write(f"**Silhouette Score (untuk k={n_clusters_viz}):** {silhouette_avg:.3f}")
        else:
            st.write(f"**Silhouette Score (untuk k={n_clusters_viz}):** Tidak dapat dihitung (membutuhkan minimal 2 cluster berbeda).")
        st.write(f"**Inertia/WCSS (untuk k={n_clusters_viz}):** {kmeans_viz.inertia_:.2f}")

        

        # Elbow Method
        st.markdown("---")
        st.subheader("Metode Elbow untuk Menentukan Jumlah Cluster Optimal (k)")
        inertia_values = []
        # Determine a safe upper limit for k in Elbow method
        # Max k should be less than number of samples. Let's cap at 10 or len-1.
        max_k_elbow = min(10, len(X_scaled_cluster_viz) -1 if len(X_scaled_cluster_viz) > 1 else 1) 
        
        if max_k_elbow >= 2: # Elbow method needs at least k=1 and k=2 to plot a line
            range_k = range(1, max_k_elbow + 1)
            for k_val in range_k:
                if k_val == 0: continue # Skip k=0 as KMeans requires n_clusters >= 1
                kmeans_elbow = KMeans(n_clusters=k_val, random_state=42, n_init='auto')
                kmeans_elbow.fit(X_scaled_cluster_viz)
                inertia_values.append(kmeans_elbow.inertia_)
            
            fig_elbow = px.line(x=list(range_k), y=inertia_values, 
                                markers=True,
                                labels={'x': 'Jumlah Cluster (k)', 'y': 'Inertia (WCSS)'},
                                title="Grafik Metode Elbow")
            fig_elbow.update_layout(xaxis_title="Jumlah Cluster (k)", yaxis_title="Inertia (WCSS)")
            st.plotly_chart(fig_elbow, use_container_width=True)
            st.caption("Metode Elbow membantu menemukan 'siku' pada grafik, yang menunjukkan titik di mana penambahan cluster tidak lagi memberikan penurunan inertia yang signifikan.")
        else:
            st.write("Tidak cukup data poin untuk menampilkan grafik Metode Elbow (membutuhkan minimal 2 data poin yang valid untuk k).")

    else:
        st.write("Data tidak cukup untuk analisis cluster.")

elif plot_option_viz == "Distribusi Gaji per Level Pengalaman":
    st.subheader("Distribusi Gaji Berdasarkan Level Pengalaman")
    fig_viz = px.box(df_viz, y="salary_in_usd", x="experience_level", 
                 color="experience_level",
                 labels={'experience_level': 'Level Pengalaman', 'salary_in_usd': 'Gaji (USD)'},
                 title="Distribusi Gaji per Level Pengalaman")
    fig_viz.update_xaxes(categoryorder='array', categoryarray=list(experience_level_map.keys()), tickvals=list(experience_level_map.keys()), ticktext=[experience_level_map[k] for k in experience_level_map.keys()])
    st.plotly_chart(fig_viz, use_container_width=True)

# --- Prediction Section ---
st.markdown("---")
st.header("Prediksi Gaji Menggunakan Kedua Model")

with st.form("prediction_form_dual"):
    st.subheader("Masukkan Detail untuk Prediksi:")
    col1_form_pred, col2_form_pred = st.columns(2)
    
    with col1_form_pred:
        work_year_input_pred = st.selectbox("Tahun", options=sorted(df_original['work_year'].unique()) + [2024, 2025], key='work_year_pred_form')
        job_title_input_pred = st.selectbox("Jabatan", options=sorted(df_original['job_title'].unique()), key='job_title_pred_form')
        experience_level_input_pred = st.selectbox("Level Pengalaman", options=list(experience_level_map.keys()), format_func=lambda x: experience_level_map[x], key='exp_level_pred_form')

    with col2_form_pred:
        employment_type_input_pred = st.selectbox("Tipe Pekerjaan", options=list(employment_type_map.keys()), format_func=lambda x: employment_type_map[x], key='emp_type_pred_form')
        company_size_input_pred = st.selectbox("Ukuran Perusahaan", options=list(company_size_map.keys()), format_func=lambda x: company_size_map[x], key='comp_size_pred_form')
        remote_ratio_input_pred = st.selectbox("Remote Ratio", options=[0, 50, 100], format_func=lambda x: f"{x}% Remote", key='remote_pred_form')
    
    submit_button_pred = st.form_submit_button("Prediksi Gaji")

if submit_button_pred:
    st.subheader("Hasil Prediksi Gaji")
    
    # Baris 1 untuk Linear dan Random Forest
    col_pred_lin, col_pred_rf = st.columns(2)

    with col_pred_lin: # Sebelumnya col_pred_lin
        st.markdown("**Model Linear Regression (Spesifik 'Data Scientist')**")
        if job_title_input_pred == 'Data Scientist' and 'linear_model_pipeline' in locals() and linear_model_pipeline is not None and not df_linear_clean.empty:
            input_linear_pred_df = pd.DataFrame({
                'work_year': [work_year_input_pred],
                'experience_level': [experience_level_input_pred],
                'employment_type': [employment_type_input_pred],
                'company_size': [company_size_input_pred],
                'remote_ratio': [remote_ratio_input_pred]
            })
            input_linear_pred_df = input_linear_pred_df[categorical_features_linear + numerical_features_linear]
            predicted_salary_linear = linear_model_pipeline.predict(input_linear_pred_df)[0]
            st.metric(label=f"Prediksi Gaji Linear ({work_year_input_pred})", value=f"USD {predicted_salary_linear:,.2f}")

            # YoY Calculation and Display for Linear Model
            if work_year_input_pred > min_historic_year:
                input_linear_prev_year_df = pd.DataFrame({
                    'work_year': [work_year_input_pred - 1],
                    'experience_level': [experience_level_input_pred],
                    'employment_type': [employment_type_input_pred],
                    'company_size': [company_size_input_pred],
                    'remote_ratio': [remote_ratio_input_pred]
                })
                input_linear_prev_year_df = input_linear_prev_year_df[categorical_features_linear + numerical_features_linear]
                predicted_salary_linear_prev_year = linear_model_pipeline.predict(input_linear_prev_year_df)[0]
                if predicted_salary_linear_prev_year > 0:
                    yoy_increase_linear = ((predicted_salary_linear - predicted_salary_linear_prev_year) / predicted_salary_linear_prev_year) * 100
                    st.metric(label=f"Kenaikan YoY Linear ({work_year_input_pred-1}-{work_year_input_pred})", value=f"{yoy_increase_linear:.2f}%")
                else:
                    st.text(f"YoY Linear ({work_year_input_pred-1}-{work_year_input_pred}): Gaji tahun lalu nol/negatif, YoY tidak dihitung.")
            
            # Confidence Interval for Linear Model
            if 'mse_linear' in locals() and mse_linear is not None:
                std_err_pred_linear = np.sqrt(mse_linear)
                margin_error_linear = std_err_pred_linear * 1.96 # 95% CI
                lower_linear = max(0, predicted_salary_linear - margin_error_linear) # Kembali ke standar, batas bawah bisa 0
                upper_linear = predicted_salary_linear + margin_error_linear # Kembali ke standar
                st.write(f"Interval Kepercayaan (95%): USD {lower_linear:,.2f} - {upper_linear:,.2f}")
                if mse_linear > 10**9: # Arbitrary threshold for 'high' MSE
                    st.caption("Catatan: MSE Model Linear tinggi, interval kepercayaan mungkin lebar.")
            
        elif job_title_input_pred != 'Data Scientist':
            st.warning("Model Linear Regression ini dilatih khusus untuk 'Data Scientist'. Pilih 'Data Scientist' untuk prediksi dengan model ini.")
        else:
            st.error("Model Linear Regression tidak dapat melakukan prediksi saat ini (data bersih kosong atau model tidak terlatih).")

    with col_pred_rf: # Sebelumnya col_pred_rf
        st.markdown("**Model Random Forest Regressor (Semua Jabatan)**")
        if 'rf_model_pipeline' in locals() and rf_model_pipeline is not None and not df_rf_clean.empty:
            input_rf_pred_df = pd.DataFrame({
                'work_year': [work_year_input_pred],
                'experience_level': [experience_level_input_pred],
                'employment_type': [employment_type_input_pred],
                'job_title': [job_title_input_pred],
                'company_size': [company_size_input_pred],
                'remote_ratio': [remote_ratio_input_pred]
            })
            input_rf_pred_df = input_rf_pred_df[categorical_features_rf + numerical_features_rf]
            predicted_salary_rf = rf_model_pipeline.predict(input_rf_pred_df)[0]
            st.metric(label=f"Prediksi Gaji Random Forest ({work_year_input_pred})", value=f"USD {predicted_salary_rf:,.2f}")

            # YoY Calculation and Display for Random Forest Model
            if work_year_input_pred > min_historic_year:
                input_rf_prev_year_df = pd.DataFrame({
                    'work_year': [work_year_input_pred - 1],
                    'experience_level': [experience_level_input_pred],
                    'employment_type': [employment_type_input_pred],
                    'job_title': [job_title_input_pred],
                    'company_size': [company_size_input_pred],
                    'remote_ratio': [remote_ratio_input_pred]
                })
                input_rf_prev_year_df = input_rf_prev_year_df[categorical_features_rf + numerical_features_rf]
                predicted_salary_rf_prev_year = rf_model_pipeline.predict(input_rf_prev_year_df)[0]
                if predicted_salary_rf_prev_year > 0:
                    yoy_increase_rf = ((predicted_salary_rf - predicted_salary_rf_prev_year) / predicted_salary_rf_prev_year) * 100
                    st.metric(label=f"Kenaikan YoY Random Forest ({work_year_input_pred-1}-{work_year_input_pred})", value=f"{yoy_increase_rf:.2f}%")
                else:
                    st.text(f"YoY Random Forest ({work_year_input_pred-1}-{work_year_input_pred}): Gaji tahun lalu nol/negatif, YoY tidak dihitung.")
            else:
                st.text(f"YoY Random Forest ({work_year_input_pred}): Tidak ada data tahun sebelumnya ({min_historic_year}) untuk perhitungan YoY.")

            # Confidence Interval for Random Forest Model
            if 'mse_rf' in locals() and mse_rf is not None:
                std_err_pred_rf = np.sqrt(mse_rf)
                margin_error_rf = std_err_pred_rf * 1.96 # 95% CI
                lower_rf = max(0, predicted_salary_rf - margin_error_rf) # Kembali ke standar
                upper_rf = predicted_salary_rf + margin_error_rf # Kembali ke standar
                st.write(f"Interval Kepercayaan (95%): USD {lower_rf:,.2f} - {upper_rf:,.2f}")
                if mse_rf > 10**9: # Arbitrary threshold for 'high' MSE
                    st.caption("Catatan: MSE Model Random Forest tinggi, interval kepercayaan mungkin lebar.")
        else:
            st.error("Model Random Forest tidak dapat melakukan prediksi saat ini (data bersih kosong atau model tidak terlatih).")

    st.markdown("--- ")
    st.write("**Faktor Input yang Digunakan untuk Prediksi:**")
    st.write(f"- Tahun: {work_year_input_pred}")
    st.write(f"- Jabatan: {job_title_input_pred}")
    st.write(f"- Level Pengalaman: {experience_level_map.get(experience_level_input_pred, experience_level_input_pred)}")
    st.write(f"- Tipe Pekerjaan: {employment_type_map.get(employment_type_input_pred, employment_type_input_pred)}")
    st.write(f"- Ukuran Perusahaan: {company_size_map.get(company_size_input_pred, company_size_input_pred)}")
    st.write(f"- Remote Ratio: {remote_ratio_input_pred}%")

# Footer
st.markdown("---")
st.markdown("Dashboard created for Data Mining Analysis") 