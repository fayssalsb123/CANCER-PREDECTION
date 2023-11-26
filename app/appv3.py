import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle
import numpy as np

def add_predictions(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))

    input_array = np.array(list(input_data.values())).reshape(1, -1)
    scaled_input_array = scaler.transform(input_array)

    prediction = model.predict(scaled_input_array)
    st.subheader("Cell cluster prediction ")
    st.write("the cell cluster is:")
    if prediction[0] == 0:
        st.write("Benign")
    else:
        st.write("malignant")

    st.write("Probability of being benign:", model.predict_proba(scaled_input_array)[0][0])
    st.write("Probability of being malignant:", model.predict_proba(scaled_input_array)[0][1])
    st.write("this app can assist medical professional in making a diagnosis, but should not be used as a substitute for professional diagnosis")

def add_sidebar(data):
    st.sidebar.header("Cell Nuclei Measurements")
    input_dict = {}
    column_names = data.columns[1:]
    sliders_labels = [(f"{column} (mean)", column) for column in column_names]

    for label, key in sliders_labels:
        input_dict[key] = st.sidebar.slider(
            label=label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
    return input_dict

def get_scaled_values(input_dict):
    data = get_clean_data()
    X = data.drop(['diagnosis'], axis=1)
    scaled_dict = {}
    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value
    return scaled_dict

def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data)
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
                  'Symmetry', 'Fractal Dimension']

    values1 = [input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
               input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
               input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
               input_data['fractal_dimension_mean']]

    values2 = [input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
               input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
               input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']]

    values3 = [input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
               input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
               input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
               input_data['fractal_dimension_worst']]

    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values1,
        theta=categories,
        fill='toself',
        name='mean value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=values2,
        theta=categories,
        fill='toself',
        name='standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r=values3,
        theta=categories,
        fill='toself',
        name='worst value'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        autosize=True
    )
    return fig

def get_clean_data():
    df = pd.read_csv('data/data.csv')
    df = df.drop(['Unnamed: 32', 'id'], axis=1)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    return df

def load_cancer_data(cancer_type):
    if cancer_type == "Breast Cancer":
        data = get_clean_data()
    elif cancer_type == "Skin Cancer":
        data = pd.read_csv('data/data.csv')
    elif cancer_type == "Lung Cancer":
        data = pd.read_csv('data/data.csv')
    return pd.DataFrame(data)

def type_selection():
    
    selected_cancer = st.sidebar.selectbox("Sélectionnez le type de cancer", ["Breast Cancer", "Skin Cancer", "Lung Cancer"])
    return selected_cancer
def description():
    st.markdown("<h1 style='color: #ff5733;'>----------CancerPredictor----------</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: #3366cc;'>Bienvenue dans l'application CancerPredictor!</h2>", unsafe_allow_html=True)
    st.markdown(
        """
        
        <p>Cette application vise à prédire le type de cancer (Breast Cancer, Skin Cancer, Lung Cancer) en se basant sur les caractéristiques des cellules.</p>
        <p>Sélectionnez le type de cancer dans le menu à gauche, ajustez les caractéristiques des cellules à l'aide des sliders, puis cliquez sur "Commencer" pour obtenir des prédictions.</p>
        <p>Les données utilisées dans cette application sont basées sur [précisez la source des données].</p>
        <p>Il est important de noter que les prédictions fournies par cette application sont basées sur des modèles statistiques et ne doivent pas remplacer un diagnostic professionnel.</p>
        <p>Consultez toujours un professionnel de la santé qualifié pour un diagnostic précis.</p>
        """
    , unsafe_allow_html=True)

    

def display_cancer_data(cancer_type):
    if cancer_type == 'Breast Cancer':
        st.markdown("<div style='font-size: 24px;'>Données pour le <span style='color: #3366cc;'>Breast Cancer:</span></div>", unsafe_allow_html=True)
        data = get_clean_data()
        input_data = add_sidebar(data)
        st.title("Breast Cancer Predictor")
        st.write("Please connect this app to your cytology lab to help diagnose from your tissue sample.")
        col1, col2 = st.columns([4, 1])
        with col1:
            radar_chart = get_radar_chart(input_data)
            st.plotly_chart(radar_chart)
        with col2:
                   add_predictions(input_data)
    elif cancer_type == 'Skin Cancer':
        st.markdown("<div style='font-size: 24px;'>Données pour le <span style='color: #3366cc;'>Skin Cancer:</span></div>", unsafe_allow_html=True)
        st.subheader(f"Les Données du {cancer_type} ne sont pas prêtes pour le moment")
    elif cancer_type == 'Lung Cancer':
        st.markdown("<div style='font-size: 24px;'>Données pour le <span style='color: #3366cc;'>Lung Cancer:</span></div>", unsafe_allow_html=True)
        st.subheader(f"Les Données du {cancer_type} ne sont pas prêtes pour le moment")

def main():
    selected_cancer = type_selection()
   
    
    

    if selected_cancer:
        if st.sidebar.button('commencer'):
            st.empty()  # Efface la première fenêtre
            display_cancer_data(selected_cancer)
        else:
            description()
        

if __name__ == "__main__":
    main()
