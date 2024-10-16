import streamlit as st
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
nltk.download('punkt')

# Configurar estilo de Seaborn para gráficos más atractivos
sns.set(style="whitegrid")

# Función para analizar el sentimiento de un texto
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity, analysis.sentiment.subjectivity

# Función para procesar las reseñas
def process_reviews(reviews):
    df = pd.DataFrame(reviews, columns=["Review"])
    df["Polarity"], df["Subjectivity"] = zip(*df["Review"].apply(analyze_sentiment))
    df["Sentiment_Label"] = df["Polarity"].apply(lambda x: "Positive" if x > 0 else ("Negative" if x < 0 else "Neutral"))
    return df

# Título y descripción en la aplicación
st.title("Advanced Sentiment Analysis")
st.write("This app performs sentiment analysis on a set of reviews and generates various complex visualizations.")

# Input: Permitir al usuario cargar un archivo o escribir reseñas manualmente
st.sidebar.header("Input Options")
input_option = st.sidebar.selectbox("Choose input method", ["Manual Input", "Upload a CSV File"])

if input_option == "Manual Input":
    reviews_input = st.text_area("Enter reviews separated by a new line (each review on a new line):")
    reviews = reviews_input.split("\n")
else:
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file with a 'Review' column", type=["csv"])
    if uploaded_file is not None:
        reviews_df = pd.read_csv(uploaded_file)
        reviews = reviews_df["Review"].tolist()

if reviews:
    # Procesar las reseñas
    df = process_reviews(reviews)
    
    # Mostrar el DataFrame
    st.subheader("Sentiment Analysis Results")
    st.write(df)

    # Gráficos de Distribución de Polaridad y Subjetividad
    st.subheader("Polarity and Subjectivity Distribution")
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.histplot(df["Polarity"], bins=20, ax=ax[0], kde=True, color='blue')
    ax[0].set_title("Polarity Distribution (Positive/Negative)")
    ax[0].set_xlabel("Polarity")
    
    sns.histplot(df["Subjectivity"], bins=20, ax=ax[1], kde=True, color='orange')
    ax[1].set_title("Subjectivity Distribution (Objective/Subjective)")
    ax[1].set_xlabel("Subjectivity")
    
    st.pyplot(fig)
    
    # Gráfico de Sentimientos (Pie Chart)
    st.subheader("Sentiment Breakdown")
    sentiment_counts = df["Sentiment_Label"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['green', 'red', 'gray'])
    ax.axis('equal')
    st.pyplot(fig)
    
    # Gráfico de Dispersión de Polaridad vs. Subjetividad
    st.subheader("Polarity vs. Subjectivity")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x="Polarity", y="Subjectivity", data=df, hue="Sentiment_Label", palette=['green', 'red', 'gray'], s=100, ax=ax)
    ax.set_title("Polarity vs. Subjectivity Scatter Plot")
    st.pyplot(fig)
    
    # Resumen Agregado
    st.subheader("Overall Sentiment Summary")
    avg_polarity = df["Polarity"].mean()
    avg_subjectivity = df["Subjectivity"].mean()
    
    st.write(f"**Average Polarity**: {avg_polarity:.2f}")
    st.write(f"**Average Subjectivity**: {avg_subjectivity:.2f}")
    
    # Gráfico de Barras Apiladas: Número de reseñas por sentimiento con promedio de polaridad
    st.subheader("Sentiment Count with Average Polarity")
    sentiment_avg_polarity = df.groupby("Sentiment_Label")["Polarity"].mean()
    sentiment_counts = df["Sentiment_Label"].value_counts()

    fig, ax = plt.subplots(figsize=(10, 6))
    sentiment_avg_polarity.plot(kind='bar', color=['green', 'red', 'gray'], ax=ax, width=0.4, position=1)
    sentiment_counts.plot(kind='bar', color=['lightgreen', 'salmon', 'lightgray'], ax=ax, width=0.4, position=0)
    ax.set_ylabel("Count / Avg Polarity")
    ax.set_title("Sentiment Count and Average Polarity")
    st.pyplot(fig)
