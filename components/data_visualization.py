import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('crop_recommendation.csv')

def plot_graph(x_param, y_param, graph_type):
    plt.figure(figsize=(10, 6))
    
    if graph_type == 'Scatter':
        for label in df['label'].unique():
            subset = df[df['label'] == label]
            plt.scatter(subset[x_param], subset[y_param], label=label, alpha=0.6)
    elif graph_type == 'Line':
        for label in df['label'].unique():
            subset = df[df['label'] == label]
            plt.plot(subset[x_param], subset[y_param], label=label, marker='o', alpha=0.6)
    elif graph_type == 'Bar':
        for label in df['label'].unique():
            subset = df[df['label'] == label]
            plt.bar(subset[x_param], subset[y_param], label=label, alpha=0.6)

    plt.title(f'{graph_type} plot between {x_param} and {y_param}')
    plt.xlabel(x_param)
    plt.ylabel(y_param)
    plt.legend(title='Crop Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

def data_visualization():
    st.title("Data Visualization")
    x_param = st.selectbox("Select X-axis parameter", df.columns)
    y_param = st.selectbox("Select Y-axis parameter", df.columns)
    graph_type = st.selectbox("Select graph type", ["Scatter", "Line", "Bar"])

    if st.button("Generate Graph"):
        plot_graph(x_param, y_param, graph_type)
