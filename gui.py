import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
from vsm import VSM  # Assuming your VSM class is stored in a file named vsm.py

def main():
    st.title("VSM  Text Classification and Clustering")
    st.header("K214889 - Muhammad Qasim Alias Haseeb")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ("KNN Implementation", "K-means Clustering"))

    if page == "KNN Implementation":
        knn_section()
    elif page == "K-means Clustering":
        kmeans_section()

def knn_section():
    v = VSM()
    v.loadtfidf()
    v.split_data()
    v.classifyKNN()

    k = st.number_input("Enter the value of (k)", min_value=2, max_value=10, value=5, step=1)
    s_r = st.number_input("Enter the split ratio ", value=0.2)
    if st.button("Run K-NN Clustering"):
        v = VSM()
        v.loadtfidf()
        v.split_data(s_r)
        v.classifyKNN(k)

        st.header("KNN Implementation")
        st.subheader("Train Documents")
        st.text(", ".join(map(str, v.train_docs)))
        st.subheader("Test Documents")
        st.text(", ".join(map(str, v.test_docs)))

        st.subheader("Results (Predicted Labels)")
        predicted_labels_str = {str(key): value for key, value in v.predicted_labels.items()}
        st.write(predicted_labels_str)

        st.subheader("Evaluation Metrics")
        precision, recall, accuracy, f1_score = v.calculate_metrics()
        st.write("Precision:", precision)
        st.write("Recall:", recall)
        st.write("Accuracy:", accuracy)
        st.write("F1 Score:", f1_score)

def kmeans_section():
    st.header("K-means Clustering")

    # User input for number of clusters (k)
    k = st.number_input("Enter the number of clusters (k)", min_value=2, max_value=10, value=5, step=1)

    # Button to start K-means clustering
    if st.button("Run K-means Clustering"):
        v = VSM()
        v.loadtfidf()
        v.split_data()
        v.start_Kmean(k)

        st.subheader("Random Seeds Taken")
        st.text(", ".join(map(str, v.seed_docs)))

        st.subheader("Final Clusters")
        st.write(v.cluster)

        st.subheader("Evaluation Metrics")
        v.getKmeanEval()
        st.write("Purity:", v.purity)
        st.write("Rand Index:", v.rand_index)

        st.subheader("Silhouette Score vs. Number of Clusters")
        k_values = range(2, 11)  # Try k from 2 to 10
        silhouette_scores = []

        for k in k_values:
            clusters = v.sil_Kmean(k)
            silhouette_avg = v.calculate_silhouette_score(clusters, v.docs)
            silhouette_scores.append(silhouette_avg)

        # Plot silhouette scores vs. number of clusters
        plt.plot(k_values, silhouette_scores, marker='o')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score vs. Number of Clusters')
        plt.grid(True)

        # Display silhouette score graph
        st.pyplot(plt)

if __name__ == "__main__":
    main()
