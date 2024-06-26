import numpy as np  # Import numpy library
import json  # Import JSON module for data serialization
import time  # Import time module for time-related operations
from sklearn.metrics import adjusted_rand_score, silhouette_score  # Import evaluation metrics
import matplotlib.pyplot as plt  # Import matplotlib for plotting

class VSM:
    """Vector Space Model class for text classification and clustering."""
    
    def __init__(self) -> None:
        """Initialize VSM class attributes."""
        self.docs = np.asarray([1, 2, 3, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26], dtype=np.int16)  # Document IDs
        self.doc_tfidf = {}  # TF-IDF dictionary
        self.output_filename = 'doc_tfidf.json'  # Output filename for TF-IDF JSON file
        self.labels = {}  # Ground truth labels
        self.predicted_labels = {}  # Predicted labels
        self.train_docs = []  # Training documents
        self.test_docs = []  # Testing documents
        self.train_labels = []  # Training labels
        self.test_labels = []  # Testing labels
        self.purity = 0  # Purity metric
        self.rand_index = 0  # Rand index metric
        self.silhouette_avg = 0  # Silhouette score metric
        self.total_correct = 0  # Total correct assignments
        self.seed_docs = []
        self.class_labels = {
            "Explainable Artificial Intelligence": [1, 2, 3, 7],
            "Heart Failure": [8, 9, 11],
            "Time Series Forecasting": [12, 13, 14, 15, 16],
            "Transformer Model": [17, 18, 21],
            "Feature Selection": [22, 23, 24, 25, 26]
        }  # Class labels for clustering
        self.golden_cluster = {
            0: [1, 2, 3, 7],
            1: [8, 9, 11],
            2: [12, 13, 14, 15, 16],
            3: [17, 18, 21],
            4: [22, 23, 24, 25, 26]
        }  # Golden clusters for evaluation
        self.cluster = {}  # Computed clusters
        
    def loadtfidf(self):
        """Load TF-IDF dictionary from a JSON file."""
        with open(self.output_filename, 'r') as json_file:
            doc_tfidf_loaded = json.load(json_file)
        self.doc_tfidf = {int(key): np.array(value) for key, value in doc_tfidf_loaded.items()}
    
    def split_data(self, test_ratio=0.2):
        """Split data into train and test sets, and assign labels."""
        for class_name, documents in self.class_labels.items():
            for doc in documents:
                self.labels[doc] = class_name
        
        document_list = np.random.permutation(self.docs)
        num_docs = len(document_list)
        num_train = int((1 - test_ratio) * num_docs)
        self.train_docs = document_list[:num_train]
        self.test_docs = document_list[num_train:]

        self.train_labels = {doc: self.labels[doc] for doc in self.train_docs}
        self.test_labels = {doc: self.labels[doc] for doc in self.test_docs}
    
    def getscore(self, doc1, train_docs):
        """Calculate similarity scores between a test document and training documents."""
        score = []
        for doc in train_docs:
            score.append((doc, np.around(np.sqrt(np.sum(np.square(self.doc_tfidf[doc] - self.doc_tfidf[doc1]))), 3)))
        sorted_score = sorted(score, key=lambda x: x[1])
        return sorted_score
    
    def classifyKNN(self, k=7):
        """Classify test documents using KNN algorithm."""
        for d in self.test_docs:
            temp_doc_list = []
            label_scores = {}
            temp_doc_list = self.getscore(d, self.train_docs)
            
            for i in range(k):
                ith_doc = temp_doc_list[i]
                if self.train_labels[ith_doc[0]] not in label_scores:
                    label_scores[self.train_labels[ith_doc[0]]] = np.around(np.sum(self.doc_tfidf[ith_doc[0]] * self.doc_tfidf[d]), 3)
                else:
                    label_scores[self.train_labels[ith_doc[0]]] += np.around(np.sum(self.doc_tfidf[ith_doc[0]] * self.doc_tfidf[d]), 3)
            
            doc_label = max(label_scores, key=lambda k: label_scores[k])
            self.predicted_labels[d] = doc_label
    
    def calculate_metrics(self):
        """Calculate precision, recall, accuracy, and F1 score."""
        p_labels = []
        t_labels = []
        precisions = {}
        recalls = {}
        macro_index = {}

        for i, j in self.predicted_labels.items():
            p_labels.append(j)
            t_labels.append(self.labels[i])
        
        for t_l in set(t_labels):
            if t_l not in macro_index:
                macro_index[t_l] = {"TP": 0, "FP": 0, "FN": 0}
            macro_index[t_l]["TP"] = sum(1 for true_label, predicted_label in zip(t_labels, p_labels) if true_label == predicted_label and true_label == t_l)
            macro_index[t_l]["FP"] = sum(1 for true_label, predicted_label in zip(t_labels, p_labels) if true_label != predicted_label and predicted_label == t_l)
            macro_index[t_l]["FN"] = sum(1 for true_label, predicted_label in zip(t_labels, p_labels) if true_label != predicted_label and  true_label == t_l)
            precisions[t_l] = macro_index[t_l]["TP"] / (macro_index[t_l]["TP"] + macro_index[t_l]["FP"]) if macro_index[t_l]["TP"] + macro_index[t_l]["FP"] != 0 else 0
            recalls[t_l] = macro_index[t_l]["TP"] / (macro_index[t_l]["TP"] + macro_index[t_l]["FN"]) if macro_index[t_l]["TP"] + macro_index[t_l]["FN"] != 0 else 0
        
        precision = sum(precisions[i] for i in precisions) / len(precisions)
        recall = sum(recalls[i] for i in recalls) / len(recalls)
        accuracy = sum(macro_index[label]["TP"] for label in macro_index) / len(self.predicted_labels)
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
        print("\nPrecision: ", precision, "Recall: ", recall, "Accuracy: ", accuracy, "F1 Score: ", f1_score)
        return precision, recall, accuracy, f1_score
    
    def euclidean_distance(self, vec1, vec2):
        """Calculate Euclidean distance between two vectors."""
        return np.around(np.sqrt(np.sum(np.square(vec1 - vec2))), 3)
    
    def calculate_centroid(self, cluster):
        """Calculate centroid of a cluster."""
        l = len(cluster)
        n_centroid = np.zeros(len(self.doc_tfidf[1]))
        for i in range(l):
            n_centroid += self.doc_tfidf[cluster[i]]
        try:
            n_centroid *= (1 / l)
        except:
            n_centroid *= 1
        return n_centroid
    
    def initialize_centroids(self, data, k, seed_docs):
        """Randomly initialize K centroids."""
        np.random.seed(int(time.time()))  # for reproducibility
        indices = np.random.choice(len(data), size=k, replace=False)
        centroids = np.array([self.doc_tfidf[data[index]] for index in indices])
        for i in indices:
            seed_docs.append(data[i])
        self.seed_docs = seed_docs
        return centroids

    def build_clusters(self, centroids, k=5):
        """Build clusters based on centroids."""
        clusters = {i: [] for i in range(k)}
        for doc in self.docs:
            distances = [self.euclidean_distance(self.doc_tfidf[doc], centroid) for centroid in centroids]
            min_dis_cen = np.argmin(distances)
            clusters[min_dis_cen].append(doc)
        return clusters

    def build_initial_clusters(self, centroids, k, seed_docs):
        """Build initial clusters based on centroids."""
        clusters = {i: [] for i in range(k)}
        for doc in self.docs:
            if doc not in seed_docs:
                distances = [self.euclidean_distance(self.doc_tfidf[doc], centroid) for centroid in centroids]
                min_dis_cen = np.argmin(distances)
                clusters[min_dis_cen].append(doc)
        return clusters

    def update_centroids(self, clusters, centroids):
        """Update centroids based on cluster means."""
        for c_id, c_docs in clusters.items():
            centroids[c_id] = self.calculate_centroid(c_docs)
        return centroids

    def calculate_rss(self, clusters, centroids):
        """Calculate the Residual Sum of Squares."""
        rss = 0
        for cluster_id in range(len(clusters)):
            for doc in clusters[cluster_id]:
                rss += np.around(np.square(self.euclidean_distance(self.doc_tfidf[doc], centroids[cluster_id])), 3)
        return rss
    
    def kmeans(self, k=5):
        """Perform K-means clustering."""
        clusters = []
        seed_docs = []
        counter = 0
        centroids = self.initialize_centroids(self.docs, k, seed_docs)
        clusters = self.build_initial_clusters(centroids, k, seed_docs)
        new_rss = self.calculate_rss(clusters, centroids)
        rss = float("inf")
        print("Initial Data: \nRSS: {}\nClusters: {}\nCentroids: {}".format(rss, clusters, centroids))
        while new_rss > 0 and new_rss < rss:
            counter += 1
            rss = new_rss
            centroids = self.update_centroids(clusters, centroids)
            clusters = self.build_clusters(centroids, k)
            new_rss = self.calculate_rss(clusters, centroids)
            print("Iteration # {} : \nRSS: {}\nClusters: {}\nCentroids: {}".format(counter, new_rss, clusters, centroids))
        return rss, clusters
    
    def skmeans(self, k=5):
        """Perform K-means clustering with silhouette score."""
        clusters = []
        seed_docs = []
        counter = 0
        centroids = self.initialize_centroids(self.docs, k, seed_docs)
        clusters = self.build_initial_clusters(centroids, k, seed_docs)
        new_rss = self.calculate_rss(clusters, centroids)
        rss = float("inf")
        while new_rss > 0 and new_rss < rss:
            counter += 1
            rss = new_rss
            centroids = self.update_centroids(clusters, centroids)
            clusters = self.build_clusters(centroids, k)
            new_rss = self.calculate_rss(clusters, centroids)
        return rss, clusters

    def start_Kmean(self, k=5):
        """Start K-means clustering."""
        rss = float('inf')
        new_rss, clusters = self.kmeans(k)
        while new_rss > 0 and new_rss < rss:
            rss = new_rss
            new_rss, clusters = self.kmeans(k)
        self.cluster = clusters
        return clusters

    def sil_Kmean(self, k=5):
        """Start K-means clustering with silhouette score."""
        rss = float('inf')
        new_rss, clusters = self.skmeans(k)
        while new_rss > 0 and new_rss < rss:
            rss = new_rss
            new_rss, clusters = self.skmeans(k)
        self.cluster = clusters
        return clusters

    def calculate_purity(self, test_clusters):
        """Calculate purity metric."""
        total_instances = sum(len(cluster) for cluster in test_clusters.values())
        total_correct = 0
        for test_cluster in test_clusters.values():
            max_common = 0
            for golden_cluster in self.golden_cluster.values():
                common = len(set(test_cluster).intersection(golden_cluster))
                if common > max_common:
                    max_common = common
            total_correct += max_common
        self.purity = total_correct / total_instances
        self.total_correct = total_correct
        return self.purity

    def calculate_rand_index(self, test_clusters):
        """Calculate Rand index metric."""
        golden_labels = []
        test_labels = []
        for golden_cluster_id, golden_cluster in self.golden_cluster.items():
            golden_labels.extend([golden_cluster_id] * len(golden_cluster))
            
        for test_cluster_id, test_cluster in test_clusters.items():
            test_labels.extend([test_cluster_id] * len(test_cluster))
        
        self.rand_index = adjusted_rand_score(golden_labels, test_labels)
        return self.rand_index

    def calculate_silhouette_score(self, test_clusters, data):
        """Calculate Silhouette score."""
        labels = []
        for cluster_id, cluster in test_clusters.items():
            labels.extend([cluster_id] * len(cluster))
        
        labels = np.array(labels)
        data = data.reshape(-1, 1)
        self.silhouette_avg = silhouette_score(data, labels)
        return self.silhouette_avg

    def getKmeanEval(self):
        """Evaluate K-means clustering."""
        self.calculate_purity(self.cluster)
        self.calculate_rand_index(self.cluster)
        print("Total Correct Assignments:", self.total_correct)
        print("Purity:", self.purity)
        print("Rand Index:", self.rand_index)
        k_values = range(2, 11)  # Try k from 2 to 10
        silhouette_scores = []  # Initialize list to store silhouette scores
        
        for k in k_values:
            clusters = self.sil_Kmean(k)
            silhouette_avg = self.calculate_silhouette_score(clusters, self.docs)
            silhouette_scores.append(silhouette_avg)

        plt.plot(k_values, silhouette_scores, marker='o')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score vs. Number of Clusters')
        plt.grid(True)
        plt.show()

# Instantiate VSM class
v = VSM()
v.loadtfidf()
v.split_data()
v.classifyKNN()
precision, recall, accuracy, f1_score = v.calculate_metrics()
v.start_Kmean()
v.getKmeanEval()
