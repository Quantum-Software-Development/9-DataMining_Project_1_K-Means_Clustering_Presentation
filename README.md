
<br>

**\[[🇧🇷 Português](README.pt_BR.md)\] \[**[🇺🇸 English](README.md)**\]**


<br><br>

# 9- [Data Mining]()  / [ K-Means Clustering Repository Presentation]()




<!-- ======================================= Start DEFAULT HEADER ===========================================  -->

<br><br>


[**Institution:**]() Pontifical Catholic University of São Paulo (PUC-SP)  
[**School:**]() Faculty of Interdisciplinary Studies  
[**Program:**]() Humanistic AI and Data Science
[**Semester:**]() 2nd Semester 2025  
Professor:  [***Professor Doctor in Mathematics Daniel Rodrigues da Silva***](https://www.linkedin.com/in/daniel-rodrigues-048654a5/)

<br><br>

#### <p align="center"> [![Sponsor Quantum Software Development](https://img.shields.io/badge/Sponsor-Quantum%20Software%20Development-brightgreen?logo=GitHub)](https://github.com/sponsors/Quantum-Software-Development)


<br><br>

<!--Confidentiality statement -->

#

<br><br><br>

> [!IMPORTANT]
> 
> ⚠️ Heads Up
>
> * Projects and deliverables may be made [publicly available]() whenever possible.
> * The course emphasizes [**practical, hands-on experience**]() with real datasets to simulate professional consulting scenarios in the fields of **Data Analysis and Data Mining** for partner organizations and institutions affiliated with the university.
> * All activities comply with the [**academic and ethical guidelines of PUC-SP**]().
> * Any content not authorized for public disclosure will remain [**confidential**]() and securely stored in [private repositories]().  
>


<br><br>

#

<!--END-->




<br><br><br><br>



<!-- PUC HEADER GIF
<p align="center">
  <img src="https://github.com/user-attachments/assets/0d6324da-9468-455e-b8d1-2cce8bb63b06" />
-->


<!-- video presentation -->


##### 🎶 Prelude Suite no.1 (J. S. Bach) - [Sound Design Remix]()

https://github.com/user-attachments/assets/4ccd316b-74a1-4bae-9bc7-1c705be80498

####  📺 For better resolution, watch the video on [YouTube.](https://youtu.be/_ytC6S4oDbM)


<br><br>


> [!TIP]
> 
>  This repository is a review of the Statistics course from the undergraduate program Humanities, AI and Data Science at PUC-SP.
>
> ### ☞ **Access Data Mining [Main Repository](https://github.com/Quantum-Software-Development/1-Main_DataMining_Repository)**
>
>


<!-- =======================================END DEFAULT HEADER ===========================================  -->


<br><br><br>



# [K-Means Algorithm - Clustering - Presentation]()

This repository contains the full presentation and step-by-step application of the K-Means clustering algorithm. The goal is to demonstrate the process from data preprocessing, through model evaluation, to the final conclusion about the optimal number of clusters, based on an included PDF presentation. This provides a comprehensive, practical example of unsupervised clustering for educational and analytical purposes.


<br><br>


[ K-Means Clustering Repository Presentation]()

This repository contains the full presentation and step-by-step application of the K-Means clustering algorithm. The goal is to demonstrate the process from data preprocessing, through model evaluation, to the final conclusion about the optimal number of clusters, based on an included PDF presentation. This provides a comprehensive, practical example of unsupervised clustering for educational and analytical purposes.

<br><br>


## [What is K-Means?]()

K-Means is a popular unsupervised machine learning algorithm used for clustering data. Its primary purpose is to partition a dataset into a pre-specified number of distinct, non-overlapping groups called "clusters." The "K" in K-Means refers to the number of clusters the user wants to identify.

The algorithm works by grouping data points that are similar to each other based on a distance metric, usually Euclidean distance. Each cluster is represented by its centroid, which is the mean position of all points within that cluster. K-Means iteratively adjusts the centroids and reassigns points to clusters until the clusters are stable or a set number of iterations is reached.


<br><br>


## [Type of Algorithm]()

K-Means is an example of a "hard" clustering algorithm because each data point belongs to exactly one cluster. It is an iterative centroid-based clustering method that aims to minimize the within-cluster variance (sum of squared distances from points to their cluster centroid).

Because it is unsupervised learning, it does not require labeled data.


<br><br>


## [When to Use K-Means]()

- When you have a dataset without labels and want to discover natural groupings based on feature similarities.
- When clusters are expected to be spherical or roughly equally sized, as K-Means works best in these cases.
- When you know or can estimate the number of clusters (K) in advance.
- When computational efficiency is important, as K-Means is relatively fast and scalable to large datasets.
- For applications like market segmentation, image compression, document clustering, and pattern recognition.


<br><br>


## [When Not to Use K-Means]()

- If clusters in data are non-spherical, overlapping, or have very different sizes or densities, K-Means may not perform well.
- When the number of clusters K is not known and difficult to estimate.
- When the data contains many outliers, since K-Means is sensitive to outliers which can distort centroids.
- For categorical or non-numeric data without proper encoding or different distance metrics.
- When clusters have complex shapes that cannot be approximated well by centroids.

In these cases, other clustering methods such as DBSCAN, hierarchical clustering, or Gaussian mixture models might be more appropriate.


<br><br>


## [Data Preprocessing]()

The original dataset consisted of multiple columns, but only “Column1” and “Column2” were used for the analysis. The column "Unnamed: 0", which was merely an index without analytical value, was dropped. The final dataset contains 2 columns and 9,308 rows.


<br><br>


## [Data Exploration]()

A plot of the original data was constructed to explore its behavior visually. The visual inspection suggested the data was suitable for clustering using the K-Means model. The initial hypothesis was the presence of 4 to 6 groups. However, this estimate was to be confirmed later through the elbow method and silhouette analysis.


<br><br>

### [Original Data Plot]()

<br><br>

<p align="center">
 <img src="https://github.com/user-attachments/assets/085453d4-d6b1-49eb-9373-531c7510128b" />

<br><br>



## [Data Preprocessing]()

There were 2 missing values per column (9,306 non-null values out of 9,308). Since the K-Means algorithm cannot handle missing values, these were imputed using the mean of each respective column to enable modeling.

The dataset initially contained 3 columns, but only "Column1" and "Column2" were used after dropping the "Unnamed: 0" index column, resulting in 2 columns and 9,308 rows.

The data plot shows that the dataset is suitable for clustering, with an initial hypothesis of 4 to 6 groups to be validated later using the elbow method and silhouette index.

Notably, there are 2 missing values per column, which were imputed using the mean of each column, as K-Means does not handle missing values.


<br><br>


```python
df['Column1'] = df['Column1'].fillna(df['Column1'].mean())  \# Fill NaNs with mean
df['Column2'] = df['Column2'].fillna(df['Column2'].mean())  \# Fill NaNs with mean
```

<br><br>



## [Duplicate Values]()

Duplicate rows were checked in each column to avoid redundant data points in clustering.

Code was used to list duplicates in “Column1” and “Column2” separately:

<br><br>


```python
df = df.drop_duplicates(subset='Column1', keep='first')
```

<br><br>

This resulted in 9,299 rows and 2 columns ready for further processing.


<br><br>


## [Data Normalization]()

K-Means clustering is sensitive to the scale of features since it relies on distance calculations. To avoid magnitude bias, the data was normalized to a [0,1] scale using MinMaxScaler.


```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
standard_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
```


The normalized dataset had minimum values of 0 and maximum values of 1 for both columns, confirming correct scaling.


<br><br>


## [Determining the Number of Clusters (Choosing K)]()

<br>

### The Elbow Method

The Elbow Method analyzes the total within-cluster sum of squares (inertia) for different values of K (number of clusters). The goal is to identify the "elbow" point where the inertia reduction rate sharply declines, indicating an optimal K.

The script runs KMeans clustering for K from 2 to 10 and stores the inertia values:


<br><br>

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

inertia_values = []
for k in range(2, 11):
kmeans = KMeans(n_clusters=k, random_state=42)
inertia_values.append(kmeans.fit(standard_df).inertia_)

plt.figure(figsize=(10, 6))
sns.lineplot(x=range(2, 11), y=inertia_values, marker='o')

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)

plt.title('Elbow Method')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')

plt.axvline(x=3, color='\#D86565', linestyle='--')  \# Candidate K=3
plt.axvline(x=5, color='\#D86565', linestyle='--')  \# Candidate K=5

plt.show()
```

<br><br>

### [Elbow Plot Analysis]()

- There is a sharp drop in inertia from \(K=2\) to \(K=3\).
- The decrease continues but less steep until \(K=5\), after which the curve flattens.
- The plot suggests two potential "elbows" at \(K=3\) and \(K=5\), indicating ambiguity in choosing between these two values solely based on the elbow method.

<br><br>


<p align="center">
 <img src="https://github.com/user-attachments/assets/73998bea-efe5-4d72-8f4b-fa25b021fc80


<br><br>


## [Silhouette Score Evaluation]()

The silhouette score is a metric that evaluates cluster quality by assessing how similar each point is to its own cluster compared to other clusters. Scores range from -1 to 1, where a high positive score indicates well-separated, coherent clusters.

The silhouette scores were calculated for \(K=3, 4, 5\):


<br><br>


```python
from sklearn.metrics import silhouette_score
import pandas as pd

scores = []
for k in :
kmeans = KMeans(n_clusters=k, random_state=43)
labels = kmeans.fit_predict(standard_df)
scores.append(silhouette_score(standard_df, labels))

pd.DataFrame({'K': , 'Silhouette Score': scores})
```

<br><br>


| [K]() | [Silhouette Score]() |
|---|------------------|
| 3 | 0.667            |
| 4 | 0.700            |
| 5 | 0.671            |



<br>

The silhouette score clearly favors \(K=4\), showing the best balance of cluster cohesion and separation among tested values.


<br><br>


## [Visual Cluster Analysis]()

Scatter plots of clusters for \(K=3\), \(K=4\), and \(K=5\) were generated, including marked centroids, for intuitive visual evaluation.


<br><br>


```python
import seaborn as sns
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(14, 20))
for ax, k in zip(axes, ):
kmeans = KMeans(n_clusters=k, random_state=43)
kmeans.fit(standard_df)

    data_with_clusters = standard_df.copy()
    data_with_clusters['Cluster'] = kmeans.labels_
    
    sns.scatterplot(data=data_with_clusters, x='Column1', y='Column2', hue='Cluster', palette='Set2', legend='full', ax=ax)
    sns.scatterplot(x=kmeans.cluster_centers_[:, 0], y=kmeans.cluster_centers_[:, 1], s=150, color='black', marker='X', label='Centroids', ax=ax)
    
    ax.set_title(f'K = {k}')
    ax.set_xlabel('Column 1')
    ax.set_ylabel('Column 2')
    ax.legend(loc='upper left')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
```

<br><br>


<p align="center">
 <img src="https://github.com/user-attachments/assets/4846f376-9899-44d2-a17e-40adcda974a1" /<


<br><br>










































<br><br><br><br>
<br><br><br><br>
<br><br><br><br>
<br><br><br><br>
<br><br><br><br>


<!-- ========================== [Bibliographr ====================  -->

<br><br>


## [Bibliography]()


[1](). **Castro, L. N. & Ferrari, D. G.** (2016). *Introdução à mineração de dados: conceitos básicos, algoritmos e aplicações*. Saraiva.

[2](). **Ferreira, A. C. P. L. et al.** (2024). *Inteligência Artificial - Uma Abordagem de Aprendizado de Máquina*. 2nd Ed. LTC.

[3](). **Larson & Farber** (2015). *Estatística Aplicada*. Pearson.


<br><br>


<!-- ======================================= Start Footer ===========================================  -->


<br><br>


## 💌 [Let the data flow... Ping Me !](mailto:fabicampanari@proton.me)

<br><br>



#### <p align="center">  🛸๋ My Contacts [Hub](https://linktr.ee/fabianacampanari)


<br>

### <p align="center"> <img src="https://github.com/user-attachments/assets/517fc573-7607-4c5d-82a7-38383cc0537d" />




<br><br><br>

<p align="center">  ────────────── 🔭⋆ ──────────────


<p align="center"> ➣➢➤ <a href="#top">Back to Top </a>

<!--
<p align="center">  ────────────── ✦ ──────────────
-->



<!-- Programmers and artists are the only professionals whose hobby is their profession."

" I love people who are committed to transforming the world "

" I'm big fan of those who are making waves in the world! "

##### <p align="center">( Rafael Lain ) </p>   -->

#

###### <p align="center"> Copyright 2025 Quantum Software Development. Code released under the [MIT License license.](https://github.com/Quantum-Software-Development/Math/blob/3bf8270ca09d3848f2bf22f9ac89368e52a2fb66/LICENSE)












