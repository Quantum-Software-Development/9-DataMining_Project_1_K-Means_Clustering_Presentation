
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


<br><br>


# [K-Means Clustering Repository Presentation]()

This repository contains the full presentation and step-by-step application of the K-Means clustering algorithm. The goal is to demonstrate the process from data preprocessing, through model evaluation, to the final conclusion about the optimal number of clusters, based on an included PDF presentation. This provides a comprehensive, practical example of unsupervised clustering for educational and analytical purposes.

<br>

# [Use of the K-Means Algorithm]()

This repository contains the step-by-step application of the K-Means clustering algorithm on a dataset, including data preprocessing, model evaluation, and final conclusions.


[ K-Means Clustering Repository Presentation]()

This repository contains the full presentation and step-by-step application of the K-Means clustering algorithm. The goal is to demonstrate the process from data preprocessing, through model evaluation, to the final conclusion about the optimal number of clusters, based on an included PDF presentation. This provides a comprehensive, practical example of unsupervised clustering for educational and analytical purposes.

<br>

# [Use of the K-Means Algorithm]()

This repository contains the step-by-step application of the K-Means clustering algorithm on a dataset, including data preprocessing, model evaluation, and final conclusions.

<br>

## [Data Preprocessing]()

The dataset initially contained 3 columns, but only "Column1" and "Column2" were used after dropping the "Unnamed: 0" index column, resulting in 2 columns and 9,308 rows.

The data plot shows that the dataset is suitable for clustering, with an initial hypothesis of 4 to 6 groups to be validated later using the elbow method and silhouette index.

Notably, there are 2 missing values per column, which were imputed using the mean of each column, as K-Means does not handle missing values.

<br>


```python
df['Column1'] = df['Column1'].fillna(df['Column1'].mean())  \# Fill NaNs with mean
df['Column2'] = df['Column2'].fillna(df['Column2'].mean())  \# Fill NaNs with mean
```

<br>

Duplicate values were identified and removed to avoid redundancy in clustering.

<br>


```python
df = df.drop_duplicates(subset='Column1', keep='first')
```

<br>



## [Data Normalization]()

Since K-Means is sensitive to scale due to its use of distances, MinMaxScaler was applied to scale the data into the [0,1] range.

<br>


```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
standard_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
```

<br>

The normalized data had minimum values of 0 and maximum values of 1 for both columns, ensuring fair comparison.


<br>



## [Choosing the Number of Clusters (K)]()

The Elbow Method was used to explore values of K from 2 to 10 by plotting the inertia (sum of squared distances to cluster centers).

<br>

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
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.axvline(x=3, color='\#D86565', linestyle='--')
plt.axvline(x=5, color='\#D86565', linestyle='--')
plt.show()
```

<br>

Analysis of the plot showed sharp decreases in inertia from K=2 to K=3 and significant further decrease until K=5, after which the curve leveled off, indicating potential K values of 3 or 5.

<br>

## [Silhouette Score Evaluation]()

To quantitatively validate the best K, the silhouette score was calculated for K values 3, 4, and 5.


<br>

```python
from sklearn.metrics import silhouette_score
import pandas as pd

scores = []
for k in :[^3][^4][^5]
kmeans = KMeans(n_clusters=k, random_state=43)
labels = kmeans.fit_predict(standard_df)
scores.append(silhouette_score(standard_df, labels))

pd.DataFrame({'K': , 'Silhouette Score': scores})[^4][^5][^3]
```

<br>


| [K]() | [Silhouette Score]() |
|---|------------------|
| 3 | 0.667            |
| 4 | 0.700            |
| 5 | 0.671            |

<br>

The silhouette score clearly favored \( K=4 \), indicating the best balance of cohesion and separation.

<br>


## [Visualizing Clusters for K=3, 4, and 5]()

Scatter plots were created for clusters with K=3, 4, and 5, including centroids, allowing visual examination.

<br>

```python
fig, axes = plt.subplots(3, 1, figsize=(14, 20))
for ax, k in zip(axes, ):[^5][^3][^4]
kmeans = KMeans(n_clusters=k, random_state=43)
kmeans.fit(standard_df)
standard_df['Cluster'] = kmeans.labels_
sns.scatterplot(data=standard_df, x='Column1', y='Column2', hue='Cluster', palette='Set2', ax=ax)
sns.scatterplot(x=kmeans.cluster_centers_[:, 0], y=kmeans.cluster_centers_[:, 1], color='black', marker='X', s=150, label='Centroids', ax=ax)
ax.set_title(f'K = {k}')
ax.set_xlabel('Column1')
ax.set_ylabel('Column2')
ax.legend(loc='upper left')
plt.show()
```

<br>


### [Visual analysis suggested]():

- K=3 groups clusters well but a large cluster seems to contain two distinct groups.
- 
- K=5 splits this large cluster into two but other cluster separations may be less ideal.
  
- K=4 provided the most natural and clearly separated clusters, matching spatial data distribution intuitively.

<br>


## [Descriptive Statistics by Cluster (K=4)]()

Cluster-wise descriptive statistics were computed to understand characteristics per group.

<br>


[Example snippet]():

<br>

| Cluster | Count | Mean Column1 | Mean Column2 |
|---------|-------|--------------|--------------|
| 0       | 1329  | 8.19         | 6.10         |
| 1       | 5311  | -4.53        | -4.98        |
| 2       | 1331  | 8.93         | -8.13        |
| 3       | 1328  | 0.35         | 9.58         |


<br>


## [Final Conclusion]()

- The Elbow Method was inconclusive, suggesting candidates \( K=3 \) or \( K=5 \).
- 
- The silhouette score clearly favored \( K=4 \) with the highest score of 0.699.
- 
- Visual inspection reinforced the choice of \( K=4 \), showing the most intuitive and spatially distinct clusters.
- 
- Hence, using K-Means with \( K=4 \) is the best-supported decision combining quantitative metrics and visual understanding.

<br>


## [End of Analysis]()

The included presentation PDF and this repository provide a full practical guide to the use of K-Means clustering, from data preparation to model evaluation and selection.




















































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












