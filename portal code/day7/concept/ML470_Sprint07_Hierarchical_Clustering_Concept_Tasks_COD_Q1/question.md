Problem Statement

Customer segmentation is one of the most important analytical tasks in retail and marketing. Understanding how customers differ in age, income levels, and spending behavior allows businesses to personalize marketing strategies, improve customer experience, and optimize sales. However, meaningful segmentation requires clean, standardized data and a structured clustering approach.

In this task, hierarchical agglomerative clustering is applied to a mall customer dataset using features such as Age, Annual Income, and Spending Score. Before performing clustering, it is necessary to detect and treat outliers and scale features to ensure that all variables contribute equally to the clustering process. Dendrograms are then plotted to visualize how customers cluster together at various distances.

This workflow ensures that the resulting customer groups are meaningful, interpretable, and suitable for targeted marketing insights.

Objectives

1.Explore and prepare customer data

Load the Mall Customers dataset and display the initial rows.
Select key numerical features such as Age, Annual Income, and Spending Score.
Prepare the data for further preprocessing and clustering.

2.Detect outliers in customer features

Use the assess_outliers() module to identify extreme values in numerical attributes.
Examine distributional irregularities that could affect clustering performance.
Visualize or log outlier counts to understand data quality.

3.Treat outliers for better clustering accuracy

Apply the treat_outliers() module to cap or modify extreme values.
Ensure Age, Income, and Spending Score lie within reasonable limits.
Maintain the natural variability of customer behavior while reducing noise.

4.Perform feature scaling

Use the data_scale() function to standardize the treated dataset.
Convert Age, Income, and Spending Score into comparable units.
Produce a scaled dataset suitable for distance-based algorithms such as hierarchical clustering.

5.Generate dendrograms for hierarchical clustering

Construct a dendrogram using Ward linkage to visualize customer group similarity.
Create a second dendrogram with a horizontal threshold line (at distance = 8) to help estimate the optimal number of clusters
Observe how customers merge step-by-step into clusters based on Euclidean distances.

6.Build the foundation for customer segmentation

Use the scaled data and dendrograms to analyze natural customer groupings.
Interpret visual clusters to understand segments such as:
High-income high-spending customers
Low-income low-spending customers
Young vs. old customer groups
Prepare the dataset for applying Agglomerative Clustering or other clustering algorithms.

7.Interpret preprocessing and clustering readiness

Ensure that all extreme values have been handled.
Confirm that scaling has normalized feature influence.
Validate that dendrograms reveal a clear clustering structure.
Prepare insights for targeted marketing or business decision-making.

This preprocessing and hierarchical clustering workflow helps analysts uncover hidden customer patterns, identify meaningful segments, and support data-driven strategies in retail marketing. By properly preparing the data and visualizing its structure, businesses can better understand customer behavior and tailor their approaches accordingly.

Sample Visualization

Bar Chart:
![sample](./sample.png)
