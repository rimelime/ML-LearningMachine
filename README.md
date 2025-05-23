# ML-LearningMachine
HCMUT CO3117 CC01 Machine Learning course repository for group ML-learningMachine
Github Link: https://github.com/rimelime/ML-LearningMachine.git
members, task distributions and contributions for Assignment 1: 
| Name                     | Student ID | Task           | Contributions |
|--------------------------|------------|----------------|---------------|
| Thái Quang Dự            | 2252136    | Neural network | 20%           |
| Đặng Quốc Huy            | 2053031    | Neural network | 20%           |
| Phùng Bá Triều           | 2053524    | Decision Tree  | 20%           |
| Lê Nguyễn Minh Giang     | 2052966    | Naive Bayes    | 20%           |
| Đinh Phạm Đăng Khoa      | 2052132    | Naive Bayes    | 20%           |
# Assignment 2 Report:
| Name                     | Student ID | Task                              | Contributions |
|--------------------------|------------|-----------------------------------|---------------|
| Thái Quang Dự            | 2252136    | Bayesian Network, Ensemble method | 20%           |
| Phùng Bá Triều           | 2053524    | Data preprocessing, SVM           | 20%           |
| Lê Nguyễn Minh Giang     | 2052966    | PCA/LDA, report                   | 20%           |
## 1. Introduction
### 1.1 Objectives
 This project aims to explore and compare the effectiveness of various machine learning
 models in handling structured and unstructured data for predictive analysis and feature
 understanding. The central goal is not only to implement each model accurately but also
 to interpret their behaviors, strengths, and limitations when applied to a shared dataset.
 The selected models represent four different families of machine learning approaches:
 1. Bayesian Networks– a probabilistic graphical model that encodes conditional de
pendencies between variables in the form of a directed acyclic graph (DAG).
 2. Ensemble Methods– techniques that combine predictions from multiple base estima
tors to enhance generalization and robustness, including Random Forest (Bagging),
 AdaBoost and Gradient Boosting (Boosting), and Voting Classifiers.
 3. Support Vector Machines (SVM)– a powerful supervised learning model effective
 in high-dimensional spaces, especially with sparse textual data using linear kernels.
 4. Dimensionality Reduction and Topic Modeling (PCA/LDA)– unsupervised tech
niques to reduce feature space, discover latent semantic structures in text through
 Principal Component Analysis (Truncated SVD) and Latent Dirichlet Allocation.
 By implementing and evaluating all four approaches on the same dataset, we aim to:
 • Compare their predictive performance and resource efficiency.
 • Understand the types of features or data structures each model handles best.
 • Investigate the interpretability and practical deployment potential of each method.
 • Develop insights into how model choice should be influenced by the data and task
 characteristics.
### 1.2 Data Set Overview
 The dataset used for this project is titled Media prediction and its cost.csv, which
 contains detailed records of various media types and their associated campaigns or dis
tribution costs. This data set offers a combination of categorical, textual, and numerical
 features, making it suitable for multifaceted analysis using different machine learning
 paradigms.
 Key characteristics:
 • Size: Over 60,000 rows of real-world data
 • Target feature: A discretized cost value categorized into three ordinal buckets (e.g.,
 low, medium, high cost).
 • Textual input: The media type column, which includes short descriptions such as
 “paper,” “TV,” “radio,” “coupon,” etc.
 • Feature types:
 1. Categorical (e.g., campaign type)
 2. Textual (e.g., media type)
 3. Numerical (e.g., reach, impressions, spend)
 This data set is particularly suitable for:
 • Probabilistic modeling of relationships among features (Bayesian networks).
 • Boosting and bagging techniques to improve classification accuracy (Ensemble meth
ods).
 • Handling high-dimensional text and numeric data (SVM).
 • Latent topic extraction and dimensionality reduction for better interpretability
 (PCA/LDA).
 By applying the same data set across all models, we ensure a consistent basis for
 evaluation and comparison. This also highlights how different techniques exploit different
 aspects of the same data: structure, hierarchy, sparsity, or semantics.
  Maximize 2
## 2. Theoretical Background