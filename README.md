# Heart Disease Prediction Model Using Machine Learning 

# Introduction 
Heart disease is one of the most critical and dangerous illnesses nowadays worldwide, with many people suffering from this disease. To address this issue, we propose to develop a machine learning model that is capable of predicting heart disease based on patient data. This model will help people easily predict their disease. To develop this model, we will use different machine learning algorithms such as logistic regression, decision trees, random forest, Artificial Neural Networks, Support Vector Machines, Decision Tree classifier, K-Nearest Neighbors, but I select the logistic regression because we want to just predication data whether the patient is has the hearth disease there are two classes one is the disease and no disease and for the binary classification the logistic regression is the best selection base on decision and we will also use the python and related libraries to train the model effectively. The model will provide a more efficient, faster, and less expensive solution for diagnosing heart disease.

# Tools and Libraries  
We will use the some tools for this project such as Jupyter note book and python library for this below is the short description of each one 
# 1. NumPy:
NumPy is a powerful Python library for scientific computing and numerical operations. It provides efficient multidimensional array objects, along with various mathematical functions to operate on these arrays. With NumPy, you can perform array manipulation, element-wise operations, linear algebra operations, and mathematical computations efficiently. It is widely used as the foundation for many other libraries in the data science ecosystem.
# 2. Scikit-learn (sklearn):
Scikit-learn is a popular machine learning library in Python. It provides a wide range of algorithms for various tasks such as classification, regression, clustering, and dimensionality reduction. Sklearn is built on top of NumPy and SciPy, making it easy to integrate with other data science libraries. It also offers consistent APIs, making it simple to use and experiment with different models. Sklearn includes functions for data preprocessing, model selection, evaluation metrics, and feature extraction, making it a comprehensive library for machine learning tasks.
# 3. Pandas:
Pandas is a versatile data manipulation and analysis library in Python. It offers robust data structures such as Data Frame and Series, which allow efficient handling of structured data. Pandas provides functions for data cleaning, transformation, merging, and analysis. It enables easy indexing, filtering, and grouping of data, making it convenient to perform operations on datasets. Pandas is commonly used for data preprocessing, exploratory data analysis, and preparing data for machine learning tasks.
# 4. Matplotlib:
Matplotlib is a powerful visualization library in Python. It provides a wide variety of customizable plotting functions and styles, making it suitable for creating publication-quality charts, graphs, and visualizations. With Matplotlib, you can generate line plots, scatter plots, bar plots, histograms, and more. It also offers fine-grained control over plot elements and supports visualization of multiple subplots. Matplotlib is widely used in data analysis, presentations, and creating visualizations to gain insights from data.
# 5. Jupyter Notebook:
Jupyter Notebook is an interactive computing environment that allows you to create and share documents containing code, visualizations, and explanatory text. It provides a web-based interface where you can write and execute code in cells, making it an excellent tool for data exploration, analysis, and collaborative work. Jupyter Notebook supports various programming languages, including Python, R, and Julia. It allows you to combine code execution with rich text elements, such as Markdown, to create dynamic and interactive narratives. Jupyter Notebook is widely used in data analysis, research, and educational settings, providing an accessible and flexible environment for data scientists and analysts.

These libraries, NumPy, sklearn, pandas, and Matplotlib, are essential tools in the machine learning system in Python. They provide the necessary functionality for data manipulation, analysis, machine learning modeling, and visualization, contributing significantly to the effectiveness and efficiency of data-driven projects.

# Data Collection 
Data collection is typically the starting point of the process. Datasets can take various forms, including structured and unstructured data, which may contain missing or noisy information. Each data type and format requires specific methods for data handling and management. In addition to this, it's important to mention that the project will also involve the identification and implementation of data cleaning techniques to ensure the accuracy and reliability of the collected data.
# Data source 
As part of our project, we aimed to collect data from multiple sources for accurate model training. However, due to time constraints, we obtained the dataset from a single online source for use in our machine learning class project. The dataset is publicly available on the Kaggle website and originates from an ongoing study on heart health in Framingham, Massachusetts. The study's primary goal is to predict a patient's 10-year risk of developing coronary heart disease. It includes information on over 4,000 patients and encompasses 15 different attributes, covering demographic, behavioral, and medical factors. We utilized this dataset to train our models and analyze heart health patterns within the scope of our project. If you're interested in accessing the dataset for your own research or projects, you can find the download link below.
{Data Set Link } (https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset/download?datasetVersionNumber=2)
 
## Data Exploration 
In this part where we looked at the data, we checked out the information we got from the Kaggle website about heart health in Framingham, Massachusetts. This information had details about more than 1025 patients and included things like age, behavior, and health info.
We took a close look at how the different details were spread out, if anything was missing, and if there were any really unusual numbers. We also made some graphs to see how the different details were connected to each other and if we could find any patterns.
We also did some basic math to understand the main things about the patients and what could predict if they might have heart problems in the next 10 years. This part helped us get a good idea about the information we had and how we could use it to build and test our models for the project.
Attribute Of the Data Sets
    1.	Age: displays the age of the individual.
    2.	Sex: displays the gender of the individual using the following format: 1 = male 0 = female.
    3.	Chest-pain type : displays the type of chest-pain experienced by the individual using the following format : 1 = typical angina 2 = atypical angina 3 = non - anginal pain 4 = asymptotic
    4.	Resting Blood Pressure : displays the resting blood pressure value of an individual in mmHg (unit)
    5.	Serum Cholestrol : displays the serum cholestrol in mg/dl (unit)
    6.	Fasting Blood Sugar : compares the fasting blood sugar value of an individual with 120mg/dl. If fasting blood sugar > 120mg/dl then : 1 (true) else : 0 (false)
    7.	Resting ECG : 0 = normal 1 = having ST-T wave abnormality 2 = left ventricular hyperthrophy
    8.	Max heart rate achieved : displays the max heart rate achieved by an individual.
    9.	Exercise induced angina : 1 = yes 0 = no
    10.	ST depression induced by exercise relative to rest : displays the value which is integer or float.
    11.	Peak exercise ST segment : 1 = upsloping 2 = flat 3 = downsloping
    12.	Number of major vessels (0-3) colored by flourosopy : displays the value as integer or float.
    13.	Thal : displays the thalassemia : 3 = normal 6 = fixed defect 7 = reversable defect
    14.	Diagnosis of heart disease: Displays whether the individual is suffering from heart disease or not : 0 = absence 1,2,3,4 = present.
