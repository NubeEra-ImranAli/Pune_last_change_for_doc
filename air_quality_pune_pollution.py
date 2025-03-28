# -*- coding: utf-8 -*-
"""Air Quality Pune Pollution Patterns

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/12VX_Kz9C7d0HDo4-BfXTwidgic-M8n44

# Title: Pune Air Quality Prediction using Machine Learning
"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
import folium
from tqdm import tqdm

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score

from scipy.stats import norm, anderson

# loading dataset of smartcity Pune
df = pd.read_csv(r"D:\nubeera\Pune\Dataset - Copy.csv")

print(df.shape)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
data = df
# Print the columns to verify their names and contents
print("Columns in DataFrame:", data.columns)

# Example: if the AQI is represented as 'AirQualityIndex' in your data
# Adjust the target variable and features accordingly
target = 'CO2_MIN'
features = ['PM10_MAX', 'PM2_MAX', 'NO2_MAX', 'CO_MAX', 'OZONE_MAX']

# Handle missing values (replace NaN with mean of the column)
imputer = SimpleImputer(strategy='mean')
data[features] = imputer.fit_transform(data[features])

# Initialize an empty DataFrame to store predictions
predictions_df = pd.DataFrame(columns=['NAME', 'Predicted_AQI'])

# Iterate over each unique NAME in the dataset
for name in data['NAME'].unique():
    # Filter data for the current NAME
    subset_data = data[data['NAME'] == name]

    # Prepare the features and target variables for the current subset
    X = subset_data[features]
    y = subset_data[target]

    #Handle missing values in the target variable 'y'
    y = y.fillna(y.mean()) # Fill NaN with the mean of the 'CO2_MIN' column for the current subset

    # Split data into training and testing sets (if needed)
    # For demonstration, assuming all data is used for training
    X_train, X_test, y_train, y_test = X, None, y, None

    # Initialize a linear regression model (or any other model)
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions for the current subset
    subset_predictions = model.predict(X_train)  # Use X_test for actual testing

    # Store the predictions in predictions_df
    subset_results = pd.DataFrame({
        'NAME': [name] * len(subset_predictions),
        'Predicted_AQI': subset_predictions
    })
    predictions_df = pd.concat([predictions_df, subset_results], ignore_index=True)

# Print or save predictions_df as needed
print(predictions_df.head())  # Print first few rows as an example

"""# Data Preprocessing"""

df.info()

# chaging datatype of "LASTUPDATEDATETIME" into "to_datetime" datatype
df['LASTUPDATEDATETIME'] = pd.to_datetime(df['LASTUPDATEDATETIME'])

df.describe()

df["NO_MIN"].unique(), df["NO_MAX"].unique()

"""NO_MIN and NO_MAX column are only containing entries value "0" of int64 datatype. So these columns of feature is dropped"""

df = df.drop(['NO_MAX', 'NO_MIN'], axis=1)

"""## Missing Values and Outlier

**computing the percentage of missing data in each column (feature)**
"""

# computing percentage of nan entry in each column of features
percentage_missing= df.isnull().sum()*100/len(df)
percentage_missing

"""There is no substantial amount of missing entry in any column so we will fill the missing value.  

"""

# Location name in Pune smart city and there corresponding Lattitudes and Longitudes.

print(f"\n\Location: {df['NAME'].unique()} \n total no. of unique entries {np.size(df['NAME'].unique())}")

print(f"\n\nLattitudes: {df['Lattitude'].unique()} \n total no. of unique entries {np.size(df['Lattitude'].unique())}")

print(f"\n\nLongitude: {df['Longitude'].unique()} \n total no. of unique entries {np.size(df['Longitude'].unique())}")

print("Location: \tnumber of entries")
for location in df["NAME"].unique():
    df_location = df.loc[df.NAME == location]
    print(f"{location}: {df_location.shape}")

"""#### Histogram plot of one location (df_BopadiSquare_65) in Pune smart city"""

# taking 'BopadiSquare_65' data to plot histogram.
df_BopadiSquare_65 = df.loc[df.NAME == 'BopadiSquare_65']

features_to_exclude = ['NAME', 'LASTUPDATEDATETIME', 'Lattitude', 'Longitude']
features_to_visualize = list(set(df_BopadiSquare_65.columns) - set(features_to_exclude))

# subplots for each feature
n_features_to_visualize = len(features_to_visualize)

#the number of rows needed
n_rows = (n_features_to_visualize // 2) + (n_features_to_visualize % 2)

fig, axes = plt.subplots(nrows=n_rows, ncols=2, figsize=(20, 18))

for i, feature in enumerate(features_to_visualize):
    row, col = divmod(i, 2)
    ax = axes[row, col]

    sns.histplot(df_BopadiSquare_65[feature], kde=True, ax=ax)
    ax.set_title(f'Distribution of {feature}')


#     # Fit a normal distribution to the data
#     mu, std = norm.fit(df_BopadiSquare_65[feature])
#     xmin, xmax = ax.get_xlim()
#     x = np.linspace(xmin, xmax, 100)
#     p = norm.pdf(x, mu, std)

# will remove empty subplots if the number of features is odd (in this case number of features are even)
if n_features_to_visualize % 2 != 0:
    fig.delaxes(axes[n_rows - 1, 1])

plt.tight_layout()
plt.savefig("histogram_data_dist.pdf", format="pdf")
plt.show()

"""This is the histohgram of plot of environmental features at location BopadiSquare_65 in Pune smart city. These data are not normally distributed, so filling of missing entries of these features with of their coressponding mean is not a good idea. And also imputation with mean value is susceptible to outlier.

**In our data there is 10 locatioin in Pune smart city and every minute sensor is recording environmental data of all these location and updating into the dataset. So idea is to arrange environmental data of each location together and do filling of missing value based on the median of dataset of only that particular location/station.**
"""

df.isna().sum()

df1 = df.copy()
# features of which nan entries will get imputed
features_to_fill_nan = list(set(df1.columns.unique()) - set(['NAME', 'LASTUPDATEDATETIME', 'Lattitude', 'Longitude']))

for location in df1["NAME"].unique(): # This will iterate through location
    df1_location = df1[df1.NAME == location].copy()  # making a copy of dataset related to where this particular location present in the origina; dataframe df1

    #in one time it will impute of nan entries of one feature of location and then another and so on
    # and after looping through all, loop will complete and another location come and again this will iterate through
    # the features of this current location and so on like this.
    for feature in list(set(df1_location.columns) - set(['NAME', 'LASTUPDATEDATETIME', 'Lattitude', 'Longitude'])):
        df1_location[feature].fillna(df1_location[feature].median(), inplace=True)

    df1.loc[df1.NAME == location] = df1_location


df1 = df1.reset_index(drop = True)

df1 #Nan entries is imputed

"""#### Sampling data hourly"""

df1

# Unique location names
dataframes = df1["NAME"].unique()

dataframes

resample_features = list(set(df1.columns.unique()) - set(['NAME', 'Lattitude', 'Longitude', 'LASTUPDATEDATETIME']))

resample_features

import pandas as pd

# Unique location names
dataframes = df1["NAME"].unique()

# Features for resampling
resample_features = list(set(df1.columns.unique()) - set(['NAME', 'Lattitude', 'Longitude', 'LASTUPDATEDATETIME']))

# Create a list to store the individual DataFrames
dfs = []

# Iterate over each unique location name
for location_name in dataframes:
    # Filter the original DataFrame for the current location
    df_location = df1[df1['NAME'] == location_name].copy()

    # Extract location-specific information
    latitude = df_location['Lattitude'].iloc[0]
    longitude = df_location['Longitude'].iloc[0]

    # Set the index and resample for the current location
    df_location.set_index('LASTUPDATEDATETIME', inplace=True)
    df_c = df_location[resample_features].resample('H').mean()

    # Reset the index to bring 'LASTUPDATEDATETIME' back as a column
    df_c.reset_index(inplace=True)

    # Add location-specific information to the result_df
    df_c['NAME'] = location_name
    df_c['Lattitude'] = latitude
    df_c['Longitude'] = longitude
    df_c = df_c.dropna()

    # Append the results to the list
    dfs.append(df_c)

# Concatenate all the individual DataFrames into one
df2 = pd.concat(dfs, ignore_index=True)

# Display the final DataFrame
print(df2)

df2 # This dataframe contain hourly data

df2

"""## Robust Scaling and Removing Outlier  
Our Data is not normally distributed, and robust scaling is suitable for non-normal distributed datas and outlier has no significant effect on robust scaling.

#### Since each location environmental data collected from not the same sensor so it should be scaled location wise data.
"""

AQI_features = ['PM10_MAX', 'PM10_MIN', 'PM2_MAX', 'PM2_MIN', 'CO_MAX', 'CO_MIN', 'CO2_MAX', 'CO2_MIN', 'SO2_MIN', 'SO2_MAX', 'NO2_MAX', 'NO2_MIN']

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest

def scale_and_remove_outliers(data, location_column="NAME", datetime_column="LASTUPDATEDATETIME", scaler_quantile_range=(25.0, 75.0), contamination=0.05, exclude_columns=[], outlier_features=[]):
    """
        this function will scale the data for each location separately using
          RobustScaler and remove outliers using Isolation Forest.
    """

    # Excluding datetime columns from scaling
    non_datetime_columns = [col for col in data.columns if col != datetime_column]
    features = data[non_datetime_columns].drop(columns=[location_column] + exclude_columns)

    # Scaling the data using RobustScaler
    robust_scaler = RobustScaler(quantile_range=scaler_quantile_range)
    scaled_features = robust_scaler.fit_transform(features)

    # Detecting outliers using Isolation Forest
    isolation_forest = IsolationForest(contamination=contamination, random_state=42)
    outlier_labels = isolation_forest.fit_predict(scaled_features)

    # only the non-outlier data
    non_outliers = data[outlier_labels == 1].copy()  # Create a copy

    # Outlier detection using IQR and removal
    Q1 = non_outliers[outlier_features].quantile(0.25)
    Q3 = non_outliers[outlier_features].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # Filtering indices that exist in the DataFrame
    lower_array = non_outliers.index[(non_outliers[outlier_features] <= lower).any(axis=1)].tolist()
    upper_array = non_outliers.index[(non_outliers[outlier_features] >= upper).any(axis=1)].tolist()

    non_outliers.drop(index=lower_array, inplace=True)

    # datetime_column  Retaining
    non_outliers[datetime_column] = data.loc[non_outliers.index, datetime_column]

    return non_outliers.reset_index(drop=True)

df3 = scale_and_remove_outliers(df2, exclude_columns=['Lattitude', 'Longitude'], outlier_features=AQI_features)
df3.info()

AQI_features = ['PM10_MAX', 'PM10_MIN', 'PM2_MAX', 'PM2_MIN', 'CO_MAX', 'CO_MIN', 'CO2_MAX', 'CO2_MIN', 'SO2_MIN', 'SO2_MAX', 'NO2_MAX', 'NO2_MIN']

plt.figure(figsize=(12, 6))
sns.boxplot(data=df3[AQI_features])
plt.title('AQI')
plt.xticks(rotation=45)

plt.show()

AQI_features = ['PM10_MAX', 'PM10_MIN', 'PM2_MAX', 'PM2_MIN', 'CO_MAX', 'CO_MIN', 'CO2_MAX', 'CO2_MIN', 'SO2_MIN', 'SO2_MAX', 'NO2_MAX', 'NO2_MIN']

# IQR
Q1 = df3[AQI_features].quantile(0.25)
Q3 = df3[AQI_features].quantile(0.75)
IQR = Q3 - Q1

# upper and lower limits
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR

# arrays of Boolean values indicating the outlier rows
upper_array = np.where(df3[AQI_features]>=upper)[0]
lower_array = np.where(df3[AQI_features]<=lower)[0]

# Removing the outliers
df3.drop(index=upper_array, inplace=True)
df3.drop(index=lower_array, inplace=True)

print("New Shape: ", df3.shape)
df3 = df3.reset_index(drop= True)

AQI_features = ['PM10_MAX', 'PM10_MIN', 'PM2_MAX', 'PM2_MIN', 'CO_MAX', 'CO_MIN', 'CO2_MAX', 'CO2_MIN', 'SO2_MIN', 'SO2_MAX', 'NO2_MAX', 'NO2_MIN']

plt.figure(figsize=(12, 6))
sns.boxplot(data=df3[AQI_features])
plt.title('AQI')

plt.xticks(rotation=45)

plt.show()

df3_dict = dict()
for loaction in df3["NAME"].unique():
    df3_dict[location] = df3[df3.NAME == location]

df4 = df3.copy()
# Calculating Excel serial numbers and add a new column 'second'
df4['second'] = (df4['LASTUPDATEDATETIME'] - pd.Timestamp("1899-12-30")) / pd.Timedelta('1D')
df4 = df4.reset_index(drop = True)

print("Location: \tnumber of entries")
for location in df4["NAME"].unique():
    df_location = df4.loc[df4.NAME == location]
    print(f"{location}: {df_location.shape}")

"""# Exploratory Data Analysis

### Correlation between features
"""

# Filter for a specific location, e.g., 'BopadiSquare_65'
df_BopadiSquare_65 = df3[df3['NAME'] == 'BopadiSquare_65']

# Select only numeric columns for correlation
numeric_cols = df_BopadiSquare_65.select_dtypes(include=[float, int]).columns

# Check correlation between environmental features for the specific location
corr_matrix = df_BopadiSquare_65[numeric_cols].corr()

# Plot the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix Heatmap for BopadiSquare_65")
plt.savefig("correlation_matrix_plot.png", format="png")
plt.show()

"""From correlation matrix it can be concluded:  
"Pressure", "Light" and "Sound" has no much effect on other features,  
  
We are focusing on max value recorded each minute by sensors as it is the extreme which will affect more livelihood.

## Plotting Temperature vs Time for different location
"""

# Unique location names
dataframes = df4["NAME"].unique()

  # Features for resampling
resample_features = list(set(df4.columns.unique()) - set(['NAME', 'Lattitude', 'Longitude', 'LASTUPDATEDATETIME']))

  # Create an empty DataFrame to store the results
df5 = pd.DataFrame()

  # Iterate over each unique location name
for location_name in dataframes:
    # Filter the original DataFrame for the current location
    df_location = df4[df4['NAME'] == location_name].copy()

    # Extract location-specific information
    latitude = df_location['Lattitude'].iloc[0]
    longitude = df_location['Longitude'].iloc[0]

    # Set the index and resample for the current location
    df_location.set_index('LASTUPDATEDATETIME', inplace=True)
    df_c = df_location[resample_features].resample('24H').mean()

    # Reset the index to bring 'LASTUPDATEDATETIME' back as a column
    df_c.reset_index(inplace=True)

    # Add location-specific information to the result_df
    df_c['NAME'] = location_name
    df_c['Lattitude'] = latitude
    df_c['Longitude'] = longitude
    df_c = df_c.dropna()

    # Append the results to df5
    df5 = pd.concat([df5, df_c], ignore_index=True)

  # Now df5 contains the resampled and mean values with location information, and 'LASTUPDATEDATETIME' as a time column

"""### Temperature vs time plot"""

# Grouping Data by Location
grouped_data = df5.groupby('NAME')

# Plotting
plt.figure(figsize=(15, 8))
for location, group in grouped_data:
    # Convert datetime.time to string for plotting
    plt.plot(range(len(group['second'])), group['TEMPRATURE_MAX'], label=location)

plt.title('Temperature Variation According to Time')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xticks(rotation=45)
plt.savefig("temperature_vs_time.png", format="png")
plt.show()

# Grouping Data by Location
grouped_data = df5.groupby('NAME')
cluster_1_locations = ['Lullanagar_Square_14',
       'Hadapsar_Gadital_01', 'PMPML_Bus_Depot_Deccan_15', 'Rajashri_Shahu_Bus_stand_19']

cluster_2_locations = ['BopadiSquare_65', 'Karve Statue Square_5',
       'Goodluck Square_Cafe_23', 'Chitale Bandhu Corner_41',
       'Pune Railway Station_28','Dr Baba Saheb Ambedkar Sethu Junction_60']

# Plotting
plt.figure(figsize=(15, 8))
for location, group in grouped_data:
    if location in cluster_1_locations:
        plt.plot(range(len(group['second'])), group['TEMPRATURE_MAX'], label=location)

plt.title('Temperature Variation According to Time')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xticks(rotation=45)
plt.show()



# Plotting
plt.figure(figsize=(15, 8))
for location, group in grouped_data:
    if location in cluster_2_locations:
        plt.plot(range(len(group['second'])), group['TEMPRATURE_MAX'], label=location)

plt.title('Temperature Variation According to Time')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xticks(rotation=45)
plt.show()

"""# Clustering

### Clustering of locations in Pune smart city
"""

location_features = ["NAME", "Lattitude", "Longitude"]
df_locations = df4[location_features]
df_locations  = df_locations.drop_duplicates()
df_locations = df_locations.reset_index(drop=True)
lat_long = df_locations[["Lattitude", "Longitude"]]

"""## Using K_Means"""

squared_distances_sum = []
K = range(1, 10)

for num_clusters in K:
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    kmeans.fit(lat_long)
    squared_distances_sum.append(kmeans.inertia_)

# Plotting the elbow curve
plt.plot(K, squared_distances_sum, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Sum of squared distances/Inertia')
plt.title('Elbow Method For Optimal k')
plt.savefig("lat_long_kmeans_elbow.png", format="png")
plt.show()

# Applying k-means clustering
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)  # Explicitly setting n_init
df_locations['Cluster'] = kmeans.fit_predict(df_locations[['Lattitude', 'Longitude']]) # Add cluster labels to df_locations

# Plotting the clusters
plt.figure(figsize=(10, 6))
for cluster_label in range(num_clusters):
    cluster_data = df_locations[df_locations['Cluster'] == cluster_label]
    plt.scatter(cluster_data['Lattitude'], cluster_data['Longitude'], label=f'Cluster {cluster_label}')

# Adding location names to the plot
for i, row in df_locations.iterrows():
    plt.text(row['Lattitude'], row['Longitude'], row['NAME'], fontsize=8, ha='right')

plt.title('Location Clustering')
plt.xlabel('Lattitude')
plt.ylabel('Longitude')
plt.legend()
plt.savefig("lat_long_kmeans_location_clustering.png", format="png")
plt.show()

import folium
from tqdm import tqdm

# Assuming df_locations is your DataFrame with "Latitude", "Longitude", "Cluster", and "NAME" columns
# Assuming Pune coordinates
pune_lat = 18.5204
pune_long = 73.8567

m_pune = folium.Map([pune_lat, pune_long], zoom_start=12)

radius = 5
cluster_color = ["red", "blue", "black", "yellow"]

def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

# Filter DataFrame for Pune
# df_pune_locations = df_locations['NAME']

for i in tqdm(range(len(df_locations))):
    lat = df_locations.iloc[i]['Lattitude']
    long = df_locations.iloc[i]['Longitude']
    cluster_index = df_locations.iloc[i]['Cluster']

    popup_text = df_locations.iloc[i]['NAME']

    folium.CircleMarker(
        location=[lat, long],
        radius=radius,
        color=cluster_color[cluster_index],
        stroke=False,
        fill=True,
        fill_opacity=0.6,
        opacity=1,
        popup=folium.Popup(popup_text)
    ).add_to(m_pune)

m_pune.save("folium_map.html")
m_pune

"""# Dimension Reduction

## Prinicipal Component Analysis
"""

AQI_features = ['PM10_MAX', 'PM10_MIN', 'PM2_MAX', 'PM2_MIN', 'CO_MAX', 'CO_MIN', 'CO2_MAX', 'CO2_MIN', 'SO2_MIN', 'SO2_MAX', 'NO2_MAX', 'NO2_MIN']

df_AQI = df5[AQI_features]

X = df_AQI.values

# feature scaling (standardisation)
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# from sklearn.preprocessing import RobustScaler
# scaler = RobustScaler()
# X_scaled = scaler.fit_transform(X)



# Principal Component Analysis (PCA)
covariance_matrix = np.cov(X_standardized.T)
covariance_matrix

evals, eig_vectors = np.linalg.eig(covariance_matrix)

print(f"Evals: {evals}")

print(f"Eigen vectors: {eig_vectors}")

variances = []
for i in range(len(evals)):
    variances.append(evals[i]/np.sum(evals))

print(np.sum(variances), '\n', variances)

np.sum(variances[0:3])

# Sorted eigenvalues and corresponding eigenvectors in descending order
sorted_indices = np.argsort(evals)[::-1]
evals = evals[sorted_indices]
eig_vectors = eig_vectors[:, sorted_indices]

# percentage of variance explained by each principal component
variances = evals / np.sum(evals)

# plot
percent_variance = np.round(variances * 100, decimals=2)
columns = [f'PC{i+1}' for i in range(len(percent_variance))]

plt.bar(x=range(1, len(percent_variance) + 1), height=percent_variance, tick_label=columns)
plt.ylabel('Percentage of Variance Explained')
plt.xlabel('Principal Component')
plt.title('PCA Scree Plot')
plt.savefig("AQI_PCA_percengtage_of_variance.png", format="png")
plt.show()

projected_1 =  X_standardized.dot(eig_vectors.T[0])
projected_2 = X_standardized.dot(eig_vectors.T[1])
projected_3 = X_standardized.dot(eig_vectors.T[2])
res = pd.DataFrame(projected_1, columns=['PC1'])
res['PC2'] = projected_2
res['PC3'] = projected_3
res['Y'] = df5['NAME']
res.head(10)


plt.figure(figsize=(20, 10))
sns.scatterplot(x=res['PC1'], y=[0] * len(res), hue=res['Y'], s=200)
plt.savefig("kmeans_AQI_PCA_1D.png", format="png")

plt.figure(figsize=(20, 10))
sns.scatterplot(x=res['PC1'], y=res['PC2'], hue=res['Y'], s=100)
plt.savefig("kmeans_AQI_PCA_2D.png", format="png")

from sklearn.preprocessing import LabelEncoder
from mpl_toolkits.mplot3d import Axes3D

# location names to numerical labels
label_encoder = LabelEncoder()
res['Label'] = label_encoder.fit_transform(res['Y'])

# 3D Scatter Plot
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(res['PC1'], res['PC2'], res['PC3'], c=res['Label'], s=100, alpha=0.7)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

# legend for each location
legend_handles = []
for label in label_encoder.classes_:
    indices = res['Y'] == label
    handle = ax.scatter(res.loc[indices, 'PC1'], res.loc[indices, 'PC2'], res.loc[indices, 'PC3'], label=label, s=100)
    legend_handles.append(handle)

ax.legend(handles=legend_handles, title='Location Names', loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5)
colorbar = fig.colorbar(scatter, ax=ax, label='Location Labels')
fig.canvas.draw()

plt.savefig("kmeans_AQI_PCA_3D.png", format="png")
plt.show()

# Commented out IPython magic to ensure Python compatibility.
#lets apply PCA with n_components = 12
pca = PCA(n_components=12)

# Fit PCA to the data
pca.fit(X_standardized)

# Now access the number of components
print(pca.n_components_)

# Access the explained variance ratio
explained_variance = pca.explained_variance_ratio_
print(explained_variance)

# Sum of the explained variance ratio for the first 3 components
print(sum(pca.explained_variance_ratio_[0:3]))
#lets visualize the explained variance ratio.
percent_variance = np.round(pca.explained_variance_ratio_* 100, decimals =2)
columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12']
plt.bar(x= range(1,13), height=percent_variance, tick_label=columns)
plt.ylabel('Percentate of Variance Explained')
plt.xlabel('Principal Component')
plt.title('PCA Scree Plot')
plt.show()

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

"""# MDS"""
import time
# Commented out IPython magic to ensure Python compatibility.
# Initialize MDS with 3 components and set normalized_stress to 'auto' explicitly
# Assuming X is your data
mds = MDS()

# Start timing
start_time = time.time()

# Fit and transform the data
reduced_features = mds.fit_transform(X)

# Calculate the elapsed time
elapsed_time = time.time() - start_time
print(f"Time taken to fit and transform: {elapsed_time:.4f} seconds")

plt.figure(figsize=(20, 10))
sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=df5["NAME"], s=100)
plt.title('MDS Visualization in 2D')
plt.xlabel('MDS Dimension 1')
plt.ylabel('MDS Dimension 2')
plt.savefig("MDS_AQI_visual.png", format="png")
plt.show()

"""## Locations of Pune Smart City Based on Similar Air Quality"""

# These already has been called in above section
# air pollutants
# AQI_features = ['PM10_MAX', 'PM10_MIN', 'PM2_MAX', 'PM2_MIN',
#                     'CO_MAX', 'CO_MIN', 'CO2_MAX', 'CO2_MIN',
#                     'SO2_MIN', 'SO2_MAX', 'NO2_MAX', 'NO2_MIN']

# df_AQI = df5[AQI_features_max]
# X = df_AQI.values
# # df_AQI_scaled = (df_AQI - df_AQI.mean()) / df_AQI.std()
# df_AQI_scaled.describe()

# Elbow method
squared_distances_sum = []
K = range(1, 10)

for num_clusters in K:
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    kmeans.fit(df_AQI)
    squared_distances_sum.append(kmeans.inertia_)

# Plotting the elbow curve
plt.plot(K, squared_distances_sum, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Sum of squared distances/Inertia')
plt.title('Elbow Method For Optimal k')
plt.savefig("elbow_for_optimal_k_AQI.png", format="png")
plt.show()

# Commented out IPython magic to ensure Python compatibility.
# Applying KMeans clustering to the reduced features
# %time
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df5['cluster_MDS_KMeans'] = kmeans.fit_predict(reduced_features)

plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=df5['cluster_MDS_KMeans'], cmap='viridis', s=30)
plt.title('MDS + KMeans Clustering', size=20)
plt.xlabel('MDS Dimension 1')
plt.ylabel('MDS Dimension 2')
plt.colorbar(label='Cluster')
plt.savefig("kmeans_with_MDS_2D.png", format="png")
plt.show()

plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=df5['cluster_MDS_KMeans'], cmap='viridis', s=30, label='Data Points')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='o', s=200, label='Cluster Centroids')
plt.title('MDS + KMeans Clustering', size=20)
plt.xlabel('MDS Dimension 1')
plt.ylabel('MDS Dimension 2')
plt.legend()
plt.savefig("kmeans_with_MDS_2D_with_centroid.png", format="png")
plt.show()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(reduced_features[:, 0], reduced_features[:, 1], c=df5['cluster_MDS_KMeans'], cmap='viridis', s=30)

ax.set_title('MDS + KMeans Clustering', size=20)
ax.set_xlabel('MDS Dimension 1')
ax.set_ylabel('MDS Dimension 2')
ax.set_zlabel('MDS Dimension 3')
plt.savefig("kmeans_with_MDS_3D.png", format="png")
plt.show()

"""# K-Means  without MDS"""

X = df5[['PM10_MAX', 'PM10_MIN']]  # Adjust with your actual feature columns

# Number of clusters
n_clusters = 3

# Initialize KMeans with explicit n_init
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

# Fit KMeans and predict clusters
df5['cluster'] = kmeans.fit_predict(X)
print(df5.info())
# Plot the clusters
plt.scatter(df5['PM10_MAX'], df5['PM10_MIN'], c=df5['cluster'], cmap='viridis')
plt.xlabel('PM10_MAX')
plt.ylabel('PM10_MIN')
plt.title(f'K-Means Clustering (k={n_clusters})')
plt.colorbar(label='Cluster')
plt.savefig("kmeans_without_MDS_2D.png", format="png")
plt.show()

X = df5[['PM10_MAX', 'PM10_MIN', 'PM2_MAX']]  # Adjust with your actual feature columns

# Number of clusters
n_clusters = 3

# Initialize KMeans with explicit n_init
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

# Fit KMeans and predict clusters
df5['cluster'] = kmeans.fit_predict(X)

# Plotting in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df5['PM10_MAX'], df5['PM10_MIN'], df5['PM2_MAX'], c=df5['cluster'], cmap='viridis')

ax.set_xlabel('PM10_MAX')
ax.set_ylabel('PM10_MIN')
ax.set_zlabel('PM2_MAX')
ax.set_title(f'K-Means Clustering (k={n_clusters})')
plt.savefig("kmeans_without_MDS_3D.png", format="png")
plt.show()

"""# Spectral Clustering"""

df_spectral = df_AQI # df_AQI is already created

# Log-transformation
df_spectral_log = np.log1p(df_spectral)  # Avoiding issues with zero values, if any

# Standardize the log-transformed data
X_spectral_log = StandardScaler().fit_transform(df_spectral_log)

# Silhouette scores for different numbers of clusters
n_clusters_range = range(2, 10)
silhouette_scores = []

for n_clusters in n_clusters_range:
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
    labels = spectral.fit_predict(X_spectral_log)
    silhouette_scores.append(silhouette_score(X_spectral_log, labels))

# Plot the silhouette scores
plt.plot(n_clusters_range, silhouette_scores, marker='o')
plt.title('Silhouette Score for Spectral Clustering')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.savefig("soulhette's score.png", format="png")
plt.show()

"""Silhoutte's score is maximum for four clusters."""

# Perform MDS
mds_spectral = MDS(n_components=3, random_state=42, normalized_stress='auto')
reduced_features_spectral = mds_spectral.fit_transform(X_spectral_log)

# Perform Spectral Clustering
n_clusters_spectral = 3  # Number of clusters
spectral = SpectralClustering(n_clusters=n_clusters_spectral, affinity='nearest_neighbors', random_state=42)

# Fit and predict clusters, ensuring df_spectral is correctly assigned
df_spectral['spectral_cluster'] = spectral.fit_predict(X_spectral_log)

# Visualize in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(reduced_features_spectral[:, 0], reduced_features_spectral[:, 1], reduced_features_spectral[:, 2],
           c=df_spectral['spectral_cluster'], cmap='viridis', s=30)
ax.set_xlabel('MDS Dimension 1')
ax.set_ylabel('MDS Dimension 2')
ax.set_zlabel('MDS Dimension 3')
ax.set_title('MDS + Spectral Clustering in 3D')
plt.savefig("spectral_clustering_3D.png", format="png")
plt.show()

from sklearn.preprocessing import PolynomialFeatures

# Initialize an empty DataFrame to store predictions
predictions_df = pd.DataFrame()

for name in data['NAME'].unique():
    subset_data = data[data['NAME'] == name]

    X = subset_data[features]
    y = subset_data[target]

    X = X.fillna(X.mean())
    y = y.fillna(y.mean())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    polynomial_features = PolynomialFeatures(degree=2)
    X_poly_train = polynomial_features.fit_transform(X_train)
    X_poly_test = polynomial_features.fit_transform(X_test)

    model = LinearRegression()
    model.fit(X_poly_train, y_train)

    subset_predictions_test = model.predict(X_poly_test)

    test_mse = mean_squared_error(y_test, subset_predictions_test)
    print(f"Mean Squared Error for {name} - Test: {test_mse}")

    subset_results = pd.DataFrame({
        'NAME': [name] * len(subset_predictions_test),
        'Predicted_AQI': subset_predictions_test,
        'Actual_AQI': y_test
    })
    predictions_df = pd.concat([predictions_df, subset_results], ignore_index=True)

print(predictions_df.head())



"""#                                                END"""