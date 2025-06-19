from sklearn.model_selection import train_test_split, GridSearchCV
from django.shortcuts import render
import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import time
import matplotlib.pyplot as plt
import io
import urllib, base64
import seaborn as sns
import matplotlib
from qualityindex.deep_learing import deep_learning
matplotlib.use('Agg')  # Use a non-interactive backend
def charts(request):
    return render(request, 'qualityindex/charts.html')
# Path to the dataset file (make sure the file exists)
DATASET_PATH =  'Dataset - Copy.csv'

def categorize_aqi(aqi_value):
    if 0 <= aqi_value <= 50:
        return "Good"
    elif 51 <= aqi_value <= 100:
        return "Moderate"
    elif 101 <= aqi_value <= 150:
        return "Unhealthy for Sensitive Groups"
    elif 151 <= aqi_value <= 200:
        return "Unhealthy"
    elif 201 <= aqi_value <= 300:
        return "Very Unhealthy"
    elif 301 <= aqi_value <= 500:
        return "Hazardous"
    else:
        return "Unknown"

def pollutants_charts(names,values,color,label,title,xlabel,ylabel,filename):
  chart_dir = os.path.join('static', 'charts')
  plt.figure(figsize=(20, 10))
  plt.bar(names, values, color=color, label=label)
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.xticks(rotation=90)
  plt.legend()
  plt.tight_layout()
  chart_path = os.path.join(chart_dir, filename +'.png')
  plt.savefig(chart_path)
  plt.close()
  
def generate_aqi_charts(results):
    chart_dir = os.path.join('static', 'charts')
    
    # Convert the results into a DataFrame
    df = pd.DataFrame(results)
    # Drop non-numeric columns that cannot be converted to float (e.g., 'name' and category columns)
    df_cleaned = df.drop(columns=['name', 'predicted_aqi_category', 'actual_aqi_category', 'input_aqi_category'])

    # Fill NaN values with 0 or another strategy if necessary
    df_cleaned = df_cleaned.fillna(0)
    names = []
    predicted_aqi_values = []
    actual_aqi_values = []
    pm25_value = []
    pm10_value = []
    no2_value = []
    so2_value = []
    co_value = []
    o3_value = []
    nox_value = []
    values = [(result.get('name', 'Unknown'),
               result.get('predicted_aqi', 0),
               result.get('actual_aqi', 0),
               result.get('PM2_MAX', 0),
               result.get('PM10_MAX', 0),
               result.get('NO2_MAX', 0),
               result.get('SO2_MAX', 0),
               result.get('CO_MAX', 0),
               result.get('OZONE_MAX', 0),
               result.get('NO_MAX', 0)
               
               ) for result in results]
    for name, predicted,actual,pm25, pm10,no2,so2,co,o3,nox in values:
      names.append(name)
      predicted_aqi_values.append(predicted)
      actual_aqi_values.append ( actual)
      pm25_value.append(pm25)
      pm10_value.append(pm10)
      no2_value.append(no2)
      so2_value.append(so2)
      co_value.append(co)
      o3_value.append(o3)
      nox_value.append(nox)
    # Create and save the Predicted AQI chart
    plt.figure(figsize=(20, 10))
    plt.plot(names, predicted_aqi_values, label='Predicted AQI', color='blue', marker='o')
    plt.xlabel('Location')
    plt.ylabel('AQI')
    plt.title('Predicted AQI by Location')
    plt.xticks(rotation=90)
    plt.tight_layout()
    chart_path = os.path.join(chart_dir, 'Predicted_AQI_by_Location.png')
    plt.savefig(chart_path)
    plt.close()

    # Create and save the Actual AQI chart
    plt.figure(figsize=(20, 10))
    plt.plot(names, actual_aqi_values, label='Actual AQI', color='green', marker='s')
    plt.xlabel('Location')
    plt.ylabel('AQI')
    plt.title('Actual AQI by Location')
    plt.xticks(rotation=90)
    plt.tight_layout()
    chart_path = os.path.join(chart_dir, 'Actual_AQI_by_Location.png')
    plt.savefig(chart_path)
    plt.close()
    
    
    
    pollutants_charts(names,pm25_value,'red','PM 2.5','PM2.5 by Location','Location','PM2_MAX','PM25_by_Location')
    pollutants_charts(names,pm10_value,'green','PM10','PM10 by Location','Location','PM10_MAX','PM10_by_Location')
    pollutants_charts(names,no2_value,'blue','NO2','NO2 by Location','Location','NO2_MAX','NO2_by_Location')
    pollutants_charts(names,so2_value,'purple','SO2','SO2 by Location','Location','SO2_MAX','SO2_by_Location')
    pollutants_charts(names,co_value,'brown','CO','CO by Location','Location','CO_MAX','CO_by_Location')
    pollutants_charts(names,o3_value,'orange','Ozone','Ozone by Location','Location','OZONE_MAX','Ozone_by_Location')
    pollutants_charts(names,nox_value,'olive','NOx','NOx by Location','Location','NOx (ug/m3)','NOx_by_Location')
   
    # Calculate the correlation matrix
    correlation_matrix = df_cleaned.corr()

    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title("Correlation Matrix Heatmap")

    # Save the plot as an image file in the static directory (if using Django)
    chart_path = os.path.join(chart_dir, 'correlation_heatmap.png')
    plt.savefig(chart_path)
    plt.close()
    # Colors for each category
    colors = {
        "Good": "Green",
        "Moderate": "Yellow",
        "Unhealthy for Sensitive Groups": "Orange",
        "Unhealthy": "Red",
        "Very Unhealthy": "Purple",
        "Hazardous": "Maroon",
        "Unknown": "magenta",
    }
    
    # AQI category distribution as a Pie Chart
    category_counts = df['actual_aqi_category'].value_counts()
    plt.figure(figsize=(8, 8))
    category_counts.plot(
        kind='pie',
        autopct='%1.1f%%',
        colors=[colors[cat] for cat in category_counts.index],  # Assign colors based on category
        startangle=90,
        legend=False
    )
    plt.title("Distribution of Actual AQI Categories", fontsize=16)
    plt.ylabel('')  # Remove y-label to make the chart cleaner
    plt.tight_layout()

    # Save the plot as an image file
    chart_path = os.path.join(chart_dir, 'AQI_Category.png')
    plt.savefig(chart_path)
    plt.close()
    
    
    # Histogram of PM2.5
    plt.figure(figsize=(10, 6))
    plt.hist(pm25_value, bins=20, color='red', edgecolor='black')
    plt.title('Distribution of PM2.5 Concentrations')
    plt.xlabel('PM2_MAX')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    chart_path = os.path.join(chart_dir, 'PM25_histogram.png')
    plt.savefig(chart_path)
    plt.close()

    # Box plot for NO2
    plt.figure(figsize=(10, 6))
    plt.boxplot(no2_value, patch_artist=True, notch=True, vert=False, boxprops=dict(facecolor='blue', color='black'))
    plt.title('NO2 Concentrations Box Plot')
    plt.xlabel('NO2_MAX')
    plt.tight_layout()
    chart_path = os.path.join(chart_dir, 'NO2_boxplot.png')
    plt.savefig(chart_path)
    plt.close()
    
    # Scatter plot of PM2.5 vs PM10
    plt.figure(figsize=(10, 6))
    plt.scatter(pm25_value, pm10_value, color='green', alpha=0.7)
    plt.title('PM2.5 vs PM10')
    plt.xlabel('PM2_MAX')
    plt.ylabel('PM10_MAX')
    plt.grid(True)
    plt.tight_layout()
    chart_path = os.path.join(chart_dir, 'PM25_vs_PM10_scatter.png')
    plt.savefig(chart_path)
    plt.close()
    
    # Stacked Bar Chart for PM2.5, PM10, NO2
    df_pollutants = pd.DataFrame({
        'PM2.5': pm25_value,
        'PM10': pm10_value,
        'NO2': no2_value
    })

    df_pollutants.plot(kind='bar', stacked=True, figsize=(20, 10), color=['red', 'green', 'blue'])
    plt.title('Stacked Pollutants by Location')
    plt.xlabel('Location')
    plt.ylabel('Concentration (ug/m3)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    chart_path = os.path.join(chart_dir, 'Stacked_Pollutants_by_Location.png')
    plt.savefig(chart_path)
    plt.close()
    
    # Area plot for PM2.5, PM10, and NO2
    df_area = pd.DataFrame({
        'PM2.5': pm25_value,
        'PM10': pm10_value,
        'NO2': no2_value
    })
    df_area.plot(kind='area', figsize=(20, 10), alpha=0.6, color=['red', 'green', 'blue'])
    plt.title('Area Plot of Pollutants by Location')
    plt.xlabel('Location')
    plt.ylabel('Concentration (ug/m3)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    chart_path = os.path.join(chart_dir, 'Area_Plot_of_Pollutants.png')
    plt.savefig(chart_path)
    plt.close()
    
    
    # Radar Chart for Pollutants at One Location
    pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone', 'NOx']
    values = [pm25_value[0], pm10_value[0], no2_value[0], so2_value[0], co_value[0], o3_value[0], nox_value[0]]

    angles = np.linspace(0, 2 * np.pi, len(pollutants), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='orange', alpha=0.25)
    ax.plot(angles, values, color='orange', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(pollutants, fontsize=12)
    plt.title('Radar Chart for Pollutants at Location 1', fontsize=16)
    plt.tight_layout()
    chart_path = os.path.join(chart_dir, 'Radar_Chart_Pollutants.png')
    plt.savefig(chart_path)
    plt.close()
# def generate_aqi_charts(results, input_aqi_values):
#     names = [result.get('name', 'Unknown') for result in results]
#     predicted_aqi_values = [result.get('predicted_aqi', 0) for result in results]
#     actual_aqi_values = [result.get('actual_aqi', 0) for result in results]
#     input_aqi_numeric = [input_aqi_values.get(name, 0) for name in names]

#     # Create and save the Predicted AQI chart
#     plt.figure(figsize=(10, 6))
#     plt.plot(names, predicted_aqi_values, label='Predicted AQI', color='blue', marker='o')
#     plt.xlabel('Location')
#     plt.ylabel('AQI')
#     plt.title('Predicted AQI by Location')
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#     buf_predicted = io.BytesIO()
#     plt.savefig(buf_predicted, format='png')
#     buf_predicted.seek(0)
#     predicted_string = base64.b64encode(buf_predicted.read())
#     predicted_uri = 'data:image/png;base64,' + urllib.parse.quote(predicted_string)
#     plt.close()

#     # Create and save the Actual AQI chart
#     plt.figure(figsize=(10, 6))
#     plt.plot(names, actual_aqi_values, label='Actual AQI', color='green', marker='s')
#     plt.xlabel('Location')
#     plt.ylabel('AQI')
#     plt.title('Actual AQI by Location')
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#     buf_actual = io.BytesIO()
#     plt.savefig(buf_actual, format='png')
#     buf_actual.seek(0)
#     actual_string = base64.b64encode(buf_actual.read())
#     actual_uri = 'data:image/png;base64,' + urllib.parse.quote(actual_string)
#     plt.close()

#     # Create and save the Input AQI chart
#     plt.figure(figsize=(10, 6))
#     plt.plot(names, input_aqi_numeric, label='Input AQI', color='orange', marker='D', linestyle='--')
#     plt.xlabel('Location')
#     plt.ylabel('AQI')
#     plt.title('Input AQI by Location')
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#     buf_input = io.BytesIO()
#     plt.savefig(buf_input, format='png')
#     buf_input.seek(0)
#     input_string = base64.b64encode(buf_input.read())
#     input_uri = 'data:image/png;base64,' + urllib.parse.quote(input_string)
#     plt.close()

#     # Plot Temperature chart (if available)
#     temperature_values = [result.get('temperature', 0) for result in results]
#     plt.figure(figsize=(10, 5))
#     plt.bar(names, temperature_values, color='green', label='Temperature')
#     plt.title('Temperature by Location')
#     plt.xlabel('Location')
#     plt.ylabel('Temperature (°C)')
#     plt.xticks(rotation=45)
#     plt.legend()
#     plt.tight_layout()
#     buf_input = io.BytesIO()
#     plt.savefig(buf_input, format='png')
#     buf_input.seek(0)
#     input_string = base64.b64encode(buf_input.read())
#     temp_uri = 'data:image/png;base64,' + urllib.parse.quote(input_string)
#     plt.close()

#     # Plot Humidity chart (if available)
#     humidity_values = [result.get('humidity', 0) for result in results]
#     plt.figure(figsize=(10, 5))
#     plt.bar(names, humidity_values, color='purple', label='Humidity')
#     plt.title('Humidity by Location')
#     plt.xlabel('Location')
#     plt.ylabel('Humidity (%)')
#     plt.xticks(rotation=45)
#     plt.legend()
#     plt.tight_layout()
#     buf_input = io.BytesIO()
#     plt.savefig(buf_input, format='png')
#     buf_input.seek(0)
#     input_string = base64.b64encode(buf_input.read())
#     hum_uri = 'data:image/png;base64,' + urllib.parse.quote(input_string)
#     plt.close()

#     return predicted_uri, actual_uri, input_uri, temp_uri, hum_uri
def calculate_user_aqi(concentration, breakpoints):
    """
    Calculate AQI for a single pollutant.
    :param concentration: Observed concentration of the pollutant
    :param breakpoints: List of tuples containing (C_low, C_high, I_low, I_high)
    :return: AQI value
    """
    for C_low, C_high, I_low, I_high in breakpoints:
        if C_low <= concentration <= C_high:
            aqi = ((I_high - I_low) / (C_high - C_low)) * (concentration - C_low) + I_low
            return round(aqi)
    return 0  # If concentration is out of range
def calculate_overall_aqi(pm10, pm25, so2, nox, co, o3):
    # Define breakpoints for each pollutant (example breakpoints for each pollutant in relevant units)
    
    # PM10 breakpoints (µg/m³)
    pm10_breakpoints = [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200), (251, 350, 201, 300), (351, 500, 301, 400)]
    
    # PM2.5 breakpoints (µg/m³)
    pm25_breakpoints = [(0, 12, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150), (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300)]
    
    # SO2 breakpoints (ppb)
    so2_breakpoints = [(0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150), (186, 304, 151, 200), (305, 604, 201, 300)]
    
    # NOx breakpoints (ppb)
    nox_breakpoints = [(0, 50, 0, 50), (51, 100, 51, 100), (101, 200, 101, 150), (201, 300, 151, 200)]
    
    # CO breakpoints (ppm)
    co_breakpoints = [(0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150), (12.5, 15.4, 151, 200)]
    
    # O3 breakpoints (ppb)
    o3_breakpoints = [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150), (255, 354, 151, 200), (355, 424, 201, 300)]
    
    
    # Calculate AQI for each pollutant
    aqi_pm10 = calculate_user_aqi(pm10, pm10_breakpoints)
    aqi_pm25 = calculate_user_aqi(pm25, pm25_breakpoints)
    aqi_so2 = calculate_user_aqi(so2, so2_breakpoints)
    aqi_nox = calculate_user_aqi(nox, nox_breakpoints)
    aqi_co = calculate_user_aqi(co, co_breakpoints)
    aqi_o3 = calculate_user_aqi(o3, o3_breakpoints)
    # Find the overall AQI by taking the maximum of the individual AQI values
    try:
        overall_aqi = max(filter(None, [aqi_pm10, aqi_pm25, aqi_so2, aqi_nox, aqi_co, aqi_o3]))
    except:
        overall_aqi = 0
    return overall_aqi

# def calculate_aqi(data):
#     """
#     Calculate AQI based on available data.
#     This is a simplified formula for demonstration purposes.
#     You can replace this with an actual AQI formula or calculation.
#     """
#     # Example: Combine different factors (just for demonstration)
#     aqi = (data['NO2_MAX'] + data['OZONE_MAX'] + data['PM10_MAX'] + data['CO2_MAX']) / 4
#     return aqi

def calculate_aqi(data):
    """
    Calculate AQI based on available data.
    This is a simplified formula for demonstration purposes.
    You can replace this with an actual AQI formula or calculation.
    """
    # Example: Combine different factors (just for demonstration)
    aqi = calculate_overall_aqi(data['PM10_MAX'], data['PM2_MAX'], data['SO2_MAX'], data['NO2_MAX'], data['CO2_MAX'], data['OZONE_MAX'])
    return aqi
def get_float_value(request, field_name):
    value = request.POST.get(field_name, '')
    return float(value) if value else 0.0
def predict_aqi(request):
    if request.method == 'POST':
        # Get selected date from the form
        selected_date = request.POST['date']
        names = request.POST['names']

         # Get the values from the form
        pm10 = get_float_value(request, 'pm10')
        pm25 = get_float_value(request, 'pm25')
        so2 = get_float_value(request, 'so2')
        nox = get_float_value(request, 'nox')
        co = get_float_value(request, 'co')
        o3 = get_float_value(request, 'o3')
        nh3 = get_float_value(request, 'nh3')

        # # Calculate AQI for each pollutant
        # aqi_pm10 = calculate_indian_aqi(pm10, 'PM10')
        # aqi_pm25 = calculate_indian_aqi(pm25, 'PM2.5')
        # aqi_so2 = calculate_indian_aqi(so2, 'SO2')
        # aqi_nox = calculate_indian_aqi(nox, 'NO2')
        # aqi_co = calculate_indian_aqi(co, 'CO')
        # aqi_o3 = calculate_indian_aqi(o3, 'O3')
        # aqi_nh3 = calculate_indian_aqi(nh3, 'NH3')

        # Find the highest AQI value (which will be the overall AQI)
        aqi = pm10+ pm25+ so2+ nox+ co+ o3+ nh3
        aqi = aqi / 7
        # Determine the status based on the AQI value
        status = categorize_aqi(aqi)
        
        # Convert the selected date to datetime format for comparison
        selected_date = pd.to_datetime(selected_date)
        
        # Load the dataset from the root directory
        data = pd.read_csv(DATASET_PATH)

        if names != 'All':
            data = data[data['NAME'] == names]
        deep_rmse_total, deep_r2_avg, deep_mae_avg, deep_time = deep_learning(data)

        # Ensure the 'LASTUPDATEDATETIME' column exists and convert it to datetime
        if 'LASTUPDATEDATETIME' not in data.columns:
            return render(request, 'qualityindex/index.html', {'error': 'LASTUPDATEDATETIME column is missing from dataset.'})
        
        data['LASTUPDATEDATETIME'] = pd.to_datetime(data['LASTUPDATEDATETIME'])

        # Calculate AQI for each row based on available columns (historical data)
        data['AQI'] = data.apply(calculate_aqi, axis=1)

        # Handle NaN values in AQI column (you can drop rows or impute)
        data.dropna(subset=['AQI'], inplace=True)

        # Prepare the features (historical data) for training
        features = data[['NO2_MAX', 'OZONE_MAX', 'PM10_MAX', 'CO2_MAX', 'AIR_PRESSURE', 'HUMIDITY', 'TEMPRATURE_MAX']]
        target = data['AQI']

        # Handle missing values in features using imputation
        imputer = SimpleImputer(strategy='mean')
        features_imputed = imputer.fit_transform(features)

        # Perform PCA for dimensionality reduction
        pca = PCA(n_components=5)
        X_pca = pca.fit_transform(features_imputed)

        # Train a Linear Regression model
        model = LinearRegression()
        model.fit(X_pca, target)

        # Start timer for execution time
        start_time = time.time()

        # For future date prediction, we will use the average of historical data
        average_features = {
            'NO2_MAX': data['NO2_MAX'].mean(),
            'OZONE_MAX': data['OZONE_MAX'].mean(),
            'PM10_MAX': data['PM10_MAX'].mean(),
            'CO2_MAX': data['CO2_MAX'].mean(),
            'AIR_PRESSURE': data['AIR_PRESSURE'].mean(),
            'HUMIDITY': data['HUMIDITY'].mean(),
            'TEMPRATURE_MAX': data['TEMPRATURE_MAX'].mean()
        }

        # Convert the averages to a 2D array for prediction
        future_features = np.array([[average_features['NO2_MAX'], 
                                     average_features['OZONE_MAX'], 
                                     average_features['PM10_MAX'], 
                                     average_features['CO2_MAX'], 
                                     average_features['AIR_PRESSURE'], 
                                     average_features['HUMIDITY'], 
                                     average_features['TEMPRATURE_MAX']]])

        # Impute missing values and apply PCA
        future_features_imputed = imputer.transform(future_features)
        future_features_pca = pca.transform(future_features_imputed)

        # Predict future AQI (for the selected date)
        future_predicted_aqi = model.predict(future_features_pca)

        # Collect results for all locations (using historical data)
        results = []
        predicted_qa_values = []  # For storing predicted AQIs
        actual_qa_values = []    # For storing actual AQIs
        unique_names = data['NAME'].unique()

        for name in unique_names:
            name_data = data[data['NAME'] == name]
            actual_aqi = name_data['AQI'].mean()

            # Get features for prediction
            name_features = name_data[['NO2_MAX', 'OZONE_MAX', 'PM10_MAX', 'CO2_MAX', 'AIR_PRESSURE', 'HUMIDITY', 'TEMPRATURE_MAX']]
            name_features_imputed = imputer.transform(name_features)
            name_features_pca = pca.transform(name_features_imputed)

            predicted_aqi = model.predict(name_features_pca)

            # Append results including input AQI
            results.append({
                'name': name,
                'predicted_aqi': predicted_aqi.mean(),
                'predicted_aqi_category': categorize_aqi(predicted_aqi.mean()),
                'actual_aqi': actual_aqi,
                'actual_aqi_category': categorize_aqi(actual_aqi),
                'input_aqi': actual_aqi,  # Use the actual AQI as input AQI for now
                'input_aqi_category': categorize_aqi(actual_aqi),
                'temperature': name_data['TEMPRATURE_MAX'].mean(),
                'humidity': name_data['HUMIDITY'].mean(),
                'PM2_MAX': name_data['PM2_MAX'].mean(),
                'PM10_MAX': name_data['PM10_MAX'].mean(),
                'NO2_MAX': name_data['NO2_MAX'].mean(),
                'SO2_MAX': name_data['SO2_MAX'].mean(),
                'CO_MAX': name_data['CO_MAX'].mean(),
                'OZONE_MAX': name_data['OZONE_MAX'].mean(),
                'NO_MAX': name_data['NO_MAX'].mean()
            })

            # Save the predicted and actual AQIs for model performance calculation
            predicted_qa_values.append(predicted_aqi.mean())
            actual_qa_values.append(actual_aqi)

        # Check if there are enough data points after filtering by location
        if len(actual_qa_values) > 0 and len(predicted_qa_values) > 0:
            r2 = r2_score(actual_qa_values, predicted_qa_values)
        else:
            # Return NaN or a suitable message if the dataset is empty
            r2 = float('nan')
        mae = mean_absolute_error(actual_qa_values, predicted_qa_values)
        rmse = np.sqrt(mean_squared_error(actual_qa_values, predicted_qa_values))

        # # Add Future Prediction (for the selected date)
        # results.append({
        #     'name': 'Future Prediction',
        #     'predicted_aqi': future_predicted_aqi.mean(),
        #     'predicted_aqi_category': categorize_aqi(future_predicted_aqi.mean()),
        #     'actual_aqi': np.nan,  # No actual value for future prediction
        #     'actual_aqi_category': 'Unknown',
        #     'input_aqi': np.nan,
        #     'input_aqi_category': 'Unknown',
        #     'temperature': average_features['TEMPRATURE_MAX'],  # Use the average temperature
        #     'humidity': average_features['HUMIDITY']            # Use the average humidity
        # })

        # Stop timer for execution time
        execution_time = time.time() - start_time

        # Generate charts (pass the correct results and input_aqi_values)
        results = sorted(results, key=lambda x: x['name'])
        generate_aqi_charts(results)
        # Function to add suffix to the day (e.g., 1st, 2nd, 3rd, 4th)
        def get_day_with_suffix(day):
            if 10 <= day <= 20:  # Handle special case for 11th, 12th, 13th, etc.
                suffix = 'th'
            else:
                suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
            return f"{day}{suffix}"
        
        day_with_suffix = get_day_with_suffix(selected_date.day)
        # Format the full date with day, month name, and year
        formatted_date = f"{day_with_suffix} {selected_date.strftime('%B %Y')}"
        # Return results along with metrics
        statistics_per_location = data[['PM2_MAX', 'PM10_MAX', 'NO2_MAX', 
                          'SO2_MAX', 'CO_MAX', 'OZONE_MAX', 
                          'NO_MAX']].describe().transpose()
         # Strip any extra spaces from column names
        statistics_per_location.index = statistics_per_location.index.str.strip()

        # Remove any text inside parentheses, including units like (ug/m3) and (ppb)
        statistics_per_location.index = statistics_per_location.index.str.replace(r' \(.+\)', '', regex=True)

        # Drop the 'count' column
        statistics_per_location = statistics_per_location.drop(columns='count')

        # Rename the percentiles in the columns
        statistics_per_location = statistics_per_location.rename(columns={'25%': 'Q1', '50%': 'Q2', '75%': 'Q3'})
        statistics_html = statistics_per_location.to_html(classes='table table-bordered table-striped')
        # Extract relevant features and target
        target = 'CO2_MIN'
        features = ['PM10_MAX', 'PM2_MAX', 'NO2_MAX', 'CO_MAX', 'OZONE_MAX']

        split_data = {}
        for name in data['NAME'].unique():
            subset_data = data[data['NAME'] == name]
            X = subset_data[features]
            y = subset_data[target]

            X = X.fillna(X.mean())
            y = y.fillna(y.mean())

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            split_data[name] = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }

        # regression_types = [
        #     "linear", "ridge", "lasso", "elasticnet", "sgd", "bayesian", 
        #     "decision_tree", "random_forest", "gradient_boosting", "xgboost", 
        #     "svr", "knn", "pls", "pcr"
        # ] 
        
        regression_types = [
            "linear", "decision_tree", "random_forest", "pcr"
        ]

        # Function to compute metrics (MSE, R², MAE, RMSE)
        def compute_metrics(model, X_train, X_test, y_train, y_test):
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)  # Mean Absolute Error
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            
            return mse, r2, mae, rmse

        # Function to perform regression
        def perform_regression(split_data, regression_type="linear"):
            sttime = time
            mse_list = []
            r2_list = []
            mae_list = []
            rmse_list = []
            execution_times = []

            for name in split_data:
                X_train = split_data[name]['X_train']
                X_test = split_data[name]['X_test']
                y_train = split_data[name]['y_train']
                y_test = split_data[name]['y_test']

                polynomial_features = PolynomialFeatures(degree=2)
                X_poly_train = polynomial_features.fit_transform(X_train)
                X_poly_test = polynomial_features.fit_transform(X_test)

                if regression_type == "linear":
                    model = LinearRegression()
                elif regression_type == "ridge":
                    model = Ridge(solver='svd')
                elif regression_type == "lasso":
                    model = Lasso(alpha=0.1)
                elif regression_type == "elasticnet":
                    model = ElasticNet(alpha=0.1)
                elif regression_type == "bayesian":
                    model = BayesianRidge()
                elif regression_type == "decision_tree":
                    model = DecisionTreeRegressor()
                elif regression_type == "random_forest":
                    model = RandomForestRegressor(n_estimators=100)
                elif regression_type == "gradient_boosting":
                    model = GradientBoostingRegressor(n_estimators=100)
                elif regression_type == "xgboost":
                    model = xgb.XGBRegressor(objective='reg:squarederror')
                elif regression_type == "svr":
                    model = SVR(kernel='rbf')
                elif regression_type == "knn":
                    model = KNeighborsRegressor(n_neighbors=5)
                elif regression_type == "pls":
                    model = PLSRegression(n_components=3)
                elif regression_type == "pcr":
                    pca = PCA(n_components=3)
                    X_train_pca = pca.fit_transform(X_poly_train)
                    X_test_pca = pca.transform(X_poly_test)
                    model = LinearRegression()
                    mse, r2, mae, rmse = compute_metrics(model, X_train_pca, X_test_pca, y_train, y_test)
                    mse_list.append(mse)
                    r2_list.append(r2)
                    mae_list.append(mae)
                    rmse_list.append(rmse)
                    continue
                else:
                    return None  # Return None if the regression type is unsupported

                mse, r2, mae, rmse = compute_metrics(model, X_poly_train, X_poly_test, y_train, y_test)
                mse_list.append(mse)
                r2_list.append(r2)
                mae_list.append(mae)
                rmse_list.append(rmse)

            if mse_list and r2_list:
                total_test_mse = sum(mse_list)
                avg_r2_score = sum(r2_list) / len(r2_list)
                avg_mae = sum(mae_list) / len(mae_list)
                avg_rmse = sum(rmse_list) / len(rmse_list)
                return total_test_mse, avg_r2_score, avg_mae, avg_rmse
            else:
                return 0, 0, 0, 0  # Return default values if no metrics are computed

        # List to store the results
        otheralog = []

        # Perform regression for all types
        for reg_type in regression_types:
            st_time = time.time()
            result = perform_regression(split_data, regression_type=reg_type)
            end_time = time.time()
            execution_time = end_time - st_time
            formatted_time = "{:.2f}".format(execution_time)
            
            if result:
                total_test_mse, avg_r2_score, avg_mae, avg_rmse = result
                otheralog.append([reg_type, f'{avg_r2_score:.4f}', f'{avg_mae:.4f}', f'{avg_rmse:.4f}', formatted_time])
            else:
                otheralog.append([reg_type, 'Error', 'Error', 'Error', formatted_time])
        
        # Start timing
        st_time = time.time()

        # Define PCR with Decision Tree
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.99)),  # Retaining 95% variance
            ("regressor", DecisionTreeRegressor(max_depth=5))  # Limit depth to avoid overfitting
        ])

        # Train the model
        pipeline.fit(X_train, y_train)

        # Predictions
        y_pred = pipeline.predict(X_test)

        # Evaluate performance
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # End timing
        execution_time = time.time() - st_time
        formatted_time = "{:.2f}".format(execution_time)

        # Append results
        otheralog.append(['PCR With Decision Tree', f'{r2:.4f}', f'{mae:.4f}', f'{rmse:.4f}', formatted_time])
        
        # Start timing
        st_time = time.time()

        # Define PCR with Ridge Regression
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95)),  # Retaining 95% variance
            ("regressor", Ridge(alpha=1.0))  # Ridge regression with regularization
        ])

        # Train the model
        pipeline.fit(X_train, y_train)

        # Predictions
        y_pred = pipeline.predict(X_test)

        # Evaluate performance
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # End timing
        execution_time = time.time() - st_time
        formatted_time = "{:.2f}".format(execution_time)

        # Append results
        otheralog.append(['PCR With Ridge Regression', f'{r2:.4f}', f'{mae:.4f}', f'{rmse:.4f}', formatted_time])

        # Start timing
        st_time = time.time()

        # Define PCR with Decision Tree
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.99)),  # Retaining 99% variance
            ("regressor", GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42))  # Using Gradient Boosting
        ])

        # Define parameter grid for GridSearchCV
        param_grid = {
            'pca__n_components': [0.95, 0.99],
            'regressor__n_estimators': [100, 200],
            'regressor__learning_rate': [0.01, 0.1],
            'regressor__max_depth': [3, 5, 7]
        }

        # Use GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Best model
        best_model = grid_search.best_estimator_

        # Predictions
        y_pred = best_model.predict(X_test)

        # Evaluate performance
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # End timing
        execution_time = time.time() - st_time
        formatted_time = "{:.2f}".format(execution_time)

        # Append results
        otheralog.append(['PCR With Gradient Boosting', f'{r2:.4f}', f'{mae:.4f}', f'{rmse:.4f}', formatted_time])
        from sklearn.preprocessing import LabelEncoder
        from sklearn.pipeline import make_pipeline
        # Assume 'data' is already loaded

        # Drop unnecessary columns
        data = data.drop(columns=['LASTUPDATEDATETIME'])  # Date-time not needed

        # Handle categorical column 'NAME'
        if 'NAME' in data.columns:
            data['NAME'] = LabelEncoder().fit_transform(data['NAME'])  # Convert station names to numbers
        # Handle missing values (Imputation)
        imputer = SimpleImputer(strategy='mean')  # Fill NaN with column mean
        data[:] = imputer.fit_transform(data)  # Apply imputation
        # Define Features (X) and Target (y)
        X = data.drop(columns=['AQI'])  # Features
        y = data['AQI']  # Target variable

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # List of regression models
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(n_estimators=100),
            "Gradient Boosting": GradientBoostingRegressor(),
            "Support Vector Regression": SVR(),
            "PCR (Principal Component Regression)": make_pipeline(StandardScaler(), PCA(n_components=5), LinearRegression()),
            "PCR with Gradient Boosting": make_pipeline(StandardScaler(), PCA(n_components=5), GradientBoostingRegressor()),
            "PCR with Ridge Regression": make_pipeline(StandardScaler(), PCA(n_components=5), Ridge()),
            "PCR with Decision Tree": make_pipeline(StandardScaler(), PCA(n_components=5), DecisionTreeRegressor()),
        }
        from sklearn.metrics import mean_absolute_percentage_error, median_absolute_error, explained_variance_score
        # Function to compute regression metrics
        def regression_metrics(y_true, y_pred, n, p, execution_time):
            mse = mean_squared_error(y_true, y_pred)
            mpe = np.mean((y_true - y_pred) / y_true) * 100
            mape = mean_absolute_percentage_error(y_true, y_pred) * 100
            medae = median_absolute_error(y_true, y_pred)
            explained_var = explained_variance_score(y_true, y_pred)
            adjusted_r2 = 1 - ((1 - explained_var) * (n - 1) / (n - p - 1))

            return [mse, mpe, mape, medae, explained_var, adjusted_r2, execution_time]

        # Store results
        results_new = []

        # Train and evaluate each model
        
        # for name, model in models.items():
        #     start_time = time.time()  # Start time tracking
        #     model.fit(X_train, y_train)
        #     y_pred = model.predict(X_test)
        #     end_time = time.time()  # End time tracking
        #     execution_time = end_time - start_time  # Calculate execution time

        #     metrics = regression_metrics(y_test, y_pred, len(y_test), X_train.shape[1], execution_time)
        #     results_new.append([name] + metrics)

        # Convert results to DataFrame
        columns = ["Regression Type", "MSE", "MPE", "MAPE", "MedAE", "Explained Variance", "Adjusted R²", "Execution Time (s)"]
        results_df = pd.DataFrame(results_new, columns=columns)
        results_df = results_df.values.tolist()
        # Display results
        return render(request, 'qualityindex/result.html', {
            'results': results,
            'results_df': results_df,
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'execution_time': execution_time,
            'selected_date': formatted_date,
            'aqi_value': aqi, 'aqi_category': status,
            'statistics_html':statistics_html,
            'statistics_per_location':statistics_per_location,
            'otheralog':otheralog,
            'deep_mae_avg':deep_mae_avg,
            'deep_r2_avg':deep_r2_avg,
            'deep_rmse_total':deep_rmse_total,
            'deep_time':deep_time
        })


    return render(request, 'qualityindex/index.html')

def user_result_view(request):
    # Get the query parameters from the URL
    aqi_value = request.GET.get('aqi_value')
    aqi_category = request.GET.get('aqi_category')

    # Render the user_result.html with the received values
    return render(request, 'qualityindex/user_result.html', {
        'aqi_value': aqi_value,
        'aqi_category': aqi_category
    })