import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import pandas as pd
from flask import Flask, request, jsonify
import os
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
ALLOWED_EXTENSIONS = {'csv'} 

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        return jsonify({'success': True, 'message': 'File uploaded successfully'})

    return jsonify({'error': 'File type not allowed'})

@app.route('/process', methods=['GET'])
def process_data():
    # Assuming the file path is defined globally
    file_path = "./static/uploads/OnlineRetail.csv"

    # Read the CSV file
    df_data = pd.read_csv(file_path, encoding='latin1')

    # Check for NaN values
    if df_data.isnull().sum().any():
        # Drop rows with NaN values
        df_data.dropna(subset=['Quantity', 'UnitPrice'], inplace=True)
        
        if df_data.empty:
            return jsonify({"error": "Dataset is empty after dropping NaN values. Please check your data"})

        # Example: Calculate Recency, Frequency, Monetary
        today_date = pd.to_datetime('2011-12-11')
        df_data['InvoiceDate'] = pd.to_datetime(df_data['InvoiceDate'])
        df_data['TotalPrice'] = df_data['Quantity'] * df_data['UnitPrice']
        df_data['Recency'] = (today_date - df_data['InvoiceDate']).dt.days
        rfm = df_data.groupby('CustomerID').agg({'Recency': 'min', 'InvoiceNo': 'nunique', 'TotalPrice': 'sum'})

        # Handle NaN values using SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        rfm_imputed = imputer.fit_transform(rfm[['Recency', 'TotalPrice']])

        # Example: Apply K-Means clustering
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm_imputed)

        kmean_model = KMeans(n_clusters=4, init='k-means++', max_iter=1000, random_state=20)
        kmean_model.fit(rfm_scaled)
        rfm['Cluster'] = kmean_model.labels_

        # Get the number of clusters
        num_clusters = len(set(kmean_model.labels_))

        # Prepare result as JSON
        result = {
            "rfm_statistics": rfm.describe().to_dict(),
            "num_clusters": num_clusters
        }

        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
