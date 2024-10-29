# Full Backend for the Risk Radar Application

This repository contains the complete backend for the sample Risk Radar application. It includes files for data generation, database management, machine learning model training, and a Flask web server to serve the application

## How to Run the Application

### Step 1: Install Dependencies

Make sure you have Python installed. Then, install the required libraries by running:

```
pip install -r requirements.txt
```

### Step 2: Configure Database Credentials

Update your database credentials in the database.py file.

**Important Note:** _For security reasons, avoid hardcoding credentials in production. Use a `.env` file and load the credentials dynamically in database.py._

### Step 3: (Optional) Train the Model

If you want to re-train the machine learning model:

Open and run model.ipynb and run all cells to train the model.
This will generate the model artifacts required by the application.

### Step 4: Generate and Load Sample Data int DB

Generate sample data by running the following script:

```
python setup/data_sample_generator.py
```

Load the generated sample data into your database using:

```
python setup/data_loader.py
```

### Step 5: Verify Backend Logic

Test the backend logic by running:

```
python service.py
```

### Step 6: Run the Flask Application

Start the Flask web server with:

```
python app.py
```

The application should now be running locally. Access it via: http://localhost:5000
