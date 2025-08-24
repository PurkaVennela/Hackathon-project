# Hackathon Project - Multivariate Time Series Anomaly Detection

This project detects anomalies in **multivariate time series sensor data** using the **Isolation Forest algorithm**.  
It preprocesses the dataset, trains on a normal time window, and then flags unusual patterns with anomaly scores and top contributing features.

---

## 📂 Files
- **main.py** → Python script for preprocessing, training, anomaly detection, and visualization  
- **TEP_Train_Test_Output.csv** → Final dataset with anomaly scores + top 7 contributing features  
- **requirements.txt** → Python libraries required to run the project  

---

## ⚙️ How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/Hackathon-project.git
   cd Hackathon-project
   ```   
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```   
3. **Run the code**
   ```bash
   python main.py
   ```   
The script will generate:

✅ A processed CSV (TEP_Train_Test_Output.csv) with anomalies

📊 Visualizations (scatter plot of anomalies over time + bar chart of top contributing features)

