{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "cdb5f871-9abf-4990-8c05-66bbc7bea224",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: pandas in /opt/anaconda3/envs/diploia/lib/python3.12/site-packages (2.2.3)\n",
      "Collecting openpyxl\n",
      "  Downloading openpyxl-3.1.5-py2.py3-none-any.whl.metadata (2.5 kB)\n",
      "Requirement already satisfied: numpy>=1.26.0 in /opt/anaconda3/envs/diploia/lib/python3.12/site-packages (from pandas) (2.2.0)\n",
      "Requirement already satisfied: python-dateutil>=2.8.2 in /opt/anaconda3/envs/diploia/lib/python3.12/site-packages (from pandas) (2.9.0.post0)\n",
      "Requirement already satisfied: pytz>=2020.1 in /opt/anaconda3/envs/diploia/lib/python3.12/site-packages (from pandas) (2024.1)\n",
      "Requirement already satisfied: tzdata>=2022.7 in /opt/anaconda3/envs/diploia/lib/python3.12/site-packages (from pandas) (2024.2)\n",
      "Collecting et-xmlfile (from openpyxl)\n",
      "  Downloading et_xmlfile-2.0.0-py3-none-any.whl.metadata (2.7 kB)\n",
      "Requirement already satisfied: six>=1.5 in /opt/anaconda3/envs/diploia/lib/python3.12/site-packages (from python-dateutil>=2.8.2->pandas) (1.16.0)\n",
      "Downloading openpyxl-3.1.5-py2.py3-none-any.whl (250 kB)\n",
      "Downloading et_xmlfile-2.0.0-py3-none-any.whl (18 kB)\n",
      "Installing collected packages: et-xmlfile, openpyxl\n",
      "Successfully installed et-xmlfile-2.0.0 openpyxl-3.1.5\n"
     ]
    }
   ],
   "source": [
    "!pip install pandas openpyxl"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "7e6e6fd7-dec9-4016-9acb-7e3286d3d47c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Precisión: 0.9648\n",
      "AUC-ROC: 0.9938\n",
      "\n",
      "Reporte de Clasificación:\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.67      1.00      0.80        30\n",
      "           1       1.00      0.96      0.98       396\n",
      "\n",
      "    accuracy                           0.96       426\n",
      "   macro avg       0.83      0.98      0.89       426\n",
      "weighted avg       0.98      0.96      0.97       426\n",
      "\n"
     ]
    }
   ],
   "source": [
    "import joblib\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.metrics import accuracy_score, roc_auc_score, classification_report\n",
    "\n",
    "# Cargar los datos\n",
    "file_path = \"datos_proyecto_ML.xlsx\"\n",
    "try:\n",
    "    df = pd.read_excel(file_path, sheet_name='Hoja1')\n",
    "except FileNotFoundError:\n",
    "    print(f\"Error: No se encontró el archivo {file_path}. Asegúrate de que esté en la misma carpeta que este notebook.\")\n",
    "    exit()\n",
    "\n",
    "# Verificar nombres de columnas y eliminar espacios\n",
    "df.columns = df.columns.str.strip()\n",
    "\n",
    "# Convertir columnas a numérico asegurando que los valores no sean texto\n",
    "cols_to_convert = [\"EDAD\", \"PESO (Kg)\", \"TALLA (mt)\", \"IMC\", \"DG NUTRICIONAL\", \"PAS\", \"PAD\", \"GLICEMIA\", \"COLESTEROL\", \"CC\"]\n",
    "for col in cols_to_convert:\n",
    "    df[col] = pd.to_numeric(df[col], errors='coerce')\n",
    "\n",
    "# Imputar valores faltantes con la mediana\n",
    "imputer = SimpleImputer(strategy='median')\n",
    "df[cols_to_convert] = imputer.fit_transform(df[cols_to_convert])\n",
    "\n",
    "# Crear la variable objetivo \"Riesgo_CV\" si no existe\n",
    "df[\"Riesgo_CV\"] = (\n",
    "    (df[\"PAS\"] >= 140) |  # Hipertensión sistólica\n",
    "    (df[\"PAD\"] >= 90) |   # Hipertensión diastólica\n",
    "    (df[\"COLESTEROL\"] >= 200) |  # Colesterol alto\n",
    "    (df[\"GLICEMIA\"] >= 126) |  # Diabetes\n",
    "    (df[\"IMC\"] >= 25)  # Sobrepeso u obesidad\n",
    ").astype(int)\n",
    "\n",
    "# Seleccionar las variables relevantes\n",
    "features = [\"EDAD\", \"PESO (Kg)\", \"TALLA (mt)\", \"IMC\", \"DG NUTRICIONAL\", \"PAS\", \"PAD\", \"GLICEMIA\", \"COLESTEROL\", \"CC\"]\n",
    "target = \"Riesgo_CV\"\n",
    "\n",
    "# Asegurar que hay suficientes datos después de la limpieza\n",
    "if df.shape[0] == 0:\n",
    "    print(\"Error: No quedan datos suficientes después de la limpieza. Revisa el archivo Excel y asegúrate de que contiene datos válidos.\")\n",
    "    exit()\n",
    "\n",
    "X = df[features]\n",
    "y = df[target]\n",
    "\n",
    "# Dividir en entrenamiento y prueba\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n",
    "\n",
    "# Escalar los datos\n",
    "scaler = StandardScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "X_test_scaled = scaler.transform(X_test)\n",
    "\n",
    "# Entrenar modelo de Regresión Logística\n",
    "log_reg = LogisticRegression(class_weight=\"balanced\", max_iter=2000, random_state=42)\n",
    "log_reg.fit(X_train_scaled, y_train)\n",
    "\n",
    "# Predicciones\n",
    "y_pred = log_reg.predict(X_test_scaled)\n",
    "y_prob = log_reg.predict_proba(X_test_scaled)[:, 1]\n",
    "\n",
    "# Evaluación\n",
    "test_accuracy = accuracy_score(y_test, y_pred)\n",
    "test_roc_auc = roc_auc_score(y_test, y_prob)\n",
    "report = classification_report(y_test, y_pred)\n",
    "\n",
    "# Guardar modelo y scaler\n",
    "joblib.dump(log_reg, \"modelo_riesgo_cardiovascular.pkl\")\n",
    "joblib.dump(scaler, \"scaler.pkl\")\n",
    "joblib.dump(imputer, \"imputer.pkl\")\n",
    "\n",
    "# Mostrar resultados\n",
    "print(f\"Precisión: {test_accuracy:.4f}\")\n",
    "print(f\"AUC-ROC: {test_roc_auc:.4f}\")\n",
    "print(\"\\nReporte de Clasificación:\")\n",
    "print(report)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "49b9191a-5f1f-42a6-bbc7-17f95b68b313",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
