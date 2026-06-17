#!/bin/bash
echo "============================================"
echo " Smart City Traffic Forecasting - Setup"
echo "============================================"

echo "[1/3] Installing dependencies..."
pip install -r requirements.txt

echo "[2/3] Done! Launching Jupyter Notebook..."
jupyter notebook Smart_City_Traffic_Forecasting.ipynb
