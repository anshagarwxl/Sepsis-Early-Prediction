#!/usr/bin/env powershell
# Script to run the Sepsis RAG Assistant Streamlit app

# Set environment variables
$env:GEMINI_API_KEY = "AIzaSyAu_8lYWrc4UreqRfvSX5eXMXJ1f17W2Xw"

# Navigate to project directory
Set-Location "C:\Users\vaxds\Documents\HACKATHON\sepsis-rag-assistant"

Write-Host "Starting Sepsis RAG Assistant..." -ForegroundColor Green
Write-Host "Gemini API Key: Set" -ForegroundColor Green
Write-Host "Port: 8502" -ForegroundColor Green
Write-Host "URL: http://localhost:8502" -ForegroundColor Cyan

# Start Streamlit app
echo "" | C:\Users\vaxds\Documents\HACKATHON\sepsis-rag-assistant\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502 --server.headless false