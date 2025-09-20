# PowerShell script to run Streamlit with environment variables
Set-Location "C:\Users\vaxds\Documents\HACKATHON\sepsis-rag-assistant"
$env:GEMINI_API_KEY = "AIzaSyAu_8lYWrc4UreqRfvSX5eXMXJ1f17W2Xw"
Write-Host "Starting Streamlit with GEMINI_API_KEY set..." -ForegroundColor Green
& ".\.venv\Scripts\streamlit.exe" run app.py --server.headless true