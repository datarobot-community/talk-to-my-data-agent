# Function to load .env file
function Import-EnvFile {
    if (-not (Test-Path '.env')) {
        Write-Host "Error: .env file not found."
        Write-Host "Please create a .env file with VAR_NAME=value pairs."
        return $false
    }
    
    $envVarsJson = python -c "import json; from quickstart import load_dotenv; env_vars = load_dotenv(); print(json.dumps(env_vars))"
    
    $envVars = $envVarsJson | ConvertFrom-Json
    
    $envVars.PSObject.Properties | ForEach-Object {
        [System.Environment]::SetEnvironmentVariable($_.Name, $_.Value)
    }
    
    Write-Host "Environment variables from .env have been set."
    return $true
}
# Function to activate the virtual environment if it exists
function Enable-VirtualEnvironment {
    if (Test-Path '.venv\Scripts\Activate.ps1') {
        Write-Host "Activated virtual environment found at .venv\Scripts\Activate.ps1"
        . '.venv\Scripts\Activate.ps1'
    }
}

# Main execution
Enable-VirtualEnvironment
Import-EnvFile