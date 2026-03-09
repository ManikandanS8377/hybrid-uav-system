@echo off
:: ============================================================
::  Local MQTT Broker Setup for ESP32 Drone
::  Run this ONCE on your Windows PC
:: ============================================================

echo.
echo ============================================================
echo  Step 1: Finding your PC's local IP address
echo ============================================================
echo.
echo Your WiFi IP addresses:
ipconfig | findstr /i "IPv4" | findstr /v "127.0.0.1"
echo.
echo COPY the IP address shown above (e.g. 192.168.1.x)
echo You will need to put it in Config.h and mqtt_sender.py
echo.

echo ============================================================
echo  Step 2: Check if Mosquitto is installed
echo ============================================================
where mosquitto >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] Mosquitto is already installed.
    goto :start_broker
)

echo Mosquitto not found. Opening download page...
echo Please download and install from: https://mosquitto.org/download/
echo Choose the Windows installer (.exe)
echo After installing, re-run this script.
start https://mosquitto.org/download/
pause
exit /b

:start_broker
echo.
echo ============================================================
echo  Step 3: Starting local MQTT broker
echo ============================================================
echo.

:: Write a minimal open config (allow anonymous, listen on all interfaces)
echo listener 1883 0.0.0.0 > "%TEMP%\mosquitto_drone.conf"
echo allow_anonymous true >> "%TEMP%\mosquitto_drone.conf"

echo Starting Mosquitto on port 1883...
echo Keep this window open while flying!
echo.
echo Press Ctrl+C to stop the broker.
echo.
mosquitto -c "%TEMP%\mosquitto_drone.conf" -v
