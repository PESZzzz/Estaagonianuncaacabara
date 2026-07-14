@echo off
:: If you want to set default values, please edit the line below.
set ngrok_token=

if "%ngrok_token%"=="" (
    set /p ngrok_token=input ngrok token: 
) else (
    echo using default ngrok token.
)

main.exe cui --https false --no_cui True --ngrok_token %ngrok_token%
