@echo off
setlocal
py -m pip install --upgrade pip
py -m pip install -r requirements.txt pyinstaller
py -m PyInstaller --noconfirm --clean --onefile --windowed --name QRImageForge --collect-all qrcode --collect-all PIL --collect-all numpy main.py
if errorlevel 1 (
  echo.
  echo Build failed. Review the error above.
  exit /b 1
)
echo.
echo Build complete: dist\QRImageForge.exe
endlocal
