#!/bin/bash
# Deploy Chrono Dashboard to VPS
# Usage: bash deploy.sh
set -e

VPS="ubuntu@57.129.126.202"
APP_DIR="/opt/chrono-dashboard"
SERVICE="chrono-dashboard"

echo "==> Conectando al VPS..."

ssh -o StrictHostKeyChecking=no "$VPS" bash << 'REMOTE'
set -e
APP_DIR="/opt/chrono-dashboard"

# Verificar puerto 8502 libre
if ss -tlnp | grep -q ':8502 '; then
    echo "ERROR: Puerto 8502 ocupado. Edita el .service y esta variable."
    exit 1
fi
echo "Puerto 8502 libre OK"

# Crear directorio
sudo mkdir -p "$APP_DIR"
sudo chown ubuntu:ubuntu "$APP_DIR"

echo "Directorio listo: $APP_DIR"
REMOTE

echo "==> Copiando archivos..."
scp chrono_dashboard.py requirements.txt "$VPS:$APP_DIR/"
scp deploy/chrono-dashboard.service "$VPS:/tmp/"
scp deploy/chrono-dashboard.conf "$VPS:/tmp/"

echo "==> Configurando entorno Python..."
ssh "$VPS" bash << 'REMOTE'
set -e
APP_DIR="/opt/chrono-dashboard"
cd "$APP_DIR"

if [ ! -d venv ]; then
    python3 -m venv venv
fi
venv/bin/pip install --upgrade pip -q
venv/bin/pip install -r requirements.txt -q
echo "Dependencias instaladas OK"
REMOTE

echo "==> Instalando servicio systemd..."
ssh "$VPS" bash << 'REMOTE'
sudo mv /tmp/chrono-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable chrono-dashboard
sudo systemctl restart chrono-dashboard
sleep 2
sudo systemctl is-active chrono-dashboard && echo "Servicio activo OK" || echo "ERROR: servicio no arrancó"
REMOTE

echo "==> Configurando nginx..."
ssh "$VPS" bash << 'REMOTE'
sudo mv /tmp/chrono-dashboard.conf /etc/nginx/sites-available/chrono-dashboard.conf
sudo ln -sf /etc/nginx/sites-available/chrono-dashboard.conf /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
echo "Nginx OK"
REMOTE

echo "==> Obteniendo certificado SSL..."
ssh "$VPS" "sudo certbot --nginx -d chrono-dashboard.ddns.net --non-interactive --agree-tos -m g3ov3r@gmail.com 2>&1 | tail -5" || echo "Certbot: ejecuta manualmente si falla"

echo ""
echo "Deploy completado."
echo "URL: https://chrono-dashboard.ddns.net"
