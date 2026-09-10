FROM python:3.11-slim

WORKDIR /app

# Evita que Python escriba archivos .pyc en el disco
ENV PYTHONDONTWRITEBYTECODE=1
# Forza a que la salida de la consola no se quede en el buffer (ideal para los logs de Rich)
ENV PYTHONUNBUFFERED=1

# Copiar solo el requirements primero para aprovechar la caché de Docker
COPY requirements.txt .

# Instalar dependencias sin guardar caché interna de pip
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Mantiene el contenedor listo para interactuar por consola
CMD ["bash"]