#!/bin/bash

# Nombre del entorno Conda y archivo YAML
ENV_NAME="tfg2"
ENV_YAML="environment.yaml"

# Verifica que conda esté disponible
if ! command -v conda &> /dev/null; then
    echo "❌ Conda no está instalado o no está en el PATH."
    exit 1
fi

# Cargar entorno base de Conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# Comprobar si el entorno ya existe
if conda env list | grep -qE "^$ENV_NAME\s"; then
    echo "⚠️ El entorno '$ENV_NAME' ya existe. Eliminándolo..."
    conda remove --name "$ENV_NAME" --all -y
fi

# Crear el entorno desde el archivo YAML
echo "📦 Creando entorno Conda desde $ENV_YAML..."
conda env create -f "$ENV_YAML" --name "$ENV_NAME"

# Activar el entorno
echo "🔁 Activando entorno $ENV_NAME..."
conda activate "$ENV_NAME"

# Exportar variables necesarias
echo "⚙️ Exportando LD_PRELOAD..."
export LD_PRELOAD=/usr/lib/libstdc++.so.6

# Clonar el repositorio con submódulos
echo "📁 Clonando repositorio llama-cpp-python con submódulos..."
git clone --recurse-submodules https://github.com/abetlen/llama-cpp-python.git
cd llama-cpp-python || exit 1

# Exportar variables para compilación con soporte CUDA
echo "🚀 Preparando compilación con soporte CUDA..."
export CMAKE_ARGS="-DLLAMA_CUDA=on"
export FORCE_CMAKE=1

# Instalar el paquete compilando desde fuente
echo "🔧 Instalando llama-cpp-python desde fuente..."
pip install .

# Volver al directorio original
cd ..

echo "✅ Instalación completada con éxito."
