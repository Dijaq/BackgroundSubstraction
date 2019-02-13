# Background Substraction using Local SVD Binary Pattern
El presente proyecto es una implementación en c++ del algoritmo LSBP

# Ejecucion
Requisitos:
- Sistema Operativo: Ubuntu
- Opencv 3.2.0 o superior
- Compilador gcc, g++

Ejecucion:
Para ejecutar el proyecto una ves que tenga los requisitos siga los siguientes pasos:
- Clone o descargue el proyecto.
- Abra un terminal y dirijase a la carpeta del proyecto que clono.
- Compile con el siguiente comando:
g++ -ggdb `pkg-config --cflags opencv` -o `main main.cpp .cpp` main.cpp `pkg-config --libs opencv`
- Ejecute la aplicacion con el comando:
./main

# Fuentes
- [1] Lili Guo, Dan Xu, and Zhenping Qiang, “Background Subtraction using Local SVD Binary Pattern”.
- [2] M. Hofmann, P. Tiefenbacher, and G. Rigoll, “Background segmentation with feedback: The pixel-based adaptive segmenter.”.
