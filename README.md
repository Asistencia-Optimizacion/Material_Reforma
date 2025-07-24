# Material de Reforma en Investigación de Operaciones

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Resumen

Este repositorio contiene material educativo y de investigación para el estudio de problemas clásicos en Investigación de Operaciones. El proyecto proporciona implementaciones computacionales rigurosas de algoritmos de optimización, programación dinámica y métodos heurísticos aplicados a problemas fundamentales:

- **Problema de la Mochila** (Knapsack Problem)
- **Problema del Viajante de Comercio** (Traveling Salesperson Problem)
- **Gestión de Inventarios** (Inventory Management)
- **Optimización de Asignación de Recursos Públicos**

La metodología pedagógica integra fundamentos teóricos con implementaciones computacionales en Python, proporcionando un entorno de aprendizaje interactivo mediante notebooks de Jupyter y visualizaciones avanzadas.

## Casos de Estudio

### Caso 1: MinObras - Optimización de Inversión en Infraestructura Pública

**Problema**: Maximización de empleos generados mediante la selección óptima de proyectos de infraestructura bajo restricciones presupuestarias.

**Formulación Matemática**: Problema de mochila con función objetivo de maximización de beneficio social.

**Metodologías Implementadas**:
- Programación Lineal Entera utilizando PuLP
- Algoritmos de Programación Dinámica
- Heurísticas constructivas tipo Greedy

> Además de un Notebook teórico con explicaciones detallas a fondo de todos los algoritmos de Programación Dinámica.

### Caso 2: OptiCoffee - Planificación Estratégica de Producción e Inventarios

**Problema**: Minimización de costos totales de producción y almacenamiento a lo largo de un horizonte de planificación multiperiodo.

**Formulación Matemática**: Problema de lot-sizing con demanda determinística variable.

**Metodologías Implementadas**:
- Programación Dinámica con recursión hacia adelante
- Modelo EOQ (Economic Order Quantity)
- Política de inventarios (s,S)
- Heurística Silver-Meal

### Caso 3: LuzLuna - Optimización de Rutas de Distribución

**Problema**: Minimización de distancia total recorrida para visitar un conjunto de clientes geográficamente distribuidos en Colombia.

**Formulación Matemática**: Traveling Salesperson Problem (TSP) con formulación de programación lineal entera mixta.

**Metodologías Implementadas**:
- Formulación MIP con restricciones de eliminación de subtours
- Heurística del Vecino Más Cercano (Nearest Neighbor)
- Heurística de Inserción de Menor Costo (Cheapest Insertion)
- Algoritmo de mejora 2-OPT

## Arquitectura del Sistema

```
Material_Reforma/
├── Clases/                         # Notebooks educativos principales
│   ├── MinObras.ipynb              # Optimización de inversión pública
│   ├── LuzLuna.ipynb               # Problema del viajante de comercio
│   ├── OptiCoffee.ipynb            # Gestión de inventarios multiperiodo
│   ├── DP/Teoria.ipynb             # Gestión de inventarios multiperiodo
|
│   └── Assets/                     # Recursos de visualización
|
├── DP/                             # Módulos de Programación Dinámica
│   ├── Algorithms/                 # Implementaciones de algoritmos DP/RL
│   │   ├── policy_evaluation.py    # Evaluación de políticas
│   │   ├── policy_iteration.py     # Iteración de políticas
│   │   └── value_iteration.py      # Iteración de valores
|
│   ├── Env/                        # Entornos de simulación
│   │   ├── Inventory.py            # Entorno de gestión de inventarios
│   │   └── Knapsack.py             # Entorno del problema de mochila
│   └── Visual/                     # Módulos de visualización
|
│       ├── policy_dag.py           # Grafos dirigidos de políticas
│       └── value_states.py         # Visualización de funciones de valor
|
├── Instances/                      # Generadores de instancias sintéticas
│   ├── InsInventory.py             # Generador de problemas de inventario
│   ├── InsKnapsack.py              # Generador de problemas de mochila
│   └── InsTSP.py                   # Generador de instancias TSP
|
├── Ruteo/                          # Algoritmos de optimización de rutas
│   ├── Heuristicas.py              # Implementación de heurísticas TSP
│   └── Lazy_TSP.py                 # Formulación con restricciones perezosas
|
├── Producción/                     # Modelos de planificación de producción
│   ├── EOQ.py                      # Economic Order Quantity
│   ├── Silver_Meal.py              # Heurística Silver-Meal
│   └── sS.py                       # Política de inventarios (s,S)
|
├── requirements.txt                # Especificación de dependencias
└── README.md                       # Documentación del proyecto
```

### Procedimiento de Instalación

#### Paso 1: Obtención del Código Fuente

**Opción A: Clonación del repositorio**
```bash
git clone https://github.com/Asistencia-Optimizacion/Material_Reforma.git
cd Material_Reforma
```

**Opción B: Descarga directa**
```bash
wget https://github.com/Asistencia-Optimizacion/Material_Reforma/archive/main.zip
unzip main.zip
cd Material_Reforma-main
```

#### Paso 2: Configuración del Entorno Virtual

Se recomienda encarecidamente el uso de entornos virtuales para aislar las dependencias del proyecto:

**En sistemas Unix/Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**En sistemas Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

#### Paso 3: Instalación de Dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Paso 4: Verificación de la Instalación

Ejecute el siguiente comando para verificar que todas las dependencias críticas estén correctamente instaladas:

```bash
python -c "
import pulp, pandas, numpy, matplotlib, plotly, geopandas, networkx
print('Verificación exitosa: Todas las dependencias están disponibles')
"
```

#### Configuración de Jupyter Notebook

Si Jupyter Notebook no está incluido en su instalación de Python:

```bash
pip install jupyter jupyterlab
```

Para iniciar el servidor de notebooks:

```bash
jupyter notebook
```

#### Solución de Problemas Comunes

**Error de instalación de GeoPandas en Windows:**
```bash
conda install geopandas
```

**Error de memoria en notebooks grandes:**
```bash
export JUPYTER_RUNTIME_DIR=/tmp
jupyter notebook --NotebookApp.max_buffer_size=2147483647
```

#### Inicio del Entorno Jupyter

```bash
jupyter notebook
```

El comando anterior iniciará un servidor local accesible mediante navegador web en `http://localhost:8888`.
O en su defecto puede usar Visual Studio Code con todas las extenciones.

#### Navegación y Ejecución

1. **Acceso a materiales**: Navegue hacia el directorio `Clases/`
2. **Selección de caso de estudio**: 
   - `MinObras.ipynb`: Optimización de inversión pública
   - `LuzLuna.ipynb`: Problemas de ruteo y TSP
   - `OptiCoffee.ipynb`: Gestión de inventarios multiperiodo
3. **Ejecución secuencial**: Ejecute las celdas en orden utilizando `Shift + Enter`

## Especificación de Dependencias

### Librerías de Optimización y Cálculo Numérico

| Paquete | Versión | Propósito |
|---------|---------|-----------|
| `pulp` | ≥2.7.0 | Modelado y resolución de problemas de programación lineal |
| `numpy` | ≥1.24.0 | Operaciones matriciales y cálculo numérico |
| `pandas` | ≥2.0.0 | Manipulación y análisis de estructuras de datos |

### Librerías de Visualización y Análisis Geoespacial

| Paquete | Versión | Propósito |
|---------|---------|-----------|
| `matplotlib` | ≥3.7.0 | Generación de gráficos estáticos |
| `plotly` | ≥5.17.0 | Visualizaciones interactivas y dashboards |
| `geopandas` | ≥0.14.0 | Análisis y manipulación de datos geoespaciales |
| `shapely` | ≥2.0.0 | Operaciones geométricas y análisis espacial |

### Librerías de Análisis de Redes y Procesamiento

| Paquete | Versión | Propósito |
|---------|---------|-----------|
| `networkx` | ≥3.0 | Análisis de grafos y redes complejas |
| `Pillow` | ≥10.0.0 | Procesamiento y manipulación de imágenes |

### Entorno de Desarrollo Interactivo

| Paquete | Versión | Propósito |
|---------|---------|-----------|
| `jupyter` | ≥1.0.0 | Entorno de notebooks interactivos |
| `ipython` | ≥8.0.0 | Shell de Python mejorado |

Para la especificación completa de versiones, consulte el archivo `requirements.txt`.

## Especificaciones Técnicas

### Algoritmos y Metodologías Implementadas

#### Técnicas de Optimización Exacta

**Programación Lineal Entera**
- Formulaciones MIP para problemas de mochila y asignación de recursos
- Implementación de restricciones de eliminación de subtours para TSP
- Uso del solver CBC a través de la interfaz PuLP

**Programación Dinámica**
- Algoritmo de iteración de valores (Value Iteration)
- Algoritmo de iteración de políticas (Policy Iteration)
- Evaluación de políticas mediante sistemas de ecuaciones lineales
- Implementación de principio de optimalidad de Bellman

#### Métodos Heurísticos y Metaheurísticos

**Heurísticas Constructivas**
- Algoritmo del Vecino Más Cercano para TSP (complejidad O(n²))
- Heurística de Inserción de Menor Costo (complejidad O(n²))
- Algoritmo Greedy para problema de mochila
- Entre otros...

**Heurísticas de Mejora**
- Algoritmo 2-OPT para TSP (complejidad O(n²) por iteración)
- Procedimientos de búsqueda local con criterios de parada
- Entre otros...

#### Capacidades de Visualización

**Representaciones Geoespaciales**
- Mapas interactivos con coordenadas georreferenciadas de Colombia
- Visualización de rutas óptimas sobre mapas base de OpenStreetMap
- Animaciones de convergencia de algoritmos iterativos

**Análisis Comparativo**
- Dashboards interactivos para análisis de rendimiento
- Gráficos de comparación de calidad de solución vs. tiempo computacional
- Visualización de fronteras de Pareto para objetivos múltiples

**Representación de Políticas**
- Grafos dirigidos acíclicos (DAG) para políticas de programación dinámica
- Heatmaps de funciones de valor sobre espacios de estados
- Visualización de trayectorias óptimas en problemas secuenciales

#### Generación de Instancias Sintéticas

**Parámetros Configurables**
- Distribuciones probabilísticas paramétricas para costos y demandas
- Generación de coordenadas geográficas realistas dentro de Colombia
- Control de semillas aleatorias para reproducibilidad de experimentos