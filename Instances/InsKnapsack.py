import pandas as pd
import random

# ============================================================================
# Coordenadas geográficas aproximadas por ciudad (para simulación de ubicación)
# ============================================================================
ciudades = {
    "Bogotá":       {"lat": (4.5, 4.85),     "lon": (-74.2, -74.0)},
    "Medellín":     {"lat": (6.15, 6.4),     "lon": (-75.65, -75.5)},
    "Cartagena":    {"lat": (10.3, 10.5),    "lon": (-75.6, -75.4)},
    "Barranquilla": {"lat": (10.9, 11.1),    "lon": (-74.9, -74.7)},
    "Cali":         {"lat": (3.3, 3.6),      "lon": (-76.6, -76.4)}
}

# ============================================================================
# Generador de datos de obras públicas con atributos económicos y ubicación
# ============================================================================
def generar_datos(num_registros: int):
    """
    ============================================================================
    Generador sintético de datos de obras públicas
    ─────────────────────────────────────────────────────────────────────────────
    Crea una tabla de obras con los siguientes campos:
      • Ciudad                       → ciudad donde se ubica la obra
      • Costo de ejecución (M COP)  → inversión estimada (valor entre 1 y 10)
      • Empleos generados (mil)     → estimación de impacto laboral (1 a 15)
      • Latitud / Longitud          → ubicación geográfica aleatoria dentro de
                                       los límites definidos por ciudad

    Entradas
      • num_registros : int
          Número total de obras a generar

    Salida
      • pd.DataFrame con las columnas:
          ['Ciudad',
           'Costo de ejecución (en millones de pesos)',
           '# de empleos generados (en miles)',
           'Latitud', 'Longitud']
    ============================================================================
    """

    datos = []

    # -------------------------------------------------------------------------
    # 1. Generar una obra aleatoria por cada iteración
    # -------------------------------------------------------------------------
    for _ in range(num_registros):
        ciudad = random.choice(list(ciudades.keys()))              # Seleccionar ciudad aleatoriamente
        lat_range = ciudades[ciudad]["lat"]
        lon_range = ciudades[ciudad]["lon"]

        # -- Generar valores económicos
        costo = round(random.uniform(1, 10), 0)                    # Costo en millones (redondeado)
        empleo = round(random.uniform(1, 15), 0)                   # Empleos en miles (redondeado)

        # -- Generar ubicación geográfica dentro de los rangos definidos
        lat = round(random.uniform(*lat_range), 6)
        lon = round(random.uniform(*lon_range), 6)

        # -- Armar el registro
        datos.append({
            "Ciudad": ciudad,
            "Costo de ejecución (en millones de pesos)": costo,
            "# de empleos generados (en miles)": empleo,
            "Latitud": lat,
            "Longitud": lon
        })

    # -------------------------------------------------------------------------
    # 2. Convertir la lista de dicts en DataFrame
    # -------------------------------------------------------------------------
    df = pd.DataFrame(datos)
    return df
