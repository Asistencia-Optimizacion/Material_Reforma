import pandas as pd
import random
import math
import matplotlib.pyplot as plt

# ============================================================================
# Geocaja de Bogotá para muestreo de coordenadas sintéticas
# ============================================================================
# Define un rango de latitudes y longitudes para simular ubicaciones dentro
# del área metropolitana de Bogotá (datos sintéticos, no reales).
# ============================================================================
bogota_bbox = {
    "lat": (4.5, 4.85),
    "lon": (-74.2, -74.0)
}


# ============================================================================
# Utilidades básicas (coordenadas, distancia geográfica)
# ============================================================================

def _random_point_in_bogota():
    """
    Retorna una tupla (lat, lon) aleatoria dentro de los límites de Bogotá.

    Salida:
      • Coordenadas en grados decimales (latitud, longitud), redondeadas a 6 decimales.
    """
    lat = random.uniform(*bogota_bbox["lat"])
    lon = random.uniform(*bogota_bbox["lon"])
    return round(lat, 6), round(lon, 6)


def _haversine_km(lat1, lon1, lat2, lon2, r_km: float = 6371.0) -> float:
    """
    Calcula la distancia del gran círculo entre dos puntos en la Tierra usando la fórmula de Haversine.

    Parámetros:
      • lat1, lon1 : coordenadas del primer punto (en grados)
      • lat2, lon2 : coordenadas del segundo punto (en grados)
      • r_km       : radio de la Tierra en kilómetros (por defecto 6371)

    Retorna:
      • Distancia entre los dos puntos en kilómetros (float)
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r_km * c


# ============================================================================
# Generar nodos para instancia de TSP: 1 bodega + N clientes
# ============================================================================
def generar_datos(num_clientes: int,
                  seed: int | None = None,
                  coords_bodega: tuple[float, float] | None = None,
                  id_bodega: str = "Bodega") -> pd.DataFrame:
    """
    ============================================================================
    Generar nodos sintéticos para una instancia de TSP en Bogotá
    ─────────────────────────────────────────────────────────────────────────────
    Retorna un DataFrame con coordenadas de una bodega y N clientes.

    Columnas devueltas:
      ['ID', 'Tipo', 'Latitud', 'Longitud']

    Parámetros:
      • num_clientes   : int  → número de clientes a generar
      • seed           : int  → semilla aleatoria (opcional)
      • coords_bodega  : tuple(float, float) → coordenadas fijas para la bodega (opcional)
      • id_bodega      : str  → identificador para el nodo bodega

    Notas:
      • La bodega aparece siempre en la primera fila.
      • Datos puramente sintéticos para fines educativos.
    ============================================================================
    """
    if seed is not None:
        random.seed(seed)

    registros = []

    # -- Nodo bodega (origen) --
    if coords_bodega is None:
        lat_b, lon_b = _random_point_in_bogota()
    else:
        lat_b, lon_b = coords_bodega
    registros.append({
        "ID": id_bodega,
        "Tipo": "Bodega",
        "Latitud": round(lat_b, 6),
        "Longitud": round(lon_b, 6),
    })

    # -- Nodos cliente --
    for i in range(1, num_clientes + 1):
        lat, lon = _random_point_in_bogota()
        registros.append({
            "ID": f"C{i:02d}",
            "Tipo": "Cliente",
            "Latitud": lat,
            "Longitud": lon,
        })

    return pd.DataFrame(registros)


# ============================================================================
# Cálculo de distancias entre nodos (matriz o lista de arcos)
# ============================================================================
def distancias_tsp(df_nodos: pd.DataFrame,
                   modo: str = "long",
                   dirigido: bool = False) -> pd.DataFrame:
    """
    ============================================================================
    Construye la representación de costos de arco entre nodos para el TSP.
    ─────────────────────────────────────────────────────────────────────────────
    Entrada:
      • df_nodos : DataFrame con columnas ['ID', 'Latitud', 'Longitud']
      • modo     : 'long' o 'wide'
          - 'long' → lista de arcos con distancias (formato largo)
          - 'wide' → matriz cuadrada de distancias (formato ancho)
      • dirigido : bool (solo aplica en modo 'long')

    Salida:
      • DataFrame con distancias según el modo especificado.

    Ejemplo:
      df_nodes = generar_datos(5)
      dist_long = distancias_tsp(df_nodes, modo="long")
      dist_mat  = distancias_tsp(df_nodes, modo="wide")
    ============================================================================
    """
    req = {"ID", "Latitud", "Longitud"}
    if not req.issubset(df_nodos.columns):
        faltan = req - set(df_nodos.columns)
        raise ValueError(f"Faltan columnas requeridas: {faltan}")

    ids = df_nodos["ID"].tolist()
    lat = dict(zip(ids, df_nodos["Latitud"]))
    lon = dict(zip(ids, df_nodos["Longitud"]))

    if modo == "long":
        filas = []
        if dirigido:
            for i in ids:
                for j in ids:
                    if i == j:
                        continue
                    d = _haversine_km(lat[i], lon[i], lat[j], lon[j])
                    filas.append({"Origen": i, "Destino": j, "Dist_km": d})
        else:
            for idx_i in range(len(ids)):
                for idx_j in range(idx_i + 1, len(ids)):
                    i, j = ids[idx_i], ids[idx_j]
                    d = _haversine_km(lat[i], lon[i], lat[j], lon[j])
                    filas.append({"Origen": i, "Destino": j, "Dist_km": d})
        return pd.DataFrame(filas)

    elif modo == "wide":
        data = {}
        for i in ids:
            row = []
            for j in ids:
                if i == j:
                    row.append(0.0)
                else:
                    row.append(_haversine_km(lat[i], lon[i], lat[j], lon[j]))
            data[i] = row
        mat = pd.DataFrame(data, index=ids)
        mat = mat.loc[ids, ids]
        mat.index.name = "ID"
        return mat

    else:
        raise ValueError("`modo` debe ser 'long' o 'wide'.")


# ============================================================================
# Reconstrucción de ruta a partir de conjunto de arcos
# ============================================================================
def reconstruir_tour(sel_arcos, N, inicio='Bodega'):
    """
    Reconstruye una ruta ordenada a partir de un conjunto de arcos seleccionados.

    Parámetros:
      • sel_arcos : conjunto de tuplas (i, j) que representan arcos dirigidos
      • N         : lista de nodos (identificadores únicos)
      • inicio    : nodo inicial desde donde se empieza el tour

    Salida:
      • tour : lista ordenada de nodos visitados (cierra ciclo si es TSP completo)

    Nota:
      Lanza error si no se puede reconstruir un tour válido (p. ej., por subtours).
    """
    succ = {i: None for i in N}
    for i, j in sel_arcos:
        succ[i] = j

    tour = [inicio]
    curr = inicio
    for _ in range(len(N)):
        nxt = succ.get(curr)
        if nxt is None:
            raise RuntimeError(f"Nodo {curr} sin sucesor en la solución.")
        if nxt == inicio:
            break
        tour.append(nxt)
        curr = nxt
    else:
        raise RuntimeError("No se cerró un tour válido; puede haber subtours.")

    tour.append(inicio)
    return tour


# ============================================================================
# Visualización de ruta geográfica en coordenadas (matplotlib)
# ============================================================================
def graficar_ruta(clientes_df: pd.DataFrame, ruta: list[str]):
    """
    Dibuja la ruta de reparto sobre un plano lat/lon con identificación visual.

    Parámetros:
      • clientes_df : DataFrame con nodos y coordenadas ['ID','Latitud','Longitud']
      • ruta        : lista de IDs de nodos en el orden en que se visitan

    La visualización:
      ▸ Marca la bodega en rojo, clientes en azul
      ▸ Dibuja la ruta en negro
      ▸ Añade márgenes automáticos al gráfico
    """
    # Asegurar que la ruta esté cerrada con la bodega al inicio y al final
    if ruta[0] != 'Bodega':
        ruta = ['Bodega'] + ruta
    if ruta[-1] != 'Bodega':
        ruta.append('Bodega')

    # Coordenadas por ID
    coord = clientes_df.set_index('ID')[['Latitud', 'Longitud']].to_dict('index')
    lats = [coord[c]['Latitud'] for c in ruta]
    lons = [coord[c]['Longitud'] for c in ruta]

    # Rango dinámico con margen
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    margin_y = (max_lat - min_lat) 
    margin_x = (max_lon - min_lon)

    # Graficar
    plt.figure(figsize=(16, 10))
    for cid in clientes_df['ID']:
        lat, lon = coord[cid]['Latitud'], coord[cid]['Longitud']
        if cid == 'Bodega':
            plt.scatter(lon, lat, color='red', s=100, label='Bodega')
            plt.annotate('Bodega', (lon, lat), textcoords="offset points", xytext=(5, 5))
        else:
            plt.scatter(lon, lat, color='blue', s=30)

    plt.plot(lons, lats, color='black', linewidth=1)
    plt.title("Ruta de Reparto")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.xlim(min_lon - margin_x, max_lon + margin_x)
    plt.ylim(min_lat - margin_y, max_lat + margin_y)
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    plt.show()
