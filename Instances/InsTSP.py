import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point

# ============================================================================
# Cargar polígono de Colombia (para filtrar puntos en tierra)
# ============================================================================
url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
world = gpd.read_file(url)
colombia_shape = world.query("ADMIN == 'Colombia'").geometry.values[0]

# ============================================================================
# Geocaja general de Colombia
# ============================================================================
colombia_bbox = {
    "lat": (-4.23, 12.46),
    "lon": (-79.02, -66.87)
}

def random_point_in_colombia():
    """
    Genera un punto aleatorio dentro del polígono de Colombia (no en mar).
    """
    while True:
        lat = random.uniform(*colombia_bbox["lat"])
        lon = random.uniform(*colombia_bbox["lon"])
        point = Point(lon, lat)
        if colombia_shape.contains(point):
            return round(lat, 6), round(lon, 6)

# ============================================================================
# Cálculo de distancia usando Haversine
# ============================================================================
def haversine_km(lat1, lon1, lat2, lon2, r_km=6371.0):

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return r_km * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

# ============================================================================
# Generación de nodos (Bodega + Clientes)
# ============================================================================
def generar_datos(num_clientes: int,
                  seed: int | None = None,
                  coords_bodega: tuple[float, float] | None = (4.7111, -74.0721),
                  id_bodega: str = "Bodega") -> pd.DataFrame:
    
    if seed is not None:
        random.seed(seed)

    registros = []

    # Nodo Bodega
    if coords_bodega is None:
        lat_b, lon_b = random_point_in_colombia()
    else:
        lat_b, lon_b = coords_bodega
    registros.append({
        "ID": id_bodega,
        "Tipo": "Bodega",
        "Latitud": round(lat_b, 6),
        "Longitud": round(lon_b, 6),
    })

    # Nodos Cliente
    for i in range(1, num_clientes + 1):
        lat, lon = random_point_in_colombia()
        registros.append({
            "ID": f"C{i:02d}",
            "Tipo": "Cliente",
            "Latitud": lat,
            "Longitud": lon,
        })

    return pd.DataFrame(registros)

# ============================================================================
# Matriz o lista de distancias
# ============================================================================
def distancias_tsp(df_nodos: pd.DataFrame,
                   modo: str = "long",
                   dirigido: bool = False) -> pd.DataFrame:
    
    ids = df_nodos["ID"].tolist()
    lat = dict(zip(ids, df_nodos["Latitud"]))
    lon = dict(zip(ids, df_nodos["Longitud"]))

    if modo == "long":
        filas = []
        if dirigido:
            for i in ids:
                for j in ids:
                    if i != j:
                        d = haversine_km(lat[i], lon[i], lat[j], lon[j])
                        filas.append({"Origen": i, "Destino": j, "Dist_km": d})
        else:
            for idx_i in range(len(ids)):
                for idx_j in range(idx_i + 1, len(ids)):
                    i, j = ids[idx_i], ids[idx_j]
                    d = haversine_km(lat[i], lon[i], lat[j], lon[j])
                    filas.append({"Origen": i, "Destino": j, "Dist_km": d})
        return pd.DataFrame(filas)

    elif modo == "wide":
        data = {}
        for i in ids:
            row = []
            for j in ids:
                row.append(0.0 if i == j else haversine_km(lat[i], lon[i], lat[j], lon[j]))
            data[i] = row
        return pd.DataFrame(data, index=ids)

    else:
        raise ValueError("`modo` debe ser 'long' o 'wide'.")

# ============================================================================
# Visualización en mapa (GeoPandas)
# ============================================================================
import plotly.graph_objects as go
import geopandas as gpd

def sucesor_a_tours(succ: dict):
    """
    Convierte un diccionario sucesor {i:j} (arcos activos) en una lista de rutas (listas de nodos).
    Soporta múltiples subtours.
    """
    usados = set()
    tours = []
    for start in succ:
        if start in usados:
            continue
        tour = [start]
        usados.add(start)
        nxt = succ[start]
        while nxt not in usados and nxt in succ:
            tour.append(nxt)
            usados.add(nxt)
            nxt = succ[nxt]
        tours.append(tour)
    return tours

def visualizar(clientes_df, rutas=None, pplot=True, nombre_archivo=None):
    """
    Visualización interactiva con Plotly de clientes, bodega y rutas.

    clientes_df: DataFrame con columnas ['ID','Latitud','Longitud']
    rutas:
        • None                      -> solo puntos
        • Lista de IDs              -> una sola ruta
        • Lista de listas de IDs    -> múltiples rutas
        • Dict {i:j} (sucesor TSP)  -> se convierte automáticamente a listas
    """
    # --- Coordenadas ---
    coord = clientes_df.set_index('ID')[['Latitud', 'Longitud']].to_dict('index')

    # --- Normalización de rutas ---
    rutas_norm = []
    if rutas is None:
        rutas_norm = []
    elif isinstance(rutas, dict):
        rutas_norm = sucesor_a_tours(rutas)
    elif all(isinstance(r, str) for r in rutas):
        rutas_norm = [rutas]
    else:
        rutas_norm = rutas  # asumimos lista de listas

    # --- Crear figura ---
    fig = go.Figure()

    # --- Colores para distinguir rutas ---
    colors = ["black", "green", "purple", "orange", "brown", "pink", "cyan", "gray", "teal"]

    # --- Dibujar rutas ---
    for idx, ruta in enumerate(rutas_norm):
        lats = [coord[c]['Latitud'] for c in ruta if c in coord]
        lons = [coord[c]['Longitud'] for c in ruta if c in coord]
        fig.add_trace(go.Scattergeo(
            lon=lons, lat=lats,
            mode='lines+markers',
            line=dict(width=2, color='grey'),
            marker=dict(size=6),
            name=f'Ruta {idx+1}',
            hovertext=ruta,
            hoverinfo='text'
        ))

    # --- Agregar puntos (bodega y clientes) ---
    leyenda_clientes_agregada = False
    for cid, datos in coord.items():
        if cid == 'Bodega':
            fig.add_trace(go.Scattergeo(
                lon=[datos['Longitud']], lat=[datos['Latitud']],
                mode='markers',
                text=[cid],
                marker=dict(size=12, color='red', symbol='star'),
                name='Bodega'
            ))
        else:
            fig.add_trace(go.Scattergeo(
                lon=[datos['Longitud']], lat=[datos['Latitud']],
                mode='markers',
                marker=dict(size=6, color='blue'),
                name='Clientes' if not leyenda_clientes_agregada else None,
                hovertext=cid,
                hoverinfo='text',
                showlegend=not leyenda_clientes_agregada
            ))
            leyenda_clientes_agregada = True

    # --- Layout ---
    fig.update_layout(
        title="Ubicación de Clientes y Rutas en Colombia",
        geo=dict(
            scope='south america',
            projection_type='mercator',
            showland=True,
            landcolor="rgb(255, 255, 255)",
            showocean=True,
            oceancolor="rgb(173, 216, 230)",
            showcountries=True,
            countrycolor="black",
            center=dict(lat=4.5, lon=-74.0),
            fitbounds="locations"
        ),
        width=900,
        height=900,
        showlegend=False
    )

    if pplot:
        fig.show()

    if nombre_archivo:
        # Necesitas kaleido instalado: pip install -U kaleido
        fig.write_image(f"Assets/{nombre_archivo}.png", width=1200, height=900)

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
# Crear GIF
# ============================================================================
from PIL import Image
import glob, re, os

def hacer_gif(pattern="Assets/*.png", salida="Assets/tour.gif", ms_por_frame=600, loop=0):
    # Buscar y ordenar por número en el nombre
    files = sorted(glob.glob(pattern), key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group(1)))
    if not files:
        raise FileNotFoundError("No encontré PNGs que coincidan con el patrón.")
    imgs = [Image.open(f) for f in files]
    imgs[0].save(
        salida,
        save_all=True,
        append_images=imgs[1:],
        duration=ms_por_frame,
        loop=loop
    )

