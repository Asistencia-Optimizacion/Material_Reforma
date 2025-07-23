import pandas as pd
import random

# ============================================================================
# Coordenadas geográficas aproximadas por ciudad
# Estas coordenadas son usadas para generar ubicaciones aleatorias de obras
# dentro de límites geográficos realistas.
# ============================================================================
ciudades = {
    "Bogotá": {"lat": (4.5, 4.85), "lon": (-74.2, -74.0)},
    # Si se desea ampliar:
    # "Medellín":     {"lat": (6.15, 6.4),     "lon": (-75.65, -75.5)},
    # "Cartagena":    {"lat": (10.3, 10.5),    "lon": (-75.6, -75.4)},
    # "Barranquilla": {"lat": (10.9, 11.1),    "lon": (-74.9, -74.7)},
    # "Cali":         {"lat": (3.3, 3.6),      "lon": (-76.6, -76.4)}
}

# ============================================================================
# Generador de datos sintéticos de obras públicas
# ============================================================================
def generar_datos(num_registros: int):
    """
    Genera un DataFrame con información sintética de obras públicas.

    Parámetros
    ----------
    num_registros : int
        Número total de obras a generar.

    Salida
    ------
    pd.DataFrame con las columnas:
        'Obra' -> identificador de la obra
        'Ciudad' -> nombre de la ciudad
        'Costo de ejecución (en millones de pesos)' -> inversión estimada
        '# de empleos generados (en miles)' -> empleos potenciales
        'Latitud' y 'Longitud' -> ubicación aleatoria dentro de la ciudad
    """
    datos = []

    for i in range(num_registros):
        # 1. Seleccionar ciudad aleatoria
        ciudad = random.choice(list(ciudades.keys()))
        lat_range = ciudades[ciudad]["lat"]
        lon_range = ciudades[ciudad]["lon"]

        # 2. Atributos económicos
        costo = round(random.uniform(1, 10), 0)    # costo entre 1 y 10 M COP
        empleo = round(random.uniform(1, 15), 0)   # empleos entre 1 y 15 mil

        # 3. Ubicación aleatoria dentro de los límites de la ciudad
        lat = round(random.uniform(*lat_range), 6)
        lon = round(random.uniform(*lon_range), 6)

        # 4. Agregar registro
        datos.append({
            "Obra": f"Obra_{i}",
            "Ciudad": ciudad,
            "Costo de ejecución (en millones de pesos)": costo,
            "# de empleos generados (en miles)": empleo,
            "Latitud": lat,
            "Longitud": lon
        })

    return pd.DataFrame(datos)


# ============================================================================
# Visualización geográfica con Plotly
# ============================================================================
import plotly.express as px

def visualizar(obras):
    """
    Genera un mapa interactivo de las obras usando coordenadas geográficas.
    
    Parámetros
    ----------
    obras : pd.DataFrame
        DataFrame generado por `generar_datos` u otro con igual estructura.
    """
    fig = px.scatter_mapbox(
        obras,
        lat="Latitud",
        lon="Longitud",
        size="# de empleos generados (en miles)",
        color="Costo de ejecución (en millones de pesos)",
        hover_data={
            "Ciudad": True,
            "Costo de ejecución (en millones de pesos)": True,
            "# de empleos generados (en miles)": True,
            "Latitud": False,
            "Longitud": False
        },
        color_continuous_scale="YlOrRd",
        size_max=40,
        zoom=10,
        height=700,
        width=1500,
        labels={
            "Costo de ejecución (en millones de pesos)": "Costo ejecución (M)",
            "# de empleos generados (en miles)": "Empleos generados (mil)"
        }
    )

    # Usar un estilo plano y centrar el mapa en Colombia
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_center={"lat": 4.7, "lon": -74.1}
    )
    fig.show()


# ============================================================================
# Reporte comparativo de obras seleccionadas
# ============================================================================
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def report(obras, obras_selec, fo, presupuesto_usado, p):
    """
    Genera un gráfico de barras y un indicador de presupuesto.
    
    Parámetros
    ----------
    obras : pd.DataFrame
        DataFrame con todas las obras.
    obras_selec : list
        Lista de índices (enteros) de las obras seleccionadas.
    fo : float
        Valor de la función objetivo (ej. empleos generados).
    presupuesto_usado : float
        Suma de costos de las obras seleccionadas.
    p : float
        Presupuesto máximo permitido.
    """
    # Clonar DataFrame para no afectar el original
    obras = obras.copy()

    # Etiquetar obras seleccionadas
    obras["Seleccionada"] = obras["Obra"].isin([f"Obra_{i}" for i in obras_selec]) 
    obras["Orden"] = obras["Obra"].str.extract(r"(\d+)").astype(int)
    obras = obras.sort_values("Orden")

    df_sel = obras[obras["Seleccionada"]]
    df_nosel = obras[~obras["Seleccionada"]]

    # Crear subplots: gráfico de barras + gauge indicador
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.5, 0.5],
        subplot_titles=(f"Empleos generados<br><sup>FO: {fo} | Obras: {len(obras_selec)}</sup>", None),
        specs=[[{"type": "xy"}, {"type": "domain"}]]
    )

    # Barras de no seleccionadas (gris)
    fig.add_trace(
        go.Bar(
            x=df_nosel["Obra"],
            y=df_nosel["# de empleos generados (en miles)"],
            marker=dict(color="lightgray"),
            hovertemplate="<b>%{x}</b><br>Empleos: %{y}<br>Costo: %{customdata} millones",
            customdata=df_nosel["Costo de ejecución (en millones de pesos)"]
        ), row=1, col=1
    )

    # Barras de seleccionadas (azul)
    fig.add_trace(
        go.Bar(
            x=df_sel["Obra"],
            y=df_sel["# de empleos generados (en miles)"],
            marker=dict(color="steelblue"),
            hovertemplate="<b>%{x}</b><br>Empleos: %{y}<br>Costo: %{customdata} millones",
            customdata=df_sel["Costo de ejecución (en millones de pesos)"]
        ), row=1, col=1
    )

    # Indicador de presupuesto
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=presupuesto_usado,
            number={"suffix": f" / {p}", "font": {"size": 40}},
            gauge={"axis": {"range": [0, p]}}
        ), row=1, col=2
    )

    # Ajuste de estilo general
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            showgrid=False,
            categoryorder="array",
            categoryarray=obras["Obra"]
        ),
        yaxis=dict(showgrid=False),
        barmode="overlay",
        showlegend=False
    )

    fig.show()
