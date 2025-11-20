"""
Funciones para añadir features de calendario a los datos de una celda.

Trabajan sobre dataframes que tienen, como mínimo:
    - una columna 'datetime' (tipo datetime64[ns])
    - una columna 'internet_total' (valor de tráfico)

Las features que se añaden son:
    - day_of_week      : 0 (lunes) ... 6 (domingo)
    - hour_of_day      : 0 ... 23
    - is_public_holiday: 1 si es festivo oficial en Italia (provincia de Milán),
                         0 en caso contrario
    - is_special_break : 1 si está dentro de algún periodo especial definido
                         en un CSV externo (por ejemplo, vacaciones de Navidad)
"""

from pathlib import Path

import pandas as pd
import holidays


def add_public_holiday_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade la columna is_public_holiday (0/1) usando la librería 'holidays'.

    Se obtienen los festivos oficiales para la provincia de Milán (IT, subdiv=MI)
    para todos los años que aparecen en la columna 'datetime'. Esto incluye tanto
    los festivos nacionales como los específicos de la provincia.
    """
    # Conjunto de años presentes en el dataframe
    years = sorted({ts.year for ts in df["datetime"]})

    # Festivos de Italia para la provincia de Milán en esos años
    it_holidays = holidays.country_holidays("IT", subdiv="MI", years=years)
    holiday_dates = set(it_holidays.keys())

    # Comparamos solo la parte de fecha (sin hora)
    dates = df["datetime"].dt.date
    df["is_public_holiday"] = dates.isin(holiday_dates).astype(int)

    return df


def add_special_break_feature(
    df: pd.DataFrame,
    periods_path: Path,
    column_name: str = "is_special_break",
) -> pd.DataFrame:
    """
    Añade una columna (por defecto 'is_special_break') a partir de un fichero
    CSV con periodos especiales.

    El CSV debe tener columnas:
        - 'label'      : nombre del periodo (no se usa en el cálculo)
        - 'date_start' : fecha de inicio (YYYY-MM-DD)
        - 'date_end'   : fecha de fin   (YYYY-MM-DD)

    Cada fila define un rango [date_start, date_end] incluido.
    Si un timestamp cae dentro de cualquiera de esos rangos, la columna
    toma valor 1; en caso contrario, 0.
    """
    periods = pd.read_csv(
        periods_path,
        parse_dates=["date_start", "date_end"],
    )

    df[column_name] = 0
    dates = df["datetime"].dt.date

    for _, row in periods.iterrows():
        start = row["date_start"].date()
        end = row["date_end"].date()
        mask = (dates >= start) & (dates <= end)
        df.loc[mask, column_name] = 1

    return df


def add_calendar_features(
    df: pd.DataFrame,
    special_periods_path: Path | None = None,
) -> pd.DataFrame:
    """
    Añade todas las features de calendario usadas en el proyecto.

    Actualmente:
        - day_of_week       : 0 (lunes) ... 6 (domingo)
        - hour_of_day       : 0 ... 23
        - is_public_holiday : 1 si es festivo oficial en Italia (provincia de Milán)
        - is_special_break  : 1 si está dentro de un periodo especial definido
                              en un CSV (si se proporciona special_periods_path)

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con al menos una columna 'datetime'.

    special_periods_path : Path o None
        Ruta al CSV de periodos especiales (por ejemplo,
        Data/calendar/special_periods.csv). Si es None, no se añade
        la columna is_special_break.

    Devuelve
    --------
    df : pd.DataFrame
        El mismo dataframe de entrada, con columnas nuevas añadidas.
    """
    # Features simples derivadas del timestamp
    df["day_of_week"] = df["datetime"].dt.dayofweek  # 0=lunes, ..., 6=domingo
    df["hour_of_day"] = df["datetime"].dt.hour       # 0, 1, ..., 23

    # Festivos oficiales en la provincia de Milán
    df = add_public_holiday_feature(df)

    # Periodos especiales definidos por el usuario (opcional)
    if special_periods_path is not None:
        df = add_special_break_feature(df, periods_path=special_periods_path)

    return df
