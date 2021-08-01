"""
Este archivo agrupa las funciones utilizadas para realizar la visualización de los datos

"""


# import
# intern
from typing import List
import warnings

# extern
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score

# custom
from data_preprocessing import nombre_mes


warnings.filterwarnings("ignore")

def plot_precipitaciones(region: str, start_date: pd.Timestamp, end_date: pd.Timestamp, prep_df: pd.DataFrame):
    """
    permita graficar series históricas de precipitaciones para un rango de fechas determinado

    :param region: región a graficar
    :param start_date: fecha de inicio
    :param end_date: fecha de término
    :param prep_df: Conjunto de datos de precipitaciones históricas por región
    :return:
    """
    # fecha mínimo y máxima en el conjunto de datos
    min_date = np.min(prep_df.index)
    max_date = np.max(prep_df.index)

    # checkeando si la region se encuentra en el DataFrame():
    region_in = region in prep_df.columns

    if region_in:

        # no se gráfica nada
        if start_date > end_date:
            print(f"Fecha de inicio: {start_date} superior a fecha de termino {end_date}")

        # no se gráfica nada
        elif end_date < min_date or start_date > max_date:
            print(f"Rango de fechas fuera del rango de fechas del conjunto de datos: {min_date}-{max_date}")

        # se gráfica
        else:
            if end_date >= min_date and start_date < min_date:
                print(
                    f"Fecha de inicio fuera del rango de fechas del conjunto de datos. Se utiliza {min_date} como fecha de partida")
                start_date = min_date

            if min_date <= start_date <= max_date and min_date <= end_date <= max_date:
                print(f"Fechas dentro del rango de fechas del conjunto de datos")

            if end_date > max_date and start_date <= max_date:
                print(
                    f"Fecha de termino fuera del rango de fechas del conjunto. Se utiliza {max_date} como fecha de termino")
                end_date = max_date

            start_date = pd.Timestamp(year=start_date.year, month=start_date.month, day=1)
            end_date = pd.Timestamp(year=end_date.year, month=end_date.month + 1,
                                    day=1) if end_date.day > 1 else pd.Timestamp(year=end_date.year,
                                                                                 month=end_date.month, day=1)

            df = prep_df[(prep_df.index >= start_date) & (prep_df.index <= end_date)][[region]]

            df.plot(figsize=(20, 20))
            plt.ylabel("Precipitaciones [mm]")
            plt.title(region)
            plt.xlabel('')
            plt.xticks(rotation=45)

            plt.show()

def plot_precipitaciones_mensuales(region:str, years:List[int], prep_df: pd.DataFrame):
    """
    Grafica múltiples series de tiempo mensuales de precipitaciones, donde cada serie de tiempo corresponda a un año
    :param region: región a graficas
    :param years: años a graficar
    :param prep_df: conjunto de datos históricos de precipitaciones por región
    :return:
    """

    # checkeando si la region se encuentra en el DataFrame():
    region_in = region in prep_df.columns

    if region_in:

        df = prep_df[[region]]
        df["Año"] = prep_df.index.year.astype(int)
        df["Mes_num"] = prep_df.index.month.astype(int)
        df["Mes"] = df["Mes_num"].apply(nombre_mes)

        excluded_years = [y for y in years if y not in df["Año"].values]
        included_years = [y for y in years if y not in excluded_years]

        if len(excluded_years) > 0:
            print(f"Los siguientes años no figuran en el conjunto de datos: {excluded_years}")

        if len(included_years) > 0:
            df = df[df["Año"].isin(included_years)]

            orden_meses = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

            df = df.reset_index().pivot(index="Mes", columns="Año", values="Maule").reindex(orden_meses)

            # plotting
            df.plot(figsize=(20, 20))
            plt.ylabel("Precipitaciones [mm]")
            plt.xlabel('')
            plt.xticks(rotation=45)
            plt.show()



def plot_pibs(pib_names: List[str], start_date: pd.Timestamp, end_date: pd.Timestamp, bc_df: pd.DataFrame):
    """
    Permite visualizar dos series históricas de PIB para un rango de fechas determinado
    :param pib_names: lista de PIBs a graficar
    :param start_date: fecha de inicio
    :param end_date: fecha de término
    :param bc_df: conjunto de datos históricos de indicadores económicos
    :return:
    """

    # fecha mínimo y máxima en el conjunto de datos
    min_date = np.min(bc_df.index)
    max_date = np.max(bc_df.index)

    # checkeando si  pib_names se encuentra en el DataFrame():

    pib_names_in = [pib_name for pib_name in pib_names if pib_name in bc_df.columns]
    pib_names_out = [pib_name for pib_name in pib_names if pib_name not in pib_names_in]

    if len(pib_names_out) > 0:
        print(f"{pib_names_out}: No se reconocen como PIB en los indicadores económicos")

    if len(pib_names_in):

        # no se gráfica nada
        if start_date > end_date:
            print(f"Fecha de inicio: {start_date} superior a fecha de termino {end_date}")

        # no se gráfica nada
        elif end_date < min_date or start_date > max_date:
            print(f"Rango de fechas fuera del rango de fechas del conjunto de datos: {min_date}-{max_date}")

        # se gráfica
        else:
            if end_date >= min_date and start_date < min_date:
                print(
                    f"Fecha de inicio fuera del rango de fechas del conjunto de datos. Se utiliza {min_date} como fecha de partida")
                start_date = min_date

            if min_date <= start_date <= max_date and min_date <= end_date <= max_date:
                print(f"Fechas dentro del rango de fechas del conjunto de datos")

            if end_date > max_date and start_date <= max_date:
                print(
                    f"Fecha de termino fuera del rango de fechas del conjunto. Se utiliza {max_date} como fecha de termino")
                end_date = max_date

            start_date = pd.Timestamp(year=start_date.year, month=start_date.month, day=1)
            end_date = pd.Timestamp(year=end_date.year, month=end_date.month + 1,
                                    day=1) if end_date.day > 1 else pd.Timestamp(year=end_date.year,
                                                                                 month=end_date.month, day=1)

            df = bc_df[(bc_df.index >= start_date) & (bc_df.index <= end_date)][pib_names_in]

            # plotting
            df.plot(figsize=(20, 20))
            plt.ylabel('')
            plt.xlabel('')
            plt.xticks(rotation=45)

            plt.show()


def distribution_plots(df: pd.DataFrame, kind:str="kde", figsize:tuple=(20, 20), title:str="", norm:bool=True):
    """
    Grafica la distribución de los datos

    :param df: conjunto de datos a grafica
    :param kind: tipo de distribución
    :param figsize: tamaño de la imágen
    :param title: título de la imágen
    :param norm: si se normalizan los datos antes de graficar
    :return:
    """

    df_aux = df.copy()

    if norm:
        df_aux = (df_aux - df_aux.mean()) / (df_aux.std())

    # plotting
    df_aux.plot(kind=kind, figsize=figsize)
    plt.ylabel('')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()


def corr_heatmap(df, figsize=(50, 50)):
    """
    Grafica un mapa de calor de la matriz de correlación entre las variables en el conjunto de datos
    :param df: conjunto de datos históricos
    :param figsize: tamaño de la imágen
    :return:
    """
    # Matriz de correlación
    corr = df.corr()

    # plotting
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, cmap="GnBu", annot=True)
    plt.xticks(rotation=45)
    plt.show()







def plot_feature_importance(feature_names, feature_importance, n=10):
    df = pd.DataFrame({"importance": feature_importance}, index=feature_names).sort_values("importance",
                                                                                           ascending=False)

    df.iloc[:n].sort_values("importance", ascending=True).plot(kind='barh', title='Feature importance',
                                                               figsize=(10, 10))

    plt.show()

    return df


def plot_accuracy(y, ypred):
    plt.plot(y, y, color="blue", linestyle='--')
    plt.scatter(y, ypred, color="red")
    plt.ylabel("Valores predecidos")
    plt.xlabel("Valores reales")
    plt.title(f"R2: {r2_score(y, ypred)}")
    plt.show()


def plot_target_and_pred(y, y_pred):
    if not isinstance(y, pd.DataFrame):
        y = y.to_frame()

    y_pred = pd.DataFrame(y_pred, index=y.index, columns=["Predicción"])
    res_y = pd.concat([y, y_pred], axis=1)
    res_y.plot(figsize=(20, 20))
    plt.show()
