"""
Este archivo agrupa las funciones utilizadas para realizar el análisis exploratorio de datos
"""


# imports

# intern
import warnings

# extern
import numpy as np
import pandas as pd

# custom


warnings.filterwarnings("ignore")


def check_date(d: pd.Timestamp)->bool:
    """
    Checkea si una fecha está dentro del rango aceptado para ser catalogada como fecha
    :param d: fecha
    :return: True si es un fecha válida False si no
    """
    try:
        pd.Timestamp(year=int(d[0:4]), month=int(d[5:7]), day=int(d[8:11]))
        return True
    except:
        return False

# versión vectorial de la función check_date
vec_check_date = np.vectorize(check_date)


def del_fake_dates(df: pd.DataFrame, date_col: str)->pd.DataFrame:
    """
    Utiliza check_date para eliminar filas con fechas no válidas
    :param df: DataFrame a limpiar
    :param date_col: columna de fechas
    :return: DataFrame limpio
    """

    df_new = df.copy()

    prev_shape = df_new.shape

    # eliminando fechas inválidas
    df_new = df_new[vec_check_date(df_new[date_col])]

    # seteando fechas válidas como index y ordenando datos
    df_new[date_col] = pd.to_datetime(df_new[date_col].apply(lambda row: str(row)[:10]))
    df_new = df_new.set_index(date_col)
    df_new = df_new.sort_index()

    post_shape = df_new.shape

    if prev_shape == post_shape:
        print("Ningun dato con fecha inexacta")
    else:
        print(f"{prev_shape[0] - post_shape[0]} con fecha inexacta fueron eliminados")

    return df_new

def missing_values_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Counts and calculates null values per column
    :param df: features's DataFrame
    :return: DataFrame resumen de valores nulos
    """
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Hay " + str(df.shape[1]) + " columnas.\n"
                                      "Hay " + str(mis_val_table_ren_columns.shape[0])
          + " columnas con valores nulos")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns


def drop_duplicates(df:pd.DataFrame, date_col: str)->pd.DataFrame:
    """
    Elimina filas repetidas. En caso que solo en la columna de fechas se encuentren valores repetidos,
    se procede a calcular el promedio entre los valores.
    :param df: dataframe de valores
    :param date_col: columna de fechas
    :return: dataframe limpio
    """
    df_new = df.copy()

    # fechas duplicados
    cant_ddup = df_new.reset_index()[date_col].shape[0] - df_new.reset_index()[date_col].drop_duplicates().shape[0]
    if cant_ddup > 0:
        print(f"""Hay {cant_ddup} fechas duplicadas""")
        df_new_ddup = df_new.reset_index()[df_new.reset_index()[date_col].duplicated()].set_index(date_col)

        if len(df_new_ddup.drop_duplicates()) == cant_ddup:
            print(f"Filas completas repetidas. Se procede a eliminar 1")
            df_new = df_new.drop_duplicates()

        else:
            # tomamos el promedio entre los valores
            print(f"Tomando promedio entre valores de fechas repetidas")
            df_new_ddup = df_new[df_new.index.isin(df_new_ddup.index)].reset_index().groupby(date_col).mean()
            df_new = df_new[~df_new.index.isin(df_new_ddup.index)]
            df_new = df_new.append(df_new_ddup)
            df_new = df_new.sort_index()


    else:
        print("DataFrame sin fechas duplicadas")

    return df_new


