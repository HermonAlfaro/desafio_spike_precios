"""
Este archivo contiene las clases y funciones que encapsulan los métodos para tratar los datos de entrada de indicadores
económicos.

Los datos del banco central vienen mal formateados de origen y es súper díficil saber la escala en la que se encuentran,
incluso entre datos de la misma columna:

ex:
en la columna Tipo_de_cambio_del_dolar_observado_diario podemos observar valores de la forma: 519.25, pero también
de la forma: 5.294.505.

Es fácil suponer que en dicho caso ambos valores son 519,25 y 529,4505 respectivamente. Sin embargo, no es trivial saber
cómo formatear los datos para las más de 80 columnas.

Por lo tanto, en virtud del tiempo, he decidido hacer uso del análisis realizado por Ana María Castillo Núñez,
actual miembro de Spike que participó en el desafío de Abril. (https://github.com/anacastillonu/desafio_spike_precios).

Las clases aquí escritas están basadas en el tratamiento que Ana María realizo mientras participaba en el desafío.
Por lo tanto, la mayor parte del crédito se le debe a ella.


Todas las clases tienen la misma estructura:

- Un método que trata el dato de manera individual
- Un método que vectoriza el método anterior para luego ser aplicado a un conjunto de datos
"""


# imports

# intern
import re
import warnings


# extern
import numpy as np
import pandas as pd


# custom



warnings.filterwarnings("ignore")


class ImacecProcessor(object):

    @staticmethod
    def _imacec_processing(imacec_str):

        # para lidiar con valores np.nan
        imacec_cleaned = str(imacec_str)

        # remplazar valores no números por ''
        imacec_cleaned = re.sub("[^0-9]", '', imacec_cleaned)

        # si el valor es efectivamente númerico
        if imacec_cleaned.isnumeric():

            # los valores que empiezan por 1 son 10 veces más grandes, es decir añadimos un cero al final de la cadena
            lenght = 10 if imacec_cleaned[0] == '1' else 9
            imacec_cleaned = imacec_cleaned.ljust(lenght, '0')

            # formato númerico
            imacec_num = int(imacec_cleaned)

            # dejamos el valor en millones
            imacec_num /= (10 ** 6)

        # si el valor no es númerico, lo dejamos como NaN
        else:
            imacec_num = np.nan

        return imacec_num

    def imacec_processing(self, imacec_list):

        vec_proc = np.vectorize(self._imacec_processing)

        return vec_proc(imacec_list)


class PIBProcessor(object):
    pib_ord_mag = {'PIB_Agropecuario_silvicola': 3,
                   'PIB_Pesca': 2,
                   'PIB_Mineria': 4,
                   'PIB_Mineria_del_cobre': 4,
                   'PIB_Otras_actividades_mineras': 3,
                   'PIB_Industria_Manufacturera': 4,
                   'PIB_Alimentos': 3,
                   'PIB_Bebidas_y_tabaco': 3,
                   'PIB_Textil': 2,
                   'PIB_Maderas_y_muebles': 2,
                   'PIB_Celulosa': 2,
                   'PIB_Refinacion_de_petroleo': 2,
                   'PIB_Quimica': 3,
                   'PIB_Minerales_no_metalicos_y_metalica_basica': 2,
                   'PIB_Productos_metalicos': 3,
                   'PIB_Electricidad': 3,
                   'PIB_Construccion': 3,
                   'PIB_Comercio': 4,
                   'PIB_Restaurantes_y_hoteles': 3,
                   'PIB_Transporte': 3,
                   'PIB_Comunicaciones': 3,
                   'PIB_Servicios_financieros': 3,
                   'PIB_Servicios_empresariales': 4,
                   'PIB_Servicios_de_vivienda': 3,
                   'PIB_Servicios_personales': 4,
                   'PIB_Administracion_publica': 3,
                   'PIB_a_costo_de_factores': 5,
                   'Impuesto_al_valor_agregado': 4,
                   'Derechos_de_Importacion': 2,
                   'PIB': 5}

    def _pib_processing(self, pib_str, pib_name):

        # para lidiar con valores np.nan
        pib_cleaned = str(pib_str)

        # remplazar valores no números por ''
        pib_cleaned = re.sub("[^0-9]", '', pib_cleaned)

        # si el valor es efectivamente númerico
        if pib_cleaned.isnumeric():

            # normalizamos para tener siempre 9 cifras
            pib_cleaned = pib_cleaned.ljust(9, '0')

            # formato númerico
            pib_num = int(pib_cleaned)

            # dejamos el valor en miles millones
            escala = 10 - int(bool(int(pib_cleaned[0]) < 8)) - self.pib_ord_mag[pib_name]
            pib_num /= (10 ** escala)


        # si el valor no es númerico, lo dejamos como NaN
        else:
            pib_num = np.nan

        return pib_num

    def pib_processing(self, pib_list, pib_name):

        vec_proc = np.vectorize(self._pib_processing)

        return vec_proc(pib_list, pib_name)


class GasolinaProcessor(object):
    limit_year = 2000

    def _gasolina_processing(self, gasolina_tuple):

        # gasolina_tuple: (date, value)
        date = gasolina_tuple[0]
        gasolina_str = gasolina_tuple[1]

        # para lidiar con valores np.nan
        gasolina_cleaned = str(gasolina_str)

        # remplazar valores no números por ''
        gasolina_cleaned = re.sub("[^0-9]", '', gasolina_cleaned)

        # si el valor es efectivamente númerico
        if gasolina_cleaned.isnumeric():

            if date.year >= self.limit_year:

                # normalizamos para tener siempre 9 cifras
                gasolina_cleaned = gasolina_cleaned.ljust(9, '0')

                # formato númerico
                gasolina_num = float(gasolina_cleaned)

                # dejamos el valor en miles millones
                escala = 6
                gasolina_num /= (10 ** escala)

            else:
                gasolina_num = float(gasolina_str)

        # si el valor no es númerico, lo dejamos como NaN
        else:
            gasolina_num = np.nan

        return gasolina_num

    def gasolina_processing(self, gasolina_tuple_list):

        # gasolina_tuple_list: [(date1, value1), (date2, value2),...]

        return np.apply_along_axis(self._gasolina_processing, 1, gasolina_tuple_list)


class OnzaPlataProcessor(object):

    @staticmethod
    def _onza_plata_processing(onza):
        onza_cleaned = onza / 10 if onza >= 100 else onza

        return onza_cleaned

    def onza_plata_processing(self, onza_list):
        vec_proc = np.vectorize(self._onza_plata_processing)

        return vec_proc(onza_list)


class CobreProcessor(object):

    @staticmethod
    def _cobre_processing(cobre_str):

        # para lidiar con valores np.nan
        cobre_cleaned = str(cobre_str)

        # remplazar valores no números por ''
        cobre_cleaned = re.sub("[^0-9]", '', cobre_cleaned)

        # si el valor es efectivamente númerico
        if cobre_cleaned.isnumeric():

            # borramos ceros de la derecha si es necesario y completamos para tener un largo igual a 9
            cobre_cleaned = cobre_cleaned[1:].ljust(9, '0') if cobre_cleaned[0] == '0' else cobre_cleaned.ljust(9, '0')

            # formato númerico
            cobre_num = float(cobre_cleaned)

            # dejamos el valor en miles millones
            escala = 9 if int(cobre_cleaned[:3]) >= 456 else 8
            cobre_num /= (10 ** escala)


        # si el valor no es númerico, lo dejamos como NaN
        else:
            cobre_num = np.nan

        return cobre_num

    def cobre_processing(self, cobre_list):

        vec_proc = np.vectorize(self._cobre_processing)

        return vec_proc(cobre_list)


class GasNaturalProcessor(object):

    @staticmethod
    def _gas_natural_processing(gas_natural):
        gas_natural_cleaned = gas_natural / 10 if gas_natural >= 100 else gas_natural

        return gas_natural_cleaned

    def gas_natural_processing(self, gas_natural_list):
        vec_proc = np.vectorize(self._gas_natural_processing)

        return vec_proc(gas_natural_list)


class KerosenoProcessor(object):

    @staticmethod
    def _keroseno_processing(keroseno_str):

        # para lidiar con valores np.nan
        keroseno_cleaned = str(keroseno_str)

        # remplazar valores no números por ''
        keroseno_cleaned = re.sub("[^0-9]", '', keroseno_cleaned)

        # si el valor es efectivamente númerico
        if keroseno_cleaned.isnumeric():

            # completamos para tener un largo igual a 9
            keroseno_cleaned = re.sub("[^0-9]", "", keroseno_cleaned).ljust(9, '0')

            # formato númerico
            keroseno_num = float(keroseno_cleaned)

            # dejamos el valor en millones
            escala = 6
            keroseno_num /= (10 ** escala)


        # si el valor no es númerico, lo dejamos como NaN
        else:
            keroseno_num = np.nan

        return keroseno_num

    def keroseno_processing(self, keroseno_list):

        vec_proc = np.vectorize(self._keroseno_processing)

        return vec_proc(keroseno_list)


class TasaObservadaProcessor(object):

    @staticmethod
    def _tasa_observada_processing(tasa_observada_str):

        # para lidiar con valores np.nan
        tasa_observada_cleaned = str(tasa_observada_str)

        # remplazar valores no números por ''
        tasa_observada_cleaned = re.sub("[^0-9]", '', tasa_observada_cleaned)

        # si el valor es efectivamente númerico
        if tasa_observada_cleaned.isnumeric():

            # completamos para tener un largo igual a 9
            tasa_observada_cleaned = re.sub("[^0-9]", "", tasa_observada_cleaned).ljust(9, '0')

            # formato númerico
            tasa_observada_num = int(tasa_observada_cleaned)

            # dejamos el valor en millones
            escala = 6
            tasa_observada_num /= (10 ** escala)


        # si el valor no es númerico, lo dejamos como NaN
        else:
            tasa_observada_num = np.nan

        return tasa_observada_num

    def tasa_observada_processing(self, tasa_observada_list):

        vec_proc = np.vectorize(self._tasa_observada_processing)

        return vec_proc(tasa_observada_list)


class TasaNominalProcessor(object):

    @staticmethod
    def _tasa_nominal_processing(tasa_nominal_str):

        # para lidiar con valores np.nan
        tasa_nominal_cleaned = str(tasa_nominal_str)

        # remplazar valores no números por ''
        tasa_nominal_cleaned = re.sub("[^0-9]", '', tasa_nominal_cleaned)

        # si el valor es efectivamente númerico
        if tasa_nominal_cleaned.isnumeric():

            # completamos para tener un largo igual a 9
            tasa_nominal_cleaned = re.sub("[^0-9]", "", tasa_nominal_cleaned).ljust(9, '0')

            # formato númerico
            tasa_nominal_num = int(tasa_nominal_cleaned)

            # dejamos el valor en millones
            escala = 7 if tasa_nominal_cleaned[0] == '9' else 6
            tasa_nominal_num /= (10 ** escala)


        # si el valor no es númerico, lo dejamos como NaN
        else:
            tasa_nominal_num = np.nan

        return tasa_nominal_num

    def tasa_nominal_processing(self, tasa_nominal_list):

        vec_proc = np.vectorize(self._tasa_nominal_processing)

        return vec_proc(tasa_nominal_list)


class OcupacionProcessor(object):
    ocu_ord_mag = {'Ocupados': 4,
                   'Ocupacion_en_Agricultura_INE': 3,
                   'Ocupacion_en_Explotacion_de_minas_y_canteras_INE': 3,
                   'Ocupacion_en_Industrias_manufactureras_INE': 3,
                   'Ocupacion_en_Suministro_de_electricidad_INE': 2,
                   'Ocupacion_en_Actividades_de_servicios_administrativos_y_de_apoyo_INE': 3,
                   'Ocupacion_en_Actividades_profesionales_INE': 3,
                   'Ocupacion_en_Actividades_inmobiliarias_INE': 2,
                   'Ocupacion_en_Actividades_financieras_y_de_seguros_INE': 3,
                   'Ocupacion_en_Informacion_y_comunicaciones_INE': 3,
                   'Ocupacion_en_Transporte_y_almacenamiento_INE': 3,
                   'Ocupacion_en_Actividades_de_alojamiento_y_de_servicio_de_comidas_INE': 3,
                   'Ocupacion_en_Construccion_INE': 3,
                   'Ocupacion_en_Comercio_INE': 4,
                   'Ocupacion_en_Suministro_de_agua_evacuacion_de_aguas_residuales_INE': 2,
                   'Ocupacion_en_Administracion_publica_y_defensa_INE': 3,
                   'Ocupacion_en_Enseanza_INE': 3,
                   'Ocupacion_en_Actividades_de_atencion_de_la_salud_humana_y_de_asistencia_social_INE': 3,
                   'Ocupacion_en_Actividades_artisticas_INE': 2,
                   'Ocupacion_en_Otras_actividades_de_servicios_INE': 3,
                   'Ocupacion_en_Actividades_de_los_hogares_como_empleadores_INE': 3,
                   'Ocupacion_en_Actividades_de_organizaciones_y_organos_extraterritoriales_INE': 1,
                   'No_sabe__No_responde_Miles_de_personas': 2
                   }

    def _ocupacion_processing(self, ocu_str, ocu_name):

        # para lidiar con valores np.nan
        ocu_cleaned = str(ocu_str)

        # remplazar valores no números por ''
        ocu_cleaned = re.sub("[^0-9]", '', ocu_cleaned)

        # si el valor es efectivamente númerico
        if ocu_cleaned.isnumeric():

            # normalizamos para tener siempre 9 cifras
            ocu_cleaned = ocu_cleaned.ljust(9, '0')

            # formato númerico
            ocu_num = int(ocu_cleaned)

            # dejamos el valor en miles millones
            escala = 9 - self.ocu_ord_mag[ocu_name]
            ocu_num /= (10 ** escala)


        # si el valor no es númerico, lo dejamos como NaN
        else:
            ocu_num = np.nan

        return ocu_num

    def ocupacion_processing(self, ocu_list, ocu_name):

        vec_proc = np.vectorize(self._ocupacion_processing)

        return vec_proc(ocu_list, ocu_name)


class IndiceProcessor(object):

    @staticmethod
    def _indice_processing(indice_str):

        # para lidiar con valores np.nan
        indice_cleaned = str(indice_str)

        # remplazar valores no números por ''
        indice_cleaned = re.sub("[^0-9]", '', indice_cleaned)

        # si el valor es efectivamente númerico
        if indice_cleaned.isnumeric():

            # los valores que empiezan por 1 son 10 veces más grandes, es decir añadimos un cero al final de la cadena
            lenght = 10 if indice_cleaned[0] == '1' else 9
            indice_cleaned = indice_cleaned.ljust(lenght, '0')

            # formato númerico
            indice_num = int(indice_cleaned)

            # dejamos el valor en millones
            indice_num /= (10 ** 6)

        # si el valor no es númerico, lo dejamos como NaN
        else:
            indice_num = np.nan

        return indice_num

    def indice_processing(self, indice_list):

        vec_proc = np.vectorize(self._indice_processing)

        return vec_proc(indice_list)


class EnergiaProcessor(object):

    @staticmethod
    def _energia_processing(energia_str):

        # para lidiar con valores np.nan
        energia_cleaned = str(energia_str)

        # remplazar valores no números por ''
        energia_cleaned = re.sub("[^0-9]", '', energia_cleaned)

        # si el valor es efectivamente númerico
        if energia_cleaned.isnumeric():

            # completamos para tener un largo igual a 9
            energia_cleaned = re.sub("[^0-9]", "", energia_cleaned).ljust(9, '0')

            # formato númerico
            energia_num = int(energia_cleaned)

            # dejamos el valor en millones
            escala = 5
            energia_num /= (10 ** escala)


        # si el valor no es númerico, lo dejamos como NaN
        else:
            energia_num = np.nan

        return energia_num

    def energia_processing(self, energia_list):

        vec_proc = np.vectorize(self._energia_processing)

        return vec_proc(energia_list)


def nombre_mes(mes_num, reversed=False):
    """
    Dado un número entre el 1 y 12 entrega el préfijo del mes correspondiente.
    En caso de reversed=True, entrega el número del mes dado el préfijo

    :param mes_num: número del mes o prefijo del mes
    :param reversed: Si es True entrega el número, en caso de ser False entrega el prefijo
    :return: prefijo del mes o número del mes
    """
    meses_dic = {
        1: "Ene",
        2: "Feb",
        3: "Mar",
        4: "Abr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Ago",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        12: "Dic"
    }

    meses_dic_rev = {v: k for k, v in meses_dic.items()}

    if reversed:
        return meses_dic_rev[mes_num]

    return meses_dic[mes_num]

def next_month(date, delta=1):
    """
    Data una fecha, entrega la misma fecha delta meses después
    :param date: fecha
    :param delta: suma de meses
    :return: fecha
    """
    m, y = (date.month+delta) % 12, date.year + ((date.month)+delta-1) // 12
    if not m: m = 12

    d = min(date.day, [31,
        29 if y%4==0 and (not y%100==0 or y%400 == 0) else 28,
        31,30,31,30,31,31,30,31,30,31][m-1])

    return pd.Timestamp(year=y, month=m, day=1)

