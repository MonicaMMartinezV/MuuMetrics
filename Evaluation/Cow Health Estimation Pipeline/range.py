# =============================================================
# Nombre del archivo: range.py
# Autor: Ulises Orlando Carrizalez Lerín
# Fecha de creación: 11-11-2025
# Descripción: Funciones para sacar rango normal y semaforo
# Dependencias: Funcion de discretize_value
# =============================================================

from discretizeData import discretize_value
import numpy  as np

x_key = np.array([0, 50, 100, 150, 200, 250, 300, 350, 400])
y_key = np.array([3.10, 2.85, 2.55, 2.25, 2.55, 2.75, 2.85, 2.95, 3.10]) - 0.07

# Polynomial coefficients (from polyfit)
coef = np.array([
    -1.63170163e-10,
    6.92566693e-08,
    1.94599845e-05,
    -8.59893810e-03,
    3.06663559-0.06663559000000019
])

rangeFunction = np.poly1d(coef)

def NormalRange(DEL):
    """
    Calcular los rangos de maximo y minimo de BCS en base al DEL

    Args:
        DEL (int): Dias en leche

    Returns:	
        Max (float): BCS maximo para que no sea un riesgo
        Min (float): BCS minimo para que no sea un riesgo
    """

    if DEL <= 400 and DEL >= 0:
        Max = discretize_value(rangeFunction(DEL)  + 0.25)
        Min = discretize_value(rangeFunction(DEL) - 0.25)
    elif DEL > 400 and DEL <= 500:
        Max = 3.25
        Min = 2.75
    else:
        raise ValueError(f"el numero esta arriba de 500 dias o menos que 0 Valor:{DEL}")
    return Max, Min

def Semaforo(BCS,DEL):
    """
    Devolver el color del semaforo

    Args:
        BCS (float): BCS de la vaca actual
        DEL (int): Dias en leche actual

    Returns:	
        _ (str): Color de semaforo
    """
    Max,Min = NormalRange(DEL)
    #print("Max: ",Max)
    #print("Min: ",Min)
    #print("BCS: ",BCS)
    if BCS <= Max and BCS >= Min:
        return "Green"
    elif BCS <= Max + 0.25 and BCS >= Min - 0.25:
        return "Yellow"
    else: 
        return "Red"