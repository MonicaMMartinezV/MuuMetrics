# =============================================================
# Nombre del archivo: range.py
# Autor: Ulises Orlando Carrizalez Lerín
# Fecha de creación: 11-11-2025
# Descripción: Funciones para sacar rango normal y semaforo
# Dependencias: Funcion de discrteisacion
# =============================================================

from discretizeData import discretize_value

def NormalRange(DEL):
    """
    Calcular los rangos de maximo y minimo de BCS en base al DEL

    Args:
        DEL (int): Dias en leche

    Returns:	
        Max (float): BCS maximo para que no sea un riesgo
        Min (float): BCS minimo para que no sea un riesgo
    """

    if DEL <= 288 and DEL >= 0:
        Max= discretize_value(-1e-8*DEL**3 + 3e-5 * DEL**2 - 0.0079*DEL + 3.2665)
        Min = Max - 0.5
    elif DEL > 288 and DEL <= 500:
        Max = 3.25
        Min = 2.25
    else:
        raise ValueError("el numero esta arriba de 500 dias o menos que 0")
    return Max, Min

def Semaforo(BCS,DEL):
    """
    Devolver el color del semaforo

    Args:
        BCS (float): BCS de la vaca actual
        DEL (int): Dias en leche actual

    Returns:	
        Semaforo (str): Color de semaforo
    """
    Max,Min = NormalRange(DEL)
    #print("Max: ",Max)
    #print("Min: ",Min)
    #print("BCS: ",BCS)
    if BCS <= Max and BCS >= Min:
        return "Verde"
    elif BCS <= Max + 0.25 and BCS >= Min - 0.25:
        return "Yellow"
    else: 
        return "Red"