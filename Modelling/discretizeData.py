# =============================================================
# Nombre del archivo: discretizeData.py
# Autor: Tomás Pérez Vera
# Fecha de creación: 22-10-2025
# Descripción: Function para discretizar salida del modelo
# Dependencias: Ninguna
# =============================================================

def discretize_value(bcsPred: float) -> float:
    """
    Función para redondear al múltiplo de 0.25 más cercano 

    Args: 
        value (float):  Valor de punto flotante

    Return: 
        _ (float): Valor de punto flotante discretizado

    """

    if value > 5.0: 
        value = 5.0
    elif value < 1.0: 
        value = 1.0

    return round(value * 4) / 4