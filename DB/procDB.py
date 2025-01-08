import csv

import json


def procesar_csv(nombre_archivo):
    resultados = {}
    proteinas = {}
    ligandos = {}

    with open(nombre_archivo, "r") as archivo_csv:
        lector_csv = csv.DictReader(archivo_csv)
        for fila in lector_csv:
            modelo = fila["graph_id"]
            tipo = fila["type"]
            id_nodo = int(fila["node/source"])

            if modelo not in resultados:
                resultados[modelo] = {}

            if tipo == "node":
                propiedades = {
                    "htype": fila["htype"],
                    "irr": fila["irr"],
                    "sasa": fila["sasa"],
                    "dssp": fila["dssp"],
                    "mol": fila["mol"],
                }
                if fila["score"] != "":
                    ligandos[id_nodo] = propiedades
                    ligandos[id_nodo]["score"] = fila["score"]
                else:
                    proteinas[id_nodo] = propiedades

            elif tipo == "edge":
                proteina = int(fila["node/source"])
                ligando = int(fila["target"])
                distancia = float(fila["dist"])
                if ligando in ligandos and proteina in proteinas:
                    if ligando not in resultados[modelo]:
                        resultados[modelo][ligando] = {}
                        resultados[modelo][ligando]["property"] = ligandos[
                            ligando
                        ].copy()
                        resultados[modelo][ligando]["proteins"] = {}
                    resultados[modelo][ligando]["proteins"][proteina] = proteinas[
                        proteina
                    ].copy()  # Usamos copia para evitar modificar el diccionario original

    return resultados


def pdb_ligando_parser(nombre_archivo_pdb, model_name: str):
    """Parsea los datos de un archivo PDB a un diccionario.

    Args:
        nombre_archivo_pdb: Nombre del archivo PDB de entrada.
    """

    try:
        with open(nombre_archivo_pdb, "r") as archivo_pdb:
            diccionario_pdb = {}
            modelo_actual = 1

            for linea in archivo_pdb:
                if linea.startswith("MODEL"):
                    try:
                        modelo_actual = int(linea.split()[1])
                        diccionario_pdb[f"{model_name}{modelo_actual}"] = (
                            {}
                        )  # Inicializa el dic del modelo
                    except IndexError:
                        print(
                            "Advertencia: Formato de línea MODEL incorrecto. Asumiendo modelo 1."
                        )
                        modelo_actual = 1
                        diccionario_pdb[f"{model_name}{modelo_actual}"] = (
                            {}
                        )  # Inicializa el dic del modelo
                elif linea.startswith("HETATM") or linea.startswith("ATOM"):
                    try:
                        atom_num = int(linea[6:11].strip())
                        x = float(linea[30:38].strip())
                        y = float(linea[38:46].strip())
                        z = float(linea[46:54].strip())

                        if (
                            f"{model_name}{modelo_actual}" not in diccionario_pdb
                        ):  # Por si no hay linea MODEL al principio
                            diccionario_pdb[f"{model_name}{modelo_actual}"] = {}
                        diccionario_pdb[f"{model_name}{modelo_actual}"][atom_num] = {
                            "x": x,
                            "y": y,
                            "z": z,
                        }
                    except (ValueError, IndexError):
                        print(
                            f"Advertencia: Línea HETATM/ATOM con formato incorrecto: {linea.strip()}"
                        )
            return diccionario_pdb

    except FileNotFoundError:
        print(f"Error: Archivo PDB '{nombre_archivo_pdb}' no encontrado.")
        return None
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
        return None


def pdb_proteina_parser(nombre_archivo_pdb):
    """Convierte datos de un archivo PDB a un diccionario, usando el número de átomo como clave.

    Args:
        nombre_archivo_pdb: Nombre del archivo PDB de entrada.

    Returns:
        Un diccionario con la estructura {atom_num: {x, y, z, atom_name, res_name, res_num, element}},
        o None si ocurre un error.
    """

    try:
        with open(nombre_archivo_pdb, "r") as archivo_pdb:
            diccionario_pdb = {}

            for linea in archivo_pdb:
                if linea.startswith("HETATM") or linea.startswith("ATOM"):
                    try:
                        atom_num = int(linea[6:11].strip())
                        x = float(linea[30:38].strip())
                        y = float(linea[38:46].strip())
                        z = float(linea[46:54].strip())

                        diccionario_pdb[atom_num] = {
                            "x": x,
                            "y": y,
                            "z": z,
                        }
                    except (ValueError, IndexError):
                        print(
                            f"Advertencia: Línea HETATM/ATOM con formato incorrecto: {linea.strip()}"
                        )
            return diccionario_pdb

    except FileNotFoundError:
        print(f"Error: Archivo PDB '{nombre_archivo_pdb}' no encontrado.")
        return None
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
        return None


def add_coordenadas(model_name: str, diccionario_resultante, ligando_pdb, proteina_pdb):
    for model in range(1, 101):
        for lig in list(diccionario_resultante[f"{model_name}{model}"].keys()):
            diccionario_resultante[f"{model_name}{model}"][lig]["property"]["x"] = (
                ligando_pdb[f"{model_name}{model}"][int(lig)]["x"]
            )
            diccionario_resultante[f"{model_name}{model}"][lig]["property"]["y"] = (
                ligando_pdb[f"{model_name}{model}"][int(lig)]["y"]
            )
            diccionario_resultante[f"{model_name}{model}"][lig]["property"]["z"] = (
                ligando_pdb[f"{model_name}{model}"][int(lig)]["z"]
            )
            for prot in list(
                diccionario_resultante[f"{model_name}{model}"][lig]["proteins"].keys()
            ):
                if int(prot) in proteina_pdb:
                    diccionario_resultante[f"{model_name}{model}"][lig]["proteins"][
                        prot
                    ]["x"] = proteina_pdb[int(prot)]["x"]
                    diccionario_resultante[f"{model_name}{model}"][lig]["proteins"][
                        prot
                    ]["y"] = proteina_pdb[int(prot)]["y"]
                    diccionario_resultante[f"{model_name}{model}"][lig]["proteins"][
                        prot
                    ]["z"] = proteina_pdb[int(prot)]["z"]


# Ejemplo de uso:
# nombre_archivo = "data4.csv"  # Reemplaza con el nombre de tu archivo
# diccionario_resultante = procesar_csv(nombre_archivo)

# prot = "M11"
# tec = "vina"

# data_name = f"{prot}-{tec}"
# data_name2 = f"{tec}-{prot}-"

# # Ejemplo de uso:
# ligando_pdb = pdb_ligando_parser(
#     f"lig-{data_name}.pdb", data_name2
# )  # Reemplaza con tus nombres de archivo
# proteina_pdb = pdb_proteina_parser(
#     f"rec-{data_name}.pdb"
# )  # Reemplaza con tus nombres de archivo
# add_coordenadas(data_name2, diccionario_resultante, ligando_pdb, proteina_pdb)

# with open("data.json", "w") as archivo_json:
#     json.dump(diccionario_resultante, archivo_json, indent=4)


with open("data.json", "r") as archivo_json:
    data = json.load(archivo_json)


# Procesar el JSON
rows = []
all_keys = set()
for modelo, ligandos in data.items():
    for ligando, details in ligandos.items():
        row = {"000_modelo": modelo, "001_ligando": ligando}

        # Agregar propiedades del ligando
        ligando_props = details["property"]
        idxq = 2
        for key, value in ligando_props.items():
            row[f"{"00" if idxq < 10 else "0"}{idxq}_{key}_ligando"] = value
            idxq += 1

        # Agregar propiedades de cada proteína
        proteins = details.get("proteins", {})
        for idx, protein_props in enumerate(proteins.values(), start=1):
            for key, value in protein_props.items():
                row[f"protein_{"00" if idx < 10 else "0"}{idx}_{key}"] = value

        rows.append(row)
        all_keys.update(row.keys())

# Escribir el CSV
with open("output.csv", "w", newline="") as csvfile:
    fieldnames = sorted(all_keys)
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(rows)

print("Archivo CSV generado como 'output.csv'")
