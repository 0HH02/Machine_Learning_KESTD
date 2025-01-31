import pandas as pd

file_path = '././DB/output.csv'  
data = pd.read_csv(file_path, low_memory=False)

data['000_modelo'] = data['000_modelo'].astype('category').cat.codes

column_name_mappings = {
    "htype": ["aromatic", "aliphatic", "polar", "methyl"],
    "irr": ["true", "false"],
    "mol": ["ligand", "receptor"],
    "dssp": ["alpha-helix", "beta-sheet", "turns", "loops"]
}

array_columns = [col for col in data.columns if data[col].apply(lambda x: isinstance(x, str) and ";" in x).any()]

for col in array_columns:

    col_index = data.columns.get_loc(col)

    col_prefix = None
    for key in column_name_mappings:
        if key in col:
            col_prefix = key
            break

    max_length = data[col].dropna().apply(lambda x: len(x.split(';')) if isinstance(x, str) else 0).max()

    if col_prefix and max_length == len(column_name_mappings[col_prefix]):
        new_col_names = [f"{col}_{name}" for name in column_name_mappings[col_prefix]]
    else:
        new_col_names = [f"{col}_{i}" for i in range(max_length)]

    new_columns = data[col].str.split(';', expand=True).astype(float)

    for i, new_col in enumerate(new_col_names):
        data.insert(col_index + i, new_col, new_columns[i])

    data.drop(columns=[col], inplace=True)

output_file_path = 'automl/DB/output_for_automl.csv'  
data.to_csv(output_file_path, index=False)

print(f"Archivo procesado guardado en: {output_file_path}")