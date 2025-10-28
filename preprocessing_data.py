from utilities import *

def preprocess():
    converted_data = csv_reader("Factory_Salary.csv")
    headers = converted_data[0]
    rows = converted_data[1:]
    
    headers.remove(headers[-1])
    targets = []
    for i in range(len(rows)):
        targets.append(rows[i][-1])
        rows[i].pop()
    headers_for_conversion = headers.copy()

    Profession = unique_value_identifier(rows,1)
    Equipment = unique_value_identifier(rows,3)

    features,headers_for_conversion = bin_encoding(rows,Profession,1,headers_for_conversion, "Hot metal cutter", "Profession")
    features,headers_for_conversion = bin_encoding(features,Equipment,2,headers_for_conversion,"Sizing mill","Equipment")
    features,headers_for_conversion = cyclic_encoding(features,0,"Date",headers_for_conversion,get_month,12)
    return features, targets, headers_for_conversion
