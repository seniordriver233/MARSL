import openpyxl


def write_data(list, file_name):
    wb = openpyxl.load_workbook(file_name)
    ws = wb.create_sheet()
    if len(list) > 1:
        for i in range(len(list)):
            ws.append([list[i]])
    else:
        ws = ws.append(list)
    wb.save(file_name)
    wb.close()
