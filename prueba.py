import torch
print(torch.cuda.is_available())  # Debe imprimir True
print(torch.cuda.device_count())  # Debe mostrar el número de GPUs
print(torch.cuda.get_device_name(0))  # Nombre de la GPU
