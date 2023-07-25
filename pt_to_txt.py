import torch
from chess_evaluation_prediction import Geohotz

name = 'geohotz_20k_cpu'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device', device)

model = Geohotz().to(device)

model.load_state_dict(
    torch.load(
        name + ".pt"
    )
)

weightCount = 0

weightOutput = ""

for param in model.parameters():
    weightOutput += str(list(param.size())) + '\n'
    weightCount += param.data.flatten().size()[0]

    print(param)
    break

    for value in param.data.flatten().tolist():
        weightOutput += str(value) + "\n"

print(f"Converted {weightCount} weights. Compressed size: {weightCount / 8}")

weightFile = open(
    f"{name}.txt", "w"
)
weightFile.write(weightOutput)
weightFile.close()