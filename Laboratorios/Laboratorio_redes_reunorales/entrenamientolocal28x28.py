import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

# Configuración de rutas
ruta_train = '/home/alexsc/Documentos/publico/ayudalaboratorioIA/EscenasNaturales/train'
ruta_test = '/home/alexsc/Documentos/publico/ayudalaboratorioIA/EscenasNaturales/test'

# Transformaciones para redimensionar y normalizar las imágenes
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Cargar datasets
train_dataset = ImageFolder(root=ruta_train, transform=transform)
test_dataset = ImageFolder(root=ruta_test, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(28*28*3, 128)  # Capa oculta 1
        self.fc2 = nn.Linear(128, 64)       # Capa oculta 2
        self.fc3 = nn.Linear(64, len(train_dataset.classes))  # Capa de salida

    def forward(self, x):
        x = x.view(-1, 28*28*3)  # Aplanar las imágenes
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = NeuralNet()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento
num_epochs = 5
loss_values = []

for epoch in range(num_epochs):
    epoch_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    loss_values.append(avg_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

# Evaluación del modelo
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Precisión del modelo: {accuracy:.2f}%')


# Graficar la pérdida durante el entrenamiento
plt.plot(range(num_epochs), loss_values)
plt.xlabel('Épocas')
plt.ylabel('Pérdida promedio')
plt.title('Pérdida durante el entrenamiento')
plt.savefig('grafico_perdida.png')
plt.show()

# Visualizar algunas imágenes con predicciones
data_iter = iter(test_loader)
images, labels = next(data_iter)
outputs = model(images)
_, predicted = torch.max(outputs, 1)

# Mostrar imágenes y sus etiquetas
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i in range(5):
    image = images[i].permute(1, 2, 0).numpy()
    image = (image * 0.5) + 0.5  # Desnormalizar para visualizar correctamente
    axes[i].imshow(image)
    axes[i].set_title(f'Predicción: {train_dataset.classes[predicted[i]]}')
    axes[i].axis('off')
plt.savefig('predicciones_visuales.png')
plt.show()
