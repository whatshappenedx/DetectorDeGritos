import torch
import torchaudio
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import transforms

# Define una función para transformar datos de audio en imágenes (espectrogramas)
def transform_data_to_image(audio, sample_rate):
    # Genera un espectrograma de Mel utilizando los parámetros dados
    spectrogram_tensor = (torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=64, n_fft=1024)(audio)[0] + 1e-10).log2()
    # Guarda el espectrograma como una imagen
    image_path = 'voice_image.png'
    plt.imsave(image_path, spectrogram_tensor.numpy(), cmap='viridis')
    return image_path

def process_file(filename, model):
    # Selecciona el dispositivo de procesamiento (GPU o CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Establece el modelo en modo de evaluación
    model.eval()

    # Mueve el modelo al dispositivo
    model.to(device)

    # Define la cadena de transformaciones de imagen
    transform = transforms.Compose([
        transforms.Resize((64, 862)),  # Redimensiona la imagen
        transforms.ToTensor(),         # Convierte la imagen a un tensor
        transforms.Lambda(lambda x: x[:3, :, :])  # Asegura que la imagen tiene 3 canales
    ])

    # Carga el archivo de audio
    audio, sample_rate = torchaudio.load(filename)

    # Transforma el audio en una imagen y guárdala
    image_path = transform_data_to_image(audio, sample_rate)

    # Carga la imagen guardada y aplica las transformaciones
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Añade una dimensión de batch para el modelo

    # Realiza predicciones usando el modelo
    with torch.no_grad():
        outputs = model(image.to(device))

    # Muestra los resultados de la salida del modelo
    print(outputs)

    # Determina la clase predicha (1 indica un grito)
    predict = outputs.argmax(dim=1).cpu().detach().numpy().ravel()[0]

    return predict == 1  # Devuelve True si la predicción es 'grito'