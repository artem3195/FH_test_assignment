import io
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from torch.autograd import Variable
from nn_arch import Net


def load_image():
    uploaded_file = st.file_uploader(label='Choose image for detection')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


def load_model():
    """Loading trained model from state dict"""
    model = Net()
    m_state_dict = torch.load('train_model_state_dict.pt')
    model.load_state_dict(m_state_dict)
    return model


def get_class_name(ind):
    """Generate class names and return class name by index"""
    class_names = dict(zip(list(range(3)), ['Bed', 'Chair', 'Sofa']))
    return class_names.get(ind)


def preprocess_image(image):
    """Preprocess image"""
    transform_img = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()]
    )
    image_tensor = transform_img(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_ = Variable(image_tensor)
    return input_


def predict_class(image, model):
    """Predict image class"""
    output = model(image)
    index = output.data.cpu().numpy().argmax()
    pred = get_class_name(index)
    return pred


model = load_model()
st.title('Image classification')
img = load_image()
result = st.button('Detect image')
if result:
    prep_img = preprocess_image(img)
    pred_class = predict_class(prep_img, model)
    st.write(f'**Detection result: {pred_class}**')

