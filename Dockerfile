FROM python:3.8.9

COPY requirements.txt .

RUN pip3 install -r requirements.txt && rm requirements.txt

EXPOSE 8501

COPY train_model_state_dict.pt train_model_state_dict.pt

COPY nn_arch.py nn_arch.py

COPY app.py app.py

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]