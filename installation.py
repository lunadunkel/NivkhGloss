import os
import subprocess
import sys
import requests
from tqdm import tqdm


NAVEC_URL = "https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar"
SPACY_MODEL = "ru_core_news_sm"

def install_dependencies():
    """Устанавливает конкретные зависимости: navec и spacy"""
    packages = [
        "navec",
        "spacy"
    ]

    for package in packages:
        print(f"Установка {package}...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Ошибка при установке {package}\n{result.stdout}")
        else:
            print(f"{package} установлен")

def download_spacy_model():
    import spacy
    print(f"Установка модели {SPACY_MODEL}...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", SPACY_MODEL])
    print(f"{SPACY_MODEL} установлен")
    nlp = spacy.load(SPACY_MODEL)
    return nlp

def download_navec():
    path = "navec_hudlit_v1_12B_500K_300d_100q.tar"
    response = requests.get(NAVEC_URL, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(path, "wb") as f:
        for data in tqdm(response.iter_content(chunk_size=1024),
                             desc="Загрузка Navec модели",
                             total=total_size // 1024 + 1,
                             unit='KB',
                             ncols=80):
            if data:
                f.write(data)
    from navec import Navec
    navec = Navec.load(path)
    return navec


install_dependencies()
nlp = download_spacy_model()
navec = download_navec()