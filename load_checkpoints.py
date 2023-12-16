import gdown
import shutil
import os
from pathlib import Path


URL_LINKS = {
    'model_best': 'https://drive.google.com/uc?id=1W3tKW81nTyLyRzMAtOMa_L-aqVVLQ5uI'
    
}

def main():
    dir = Path(__file__).absolute().resolve().parent
    for name in URL_LINKS:
        gdown.download(URL_LINKS[name], 'model.pth', quiet=False)

if __name__ == "__main__":
    main()