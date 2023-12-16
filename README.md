# FastSpeech2



## Installation guide

install requirements

```
pip install -r requirements.txt
```

load model

```
python3 load_checkpoints.py
```

run test

```
python3 test.py -c ./hw_asr/configs/one_batch_resume.json -r ./model.pth -t ./archive -o ./results.txt
```

-t is a folder where all audios are, -o is a file to put result in 

result look like this  
FILENAME|SPOOF PROB|BONAFIDE PROB  
audio_2.flac|0.9958542585372925|0.004145755432546139  
audio_1.flac|0.9957783222198486|0.004221669398248196  
audio_3.flac|0.9957547187805176|0.004245298448950052  

