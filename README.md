# Generic Event Boundary Detection (GEBD)

❏ GEBD 구조는 다음과 같습니다.
```
📂GEBD 
├─ 📂cla ( GEBD model for nia2022 )
│   ├─ 📄config.py
│   ├─ 📄dataset.py
│   ├─ 📄main.py
│   ├─ 📄network.py
│   ├─ 📄resnet.py
│   ├─ 📄test.py
│   ├─ 📄validate.py
│   └─ 📄validation.py
├─ 📂mmaction2 ( video preprocess module )
├─ 📂prepare 
└─ 📄README.md
```

❏ 테스트 시스템 사양은 다음과 같습니다.
```
Ubuntu 22.04 LTS
Python 3.8.10 
Torch 1.8.1+cu111 
CUDA 11.1
cuDnn 8.2.0    
```

❏ 사용 라이브러리 및 프로그램입니다.
```bash
$ cat requirements.txt

torch==1.8.1+cu111
torchvision==0.9.1+cu111
tqdm
parmap
openmim

# package location
--find-links https://download.pytorch.org/whl/torch_stable.html

$ pip install -r requirements.txt
```

```bash
$ pip3 install openmim
$ mim install mmcv-full
$ git clone https://github.com/open-mmlab/mmaction2.git
$ cd mmaction2
$ pip3 install -e .
```
```bash
$ sed -i 's/%Y%m%d_%H%M%S/%Y%m%d_%H%M%S%f/g' tools/misc/clip_feature_extraction.py
```

# 실행 방법 (예시)
❏ 훈련 방법입니다.
```bash
python main.py
```
❏ 평가 방법입니다.(validation)
```bash
python validation.py \
--model < MODEL_PATH > \
--sigma < float > 
```

❏ 평가 방법입니다.(test)
```bash
python test.py --model < MODEL_PATH >
```



# Original and Reference
- [contrastive learning approach](https://github.com/hello-jinwoo/LOVEU-CVPR2021)  
- [mmaction2](https://github.com/open-mmlab/mmaction2)
