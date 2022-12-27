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
├─ 📂data ( data directory )
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


❏ 사용 전처리 라이브러리 및 프로그램입니다.

```bash
$ pip3 install openmim
$ mim install mmcv-full
$ git clone https://github.com/open-mmlab/mmaction2.git
$ cd mmaction2
$ pip3 install -e .

$ apt-get install ffmpeg

$ sed -i 's/%Y%m%d_%H%M%S/%Y%m%d_%H%M%S%f/g' tools/misc/clip_feature_extraction.py
```

# 실행 방법 (예시)
❏ 비디오 특징 처리방법입니다.
```bash
prepare$ python extract_feature.py \
--video_root_path ../data/videos \
--feature_root_path ../data/features \
--gpu 0 --core 4
```

❏ 데이터세트 분할 방법입니다.
```bash
prepare$ python prepare_dataset.py \
--json_root_path ../data/annotations \
--feature_root_path ../data/features \
--all_data_path ../data/all_data.json \
--dataset_list_path ../data/dataset_split_list.json
```


❏ 훈련 방법입니다.
```bash
cla$ python main.py
```
> 세부 설정은 `config.py`를 참고하세요.

❏ 평가 방법입니다.(test)
```bash
cla$ python test.py --model < MODEL_PATH >
```


# NIA 2022 GEBD ( Generic Event Boundary Detection )
❏ NIA 2022 AI 학습용 데이터로 8:1:1 훈련, 검증, 시험 분할 학습 진행
```
NIA 2022 GEBD 데이터 총 182991 -> train 146392 valid 18299 test 18300  
```
※ 전체 데이터는 [AI - HUB](https://aihub.or.kr/)에서 받을 수 있습니다.  



# Original and Reference
- [contrastive learning approach](https://github.com/hello-jinwoo/LOVEU-CVPR2021)  
- [mmaction2](https://github.com/open-mmlab/mmaction2)
