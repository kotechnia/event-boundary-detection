# Contranstive Learning Approach (CLA)

❏ CLA의 구조는 다음과 같습니다.
```
📂CLA 
├─ 📂data
├─ 📂ensemble
├─ 📂sf_tsn_each_branch
├─ 📂using_similarity_map
├─ 📂cla ( for nia2022 )
│   ├─ 📄config.py
│   ├─ 📄dataset.py
│   ├─ 📄main.py
│   ├─ 📄network.py
│   ├─ 📄resnet.py
│   ├─ 📄test.py
│   ├─ 📄validate.py
│   └─ 📄validation.py
└─ 📄README.md
```

❏ 테스트 시스템 사양은 다음과 같습니다.
```
```

❏ 사용 라이브러리 및 프로그램입니다.
```bash
$ pip install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install numpy==1.19.2 matplotlib==3.4.1 tqdm
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
--sigma < float > \
--fold < int >
```

❏ 평가 방법입니다.(test)
```bash
python test1.py --model < MODEL_PATH >
```
