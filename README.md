# vscode에서 cuda를 사용하여 딥러닝 모델 학습하기
cuda를 사용한 딥러닝 모델 학습 테스트 코드 입니다.

시계열 온도 데이터인 Jena Climate Dataset을 이용하여 GRU 모델을 학습하였습니다.

모델의 성능이 높고 낮음을 확인하기보다 Training loss가 epoch마다 떨어지는지 확인하는 것이 목적입니다.
따라서 epochs == 10입니다.

자세한 내용은 [링크](https://velog.io/@youarethewon/vscode%EC%97%90%EC%84%9C-%EB%94%A5%EB%9F%AC%EB%8B%9D-%ED%99%9C%EC%9A%A9)를 참고해주세요.

## 사전 준비 사항
vscode와 파이썬이 설치되어 있고 vscode에서 cuda를 사용할 수 있도록 설정되어 있어야 합니다.

cuda를 사용하기위한 설정은 [링크](https://velog.io/@youarethewon/vscode%EC%97%90%EC%84%9C-cuda-%EC%82%AC%EC%9A%A9)를 참고해주세요.

가상 환경을 준비하는 것을 추천합니다.

## 소스 파일 다운로드
```
git clone https://github.com/finale22/test_for_cuda.git
cd test_for_cuda
```
가상 환경을 만들고 활성화한 후 필요한 패키지를 설치합니다.

가상 환경 명은 myenv로 하겠습니다.

다른 이름을 사용하고 싶다면 myenv 대신 원하는 이름을 입력하여 진행해주세요.
```
python -m venv myenv
source myenv/Scripts/activate
pip install -r requirements.txt
```

## 사용법
학습 명령어
```
python main.py --mode train
```

성능 평가 명령어
```
python main.py --mode evaluate
```

