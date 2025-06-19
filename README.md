# 🧠 brain_meningioma

뇌수막종(Meningioma) segmentation을 위한 2D 모델 기반 의료영상 처리 프로젝트.

---

## 📦 환경 정보 (conda env: `meningioma`)

- Python: 3.10.x  
- CUDA: 12.1  
- PyTorch: 2.5.1  
- Torchvision: 0.20.1  
- MONAI: 1.5.0

> ⚠️ 위 버전 외에는 HD-BET, MONAI 호환성 문제가 발생할 수 있음

---

## 📁 디렉토리 구조 (예정)
brain_meningioma/
├── HD-BET/                # skull-stripping 도구
├── data/                  # 의료 영상 (soft link or 원본)
├── outputs/               # 추론/마스크 결과
├── notebooks/             # 분석용 Jupyter 노트북
├── src/                   # 학습/전처리 코드
├── environment.yml        # Conda 환경 파일
└── README.md
