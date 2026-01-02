from pathlib import Path
import os

from paddleocr import PaddleOCR

BASE_DIR = Path(__file__).resolve().parent
image_path = (BASE_DIR / "menu3.png").resolve()   # 지금 당신 로그에 맞춤

print("CWD:", os.getcwd())
print("SCRIPT:", BASE_DIR)
print("IMAGE:", image_path)
print("EXISTS:", image_path.exists())

ocr = PaddleOCR(
    lang="korean",
    use_textline_orientation=True,     # ✅ use_angle_cls 대신
)

print("\n=== PREDICT START ===")
results = ocr.predict(str(image_path))   # ✅ 최신 권장 API

got_any = False

# ✅ predict()가 generator/iterable이면 반드시 순회해야 출력이 나옵니다.
for i, res in enumerate(results, start=1):
    got_any = True
    print(f"\n--- RESULT #{i} ---")

    # 1) 가장 확실: res.print() (문서/이슈에서 많이 쓰는 방식)
    if hasattr(res, "print"):
        res.print()

    # 2) JSON으로 저장/확인
    if hasattr(res, "save_to_json"):
        out_dir = BASE_DIR / "ocr_output"
        out_dir.mkdir(exist_ok=True)
        res.save_to_json(str(out_dir))   # ocr_output 폴더에 json 저장
        print("Saved JSON to:", out_dir)

    # 3) 시각화 이미지 저장(박스 그려진 결과)
    if hasattr(res, "save_to_img"):
        out_dir = BASE_DIR / "ocr_output"
        out_dir.mkdir(exist_ok=True)
        res.save_to_img(str(out_dir))
        print("Saved IMG to:", out_dir)

print("\n=== PREDICT END ===")
if not got_any:
    print("결과 객체가 하나도 생성되지 않았습니다. (pipeline 출력이 비어 있음)")

