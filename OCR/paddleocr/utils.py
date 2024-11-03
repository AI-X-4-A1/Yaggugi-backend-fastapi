from PIL import Image
import re
import cv2
import io
import numpy as np
import matplotlib.pyplot as plt


def prepreprocess_text(content: bytes):

    # Bytes 데이터를 이미지로 변환
    image = Image.open(io.BytesIO(content))
    img_np = np.array(image)

    # 그레이스케일 변환
    gray_image = image.convert("L")  # 'L' 모드는 그레이스케일
    gray_np = np.array(gray_image)
    is_success, buffer = cv2.imencode(".png", gray_np)
    gray_bytes = buffer.tobytes()  # bytes 형식으로 변환
    return gray_bytes

    # 1. Median Blur 적용
    # blurred = cv2.medianBlur(gray_np, 5)

    # 2. 히스토그램 정규화
    # hist_equalized = cv2.equalizeHist(gray_np)

    # 3. 얼룩 제거 (Morphological Operations)
    # kernel = np.ones((5, 5), np.uint8)
    # morphed = cv2.morphologyEx(gray_np, cv2.MORPH_OPEN, kernel)

    # 원본 및 처리된 이미지 비교
    # plt.figure(figsize=(15, 10))
    # plt.subplot(1, 3, 1)
    # plt.title("Original Image")
    # plt.imshow(img_np)
    # plt.axis('off')
    # plt.subplot(1, 3, 2)
    # plt.title("Median Blurred")
    # plt.imshow(gray_np, cmap='gray')
    # plt.axis('off')
    # plt.subplot(1, 3, 3)
    # plt.title("Histogram Equalized")
    # plt.imshow(morphed, cmap='gray')
    # plt.axis('off')
    # plt.show()


def get_ocr_result(content: bytes, ocr_texts: list, ocr_unique_texts: set, ocr_model):
    ocr_results = []

    ocr_results = ocr_model.ocr(content)
    print(f"ocr_en_results: {ocr_results}")

    for ocr_result in ocr_results[0]:
        ocr_text = ocr_result[1][0]
        # print(f'ocr_en_result: {ocr_en_text}')
        if ocr_text not in ocr_unique_texts:
            ocr_texts.append(ocr_text)
            ocr_unique_texts.add(ocr_text)

    print(f"ocr_texts: {ocr_texts}")


def postpreprocess_text(ocr_texts: list, ocr_unique_texts: set):

    kr_en_pattern = r"^[가-힣a-zA-Z]+$"
    parsed_ocr_set = set()

    for text in ocr_texts:
        # 단위 제외
        if text in ["mg", "g", "ml", "l"]:
            continue

        elif re.match(kr_en_pattern, text):
            parsed_ocr_set.add(text)

    return parsed_ocr_set
