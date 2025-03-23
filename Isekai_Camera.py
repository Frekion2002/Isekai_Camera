import cv2 as cv
import numpy as np

def cartoonize_image(image_path, output_path):
    # 이미지 불러오기
    img = cv.imread(image_path)
    assert img is not None, "Cannot read given image"

    # 크기 조정 (속도 향상)
    scale = 0.7
    img = cv.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

    # **색상 대비 강화 (CLAHE 적용)**
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv.merge((l, a, b))
    img = cv.cvtColor(lab, cv.COLOR_LAB2BGR)

    # **노이즈 제거**
    img = cv.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    # **그레이스케일 변환 + 블러 줄임**
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gray = cv.medianBlur(img_gray, 3)  # 블러 강도 5 → 3 (덜 뭉개지게)

    # **Canny 엣지 검출 (더 강하게)**
    edges = cv.Canny(img_gray, 100, 200)  # 임계값 증가 (50,150 → 100,200)

    # **Morphology 연산으로 엣지 보강**
    kernel = np.ones((2, 2), np.uint8)  # 커널 크기 조정 가능
    edges = cv.dilate(edges, kernel, iterations=1)  # 엣지를 더 강하게 강조

    # **Bilateral Filter로 색상을 부드럽게**
    color = cv.bilateralFilter(img, d=9, sigmaColor=50, sigmaSpace=50)

    # **엣지를 반전하여 적용**
    edges_inv = cv.bitwise_not(edges)
    cartoon = cv.bitwise_and(color, color, mask=edges_inv)

    # **원본과 변환된 이미지 출력**
    merge = np.hstack((img, cartoon))
    cv.imshow("Cartoon Image: Original | Cartoon", merge)

    cv.imwrite(output_path, cartoon)
    cv.waitKey(0)
    cv.destroyAllWindows()

# 실행 예제
image_path = "D:\\ComputerVision\\Cartoon_image\\fighter2.jpg"
output_path = "D:\\ComputerVision\\Cartoon_image\\cartoon_fighter2_2.jpg"

cartoonize_image(image_path, output_path)
