r"""
Ứng dụng thư viện OpenCV - các thuật toán xử lý ảnh cơ bản.
Chạy: python main.py
Cài đặt: pip install opencv-python matplotlib
"""

import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMAGE_DIR = Path("image")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
THRESHOLD_VALUE = 128
BLUR_KERNEL = (3, 3)
GAUSSIAN_KERNEL = (5, 5)
GAUSSIAN_SIGMA = 1.0


def cau_hinh_console_utf8():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass


def quet_danh_sach_anh():
    """Quét thư mục image/, trả về danh sách file ảnh."""
    if not IMAGE_DIR.is_dir():
        return []
    return sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def hien_menu_anh(danh_sach):
    print("\n===== DANH SÁCH ẢNH =====\n")
    for i, f in enumerate(danh_sach, 1):
        print(f"{i}. {f.name}")
    print("\n0. Thoát chương trình")


def hien_menu_thuat_toan():
    print("\n===== THUẬT TOÁN =====\n")
    print("1. Chuyển ảnh xám (Grayscale)")
    print("2. Nhị phân hóa (Thresholding)")
    print("3. Biểu đồ histogram (Histogram)")
    print("4. Lọc trung bình (Mean Filter)")
    print("5. Lọc Gaussian (Gaussian Filter)")
    print("6. Phát hiện biên Sobel (Sobel Edge Detection)")
    print("7. Thoát chương trình")
    print("8. Chọn ảnh khác")


def luu_ket_qua(anh, thu_muc, ten_file):
    """Lưu ảnh kết quả vào thư mục con của image/."""
    out_dir = IMAGE_DIR / thu_muc
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ten_file
    if cv2.imwrite(str(path), anh):
        print(f"Đã lưu kết quả tại: {path.resolve()}")
    else:
        print("Lỗi: không lưu được file kết quả.")


def xu_ly_grayscale(anh):
    ket_qua = cv2.cvtColor(anh, cv2.COLOR_BGR2GRAY)
    luu_ket_qua(ket_qua, "grayscale", "result.jpg")


def xu_ly_thresholding(anh):
    xam = cv2.cvtColor(anh, cv2.COLOR_BGR2GRAY)
    _, ket_qua = cv2.threshold(xam, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
    luu_ket_qua(ket_qua, "threshold", "result.jpg")


def xu_ly_histogram(anh):
    xam = cv2.cvtColor(anh, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([xam], [0], None, [256], [0, 256])
    out_dir = IMAGE_DIR / "histogram"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "histogram.png"
    plt.figure(figsize=(10, 5))
    plt.plot(hist, color="steelblue")
    plt.title("Histogram ảnh xám")
    plt.xlabel("Mức cường độ")
    plt.ylabel("Số lượng pixel")
    plt.xlim(0, 255)
    plt.tight_layout()
    plt.savefig(str(path), dpi=150)
    plt.close()
    print(f"Đã lưu kết quả tại: {path.resolve()}")


def xu_ly_mean_filter(anh):
    ket_qua = cv2.blur(anh, BLUR_KERNEL)
    luu_ket_qua(ket_qua, "mean_filter", "result.jpg")


def xu_ly_gaussian_filter(anh):
    ket_qua = cv2.GaussianBlur(anh, GAUSSIAN_KERNEL, GAUSSIAN_SIGMA)
    luu_ket_qua(ket_qua, "gaussian_filter", "result.jpg")


def xu_ly_sobel(anh):
    xam = cv2.cvtColor(anh, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(xam, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(xam, cv2.CV_64F, 0, 1, ksize=3)
    abs_x = cv2.convertScaleAbs(gx)
    abs_y = cv2.convertScaleAbs(gy)
    ket_qua = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
    luu_ket_qua(ket_qua, "sobel", "result.jpg")


def chon_anh(danh_sach):
    while True:
        hien_menu_anh(danh_sach)
        try:
            chon = int(input("Chọn ảnh: ").strip())
        except ValueError:
            print("Lỗi: vui lòng nhập số nguyên.")
            continue
        if chon == 0:
            return None
        if 1 <= chon <= len(danh_sach):
            return danh_sach[chon - 1]
        print(f"Lỗi: hãy nhập từ 0 đến {len(danh_sach)}.")


def vong_lap_thuat_toan(anh, ten_anh):
    xu_ly = {
        1: xu_ly_grayscale,
        2: xu_ly_thresholding,
        3: xu_ly_histogram,
        4: xu_ly_mean_filter,
        5: xu_ly_gaussian_filter,
        6: xu_ly_sobel,
    }
    print(f"\nĐang xử lý: {ten_anh}")
    h, w = anh.shape[:2]
    print(f"Kích thước ảnh: {w} x {h}")
    while True:
        hien_menu_thuat_toan()
        try:
            chon = int(input("Chọn thuật toán: ").strip())
        except ValueError:
            print("Lỗi: vui lòng nhập số nguyên.")
            continue
        if chon in (0, 7):
            return "thoat"
        if chon == 8:
            return "doi_anh"
        if chon in xu_ly:
            print(f"\nĐang chạy thuật toán {chon}...")
            xu_ly[chon](anh)
        else:
            print("Lỗi: lựa chọn không hợp lệ.")


def main():
    cau_hinh_console_utf8()
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 50)
    print("XU LY ANH BANG OpenCV")
    print("=" * 50)
    while True:
        danh_sach = quet_danh_sach_anh()
        if not danh_sach:
            print(f"\nLỗi: không có ảnh trong '{IMAGE_DIR.resolve()}'.")
            print("Hãy thêm file .jpg, .jpeg, .png hoặc .bmp vào thư mục image/.")
            sys.exit(1)
        duong_dan = chon_anh(danh_sach)
        if duong_dan is None:
            print("\nKết thúc chương trình.")
            break
        anh = cv2.imread(str(duong_dan))
        if anh is None:
            print(f"Lỗi: không đọc được ảnh '{duong_dan.name}'.")
            continue
        if vong_lap_thuat_toan(anh, duong_dan.name) == "thoat":
            print("\nKết thúc chương trình.")
            break


if __name__ == "__main__":
    main()
