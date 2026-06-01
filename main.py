r"""
Ứng dụng thư viện OpenCV - các thuật toán xử lý ảnh cơ bản.
Chạy: .venv\Scripts\python.exe main.py
Cài đặt: pip install opencv-python numpy matplotlib
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib

# Backend Agg: ve histogram khong can man hinh (tranh loi tren may khong co GUI)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Thu muc chua anh dau vao va cac thu muc ket qua con
IMAGE_DIR = Path("image")

# Cac phan mo rong anh duoc ho tro
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Nguong nhi phan hoa mac dinh
THRESHOLD_VALUE = 128

# Kich thuoc kernel cho bo loc
MEAN_KERNEL_SIZE = 3
GAUSSIAN_KERNEL_SIZE = 5
GAUSSIAN_SIGMA = 1.0

# So pixel canh bao anh lon (loc/sobel se cham hon)
NGUONG_ANH_LON = 2_000_000


def cau_hinh_console_utf8() -> None:
    """Cau hinh terminal Windows hien thi tieng Viet (UTF-8)."""
    if sys.platform != "win32":
        return
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    try:
        import os

        os.system("chcp 65001 >nul 2>&1")
    except OSError:
        pass


def quet_danh_sach_anh() -> list[Path]:
    """Quet thu muc image, tra ve danh sach file anh hop le (sap xep theo ten)."""
    if not IMAGE_DIR.is_dir():
        return []
    files = [
        p
        for p in sorted(IMAGE_DIR.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return files


def hien_menu_anh(danh_sach: list[Path]) -> None:
    """In menu danh sách ảnh."""
    print("\n===== DANH SÁCH ẢNH =====\n", flush=True)
    for i, duong_dan in enumerate(danh_sach, start=1):
        print(f"{i}. {duong_dan.name}", flush=True)
    print("\n0. Thoát chương trình", flush=True)


def hien_menu_thuat_toan() -> None:
    """In menu các thuật toán xử lý."""
    print("\n===== THUẬT TOÁN =====\n", flush=True)
    print("1. Chuyển ảnh xám (Grayscale)", flush=True)
    print("2. Nhị phân hóa (Thresholding)", flush=True)
    print("3. Biểu đồ histogram (Histogram)", flush=True)
    print("4. Lọc trung bình (Mean Filter)", flush=True)
    print("5. Lọc Gaussian (Gaussian Filter)", flush=True)
    print("6. Phát hiện biên Sobel (Sobel Edge Detection)", flush=True)
    print("7. Thoát chương trình", flush=True)
    print("8. Chọn ảnh khác", flush=True)


def doc_anh(duong_dan: Path) -> np.ndarray | None:
    """Doc anh tu duong dan; tra ve None neu khong doc duoc."""
    anh = cv2.imread(str(duong_dan))
    return anh


def tao_thu_muc_ket_qua(ten_thu_muc: str) -> Path:
    """Tao thu muc con trong image/ neu chua ton tai."""
    thu_muc = IMAGE_DIR / ten_thu_muc
    thu_muc.mkdir(parents=True, exist_ok=True)
    return thu_muc


def chuyen_xam_thu_cong(anh_bgr: np.ndarray) -> np.ndarray:
    """
    Grayscale: chuyen anh mau BGR sang anh xam.
    Cong thuc luminance ITU-R BT.601: Gray = 0.299*R + 0.587*G + 0.114*B.
    Dung NumPy ap dung cong thuc cho toan anh (khong dung cv2.cvtColor).
    """
    b = anh_bgr[:, :, 0].astype(np.float64)
    g = anh_bgr[:, :, 1].astype(np.float64)
    r = anh_bgr[:, :, 2].astype(np.float64)
    xam = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(xam, 0, 255).astype(np.uint8)


def nhi_phan_hoa(anh_xam: np.ndarray, nguong: int = THRESHOLD_VALUE) -> np.ndarray:
    """
    Thresholding: nhi phan hoa anh xam.
    Pixel > nguong -> 255 (trang), nguoc lai -> 0 (den).
    """
    ket_qua = np.where(anh_xam > nguong, 255, 0)
    return ket_qua.astype(np.uint8)


def tinh_histogram(anh_xam: np.ndarray) -> np.ndarray:
    """
    Histogram: dem so luong pixel theo tung muc cuong do 0-255.
    """
    hist = np.zeros(256, dtype=np.int64)
    for gia_tri in anh_xam.ravel():
        hist[int(gia_tri)] += 1
    return hist


def ve_va_luu_histogram(anh_xam: np.ndarray, duong_dan_luu: Path) -> None:
    """Ve bieu do histogram bang Matplotlib va luu file PNG."""
    hist = tinh_histogram(anh_xam)
    muc = np.arange(256)

    plt.figure(figsize=(10, 5))
    plt.bar(muc, hist, width=1.0, color="steelblue", edgecolor="none")
    plt.title("Histogram ảnh xám")
    plt.xlabel("Mức cường độ")
    plt.ylabel("Số lượng pixel")
    plt.xlim(0, 255)
    plt.tight_layout()
    plt.savefig(str(duong_dan_luu), dpi=150)
    plt.close()


def tao_kernel_trung_binh(kich_thuoc: int) -> np.ndarray:
    """Tao kernel mean: moi phan tu bang 1/(k*k) de lam min anh."""
    k = float(kich_thuoc)
    return np.ones((kich_thuoc, kich_thuoc), dtype=np.float64) / (k * k)


def tao_kernel_gaussian(kich_thuoc: int, sigma: float) -> np.ndarray:
    """
    Gaussian Filter: tao kernel Gaussian 2D.
    Gia tri theo ham mu exp(-(x^2+y^2)/(2*sigma^2)), chuan hoa tong = 1.
    """
    if kich_thuoc % 2 == 0:
        kich_thuoc += 1

    ban_kinh = kich_thuoc // 2
    truc = np.arange(-ban_kinh, ban_kinh + 1, dtype=np.float64)
    xx, yy = np.meshgrid(truc, truc)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    tong = np.sum(kernel)
    if tong > 0:
        kernel /= tong
    return kernel


def tich_chap_2d(kenh: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Tich chap 2D tu cai dat (khong dung cv2.filter2D).
    Padding bien kieu 'edge', cong don tung vi tri kernel * vung anh tuong ung.
    """
    h, w = kenh.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    padded = np.pad(kenh.astype(np.float64), ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
    ket_qua = np.zeros((h, w), dtype=np.float64)

    for ki in range(kh):
        for kj in range(kw):
            ket_qua += kernel[ki, kj] * padded[ki : ki + h, kj : kj + w]

    return np.clip(ket_qua, 0, 255).astype(np.uint8)


def loc_trung_binh(anh_bgr: np.ndarray, kich_thuoc: int = MEAN_KERNEL_SIZE) -> np.ndarray:
    """
    Mean Filter: lam min anh bang tich chap kernel trung binh.
    Ap dung rieng tung kenh B, G, R.
    """
    kernel = tao_kernel_trung_binh(kich_thuoc)
    h, w, c = anh_bgr.shape
    ket_qua = np.zeros_like(anh_bgr)

    for ch in range(c):
        ket_qua[:, :, ch] = tich_chap_2d(anh_bgr[:, :, ch], kernel)

    return ket_qua


def loc_gaussian(
    anh_bgr: np.ndarray,
    kich_thuoc: int = GAUSSIAN_KERNEL_SIZE,
    sigma: float = GAUSSIAN_SIGMA,
) -> np.ndarray:
    """
    Gaussian Filter: lam min co trong so theo khoang cach tam kernel.
    Giam nhieu tot hon mean filter, giu bien mem hon.
    """
    kernel = tao_kernel_gaussian(kich_thuoc, sigma)
    h, w, c = anh_bgr.shape
    ket_qua = np.zeros_like(anh_bgr)

    for ch in range(c):
        ket_qua[:, :, ch] = tich_chap_2d(anh_bgr[:, :, ch], kernel)

    return ket_qua


def phat_hien_bien_sobel(anh_xam: np.ndarray) -> np.ndarray:
    """
    Sobel Edge Detection: xap xi dao ham theo huong x va y.
    Gx, Gy la kernel 3x3; do lon gradient = sqrt(Gx^2 + Gy^2).
    """
    gx_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    gy_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

    gx = tich_chap_2d(anh_xam.astype(np.float64), gx_kernel).astype(np.float64)
    gy = tich_chap_2d(anh_xam.astype(np.float64), gy_kernel).astype(np.float64)

    do_lon = np.sqrt(gx ** 2 + gy ** 2)
    do_lon = np.clip(do_lon, 0, 255)
    return do_lon.astype(np.uint8)


def luu_anh_xam(anh_xam: np.ndarray, duong_dan: Path) -> bool:
    """Luu anh mot kenh (grayscale/binary/sobel)."""
    return bool(cv2.imwrite(str(duong_dan), anh_xam))


def luu_anh_mau(anh_bgr: np.ndarray, duong_dan: Path) -> bool:
    """Luu anh BGR ba kenh."""
    return bool(cv2.imwrite(str(duong_dan), anh_bgr))


def in_loi_luu_ket_qua() -> None:
    """In thông báo lỗi khi không ghi được file kết quả."""
    print("Lỗi: không lưu được file kết quả. Vui lòng kiểm tra lại đường dẫn.", flush=True)


def in_thanh_cong_luu(duong_dan: Path) -> None:
    """In thông báo sau khi xử lý và lưu file thành công."""
    print(f"Đã xử lý thành công. Kết quả đã được lưu tại: {duong_dan.resolve()}", flush=True)


def in_thong_tin_anh(anh_bgr: np.ndarray) -> None:
    """In kích thước ảnh và cảnh báo nếu ảnh lớn (xử lý có thể lâu)."""
    h, w = anh_bgr.shape[:2]
    so_pixel = h * w
    print(f"Kích thước ảnh: {w} x {h} ({so_pixel:,} pixel)", flush=True)
    if so_pixel > NGUONG_ANH_LON:
        print(
            "Cảnh báo: Ảnh lớn — thuật toán 4, 5, 6 có thể mất vài phút. "
            "Vui lòng đợi, chương trình vẫn đang chạy.",
            flush=True,
        )


def xu_ly_grayscale(anh_bgr: np.ndarray) -> Path | None:
    """Chay Grayscale va luu result.jpg."""
    anh_xam = chuyen_xam_thu_cong(anh_bgr)
    thu_muc = tao_thu_muc_ket_qua("grayscale")
    duong_dan = thu_muc / "result.jpg"
    if not luu_anh_xam(anh_xam, duong_dan):
        in_loi_luu_ket_qua()
        return None
    in_thanh_cong_luu(duong_dan)
    return duong_dan


def xu_ly_thresholding(anh_bgr: np.ndarray) -> Path | None:
    """Chay Thresholding va luu result.jpg."""
    anh_xam = chuyen_xam_thu_cong(anh_bgr)
    anh_np = nhi_phan_hoa(anh_xam)
    thu_muc = tao_thu_muc_ket_qua("threshold")
    duong_dan = thu_muc / "result.jpg"
    if not luu_anh_xam(anh_np, duong_dan):
        in_loi_luu_ket_qua()
        return None
    in_thanh_cong_luu(duong_dan)
    return duong_dan


def xu_ly_histogram(anh_bgr: np.ndarray) -> Path | None:
    """Chay Histogram va luu histogram.png."""
    anh_xam = chuyen_xam_thu_cong(anh_bgr)
    thu_muc = tao_thu_muc_ket_qua("histogram")
    duong_dan = thu_muc / "histogram.png"
    try:
        ve_va_luu_histogram(anh_xam, duong_dan)
    except (OSError, RuntimeError, ValueError) as e:
        print(f"Lỗi khi vẽ histogram: {e}", flush=True)
        in_loi_luu_ket_qua()
        return None
    in_thanh_cong_luu(duong_dan)
    return duong_dan


def xu_ly_mean_filter(anh_bgr: np.ndarray) -> Path | None:
    """Chay Mean Filter va luu result.jpg."""
    ket_qua = loc_trung_binh(anh_bgr)
    thu_muc = tao_thu_muc_ket_qua("mean_filter")
    duong_dan = thu_muc / "result.jpg"
    if not luu_anh_mau(ket_qua, duong_dan):
        in_loi_luu_ket_qua()
        return None
    in_thanh_cong_luu(duong_dan)
    return duong_dan


def xu_ly_gaussian_filter(anh_bgr: np.ndarray) -> Path | None:
    """Chay Gaussian Filter va luu result.jpg."""
    ket_qua = loc_gaussian(anh_bgr)
    thu_muc = tao_thu_muc_ket_qua("gaussian_filter")
    duong_dan = thu_muc / "result.jpg"
    if not luu_anh_mau(ket_qua, duong_dan):
        in_loi_luu_ket_qua()
        return None
    in_thanh_cong_luu(duong_dan)
    return duong_dan


def xu_ly_sobel(anh_bgr: np.ndarray) -> Path | None:
    """Chay Sobel Edge Detection va luu result.jpg."""
    anh_xam = chuyen_xam_thu_cong(anh_bgr)
    bien = phat_hien_bien_sobel(anh_xam)
    thu_muc = tao_thu_muc_ket_qua("sobel")
    duong_dan = thu_muc / "result.jpg"
    if not luu_anh_xam(bien, duong_dan):
        in_loi_luu_ket_qua()
        return None
    in_thanh_cong_luu(duong_dan)
    return duong_dan


def nhap_so_nguyen(nhan: str) -> int | None:
    """Đọc số nguyên từ bàn phím; trả về None nếu nhập không hợp lệ."""
    try:
        chuoi = input(nhan).strip()
        if not chuoi:
            return None
        return int(chuoi)
    except (ValueError, EOFError):
        return None


def chon_anh(danh_sach: list[Path]) -> Path | None:
    """
    Cho nguoi dung chon anh.
    Tra ve Path anh, None neu thoat (0), hoac sentinel dac biet khong dung o day.
    """
    while True:
        hien_menu_anh(danh_sach)
        lua_chon = nhap_so_nguyen("Chọn ảnh: ")
        if lua_chon is None:
            print("Lỗi: vui lòng nhập số nguyên (ví dụ: 1).", flush=True)
            continue
        if lua_chon == 0:
            return None
        if 1 <= lua_chon <= len(danh_sach):
            return danh_sach[lua_chon - 1]
        print(f"Lỗi: chọn sai số. Hãy nhập từ 0 đến {len(danh_sach)}.", flush=True)


def thuc_thi_thuat_toan(lua_chon: int, anh_bgr: np.ndarray) -> None:
    """Gọi hàm xử lý tương ứng; bắt lỗi để in ra terminal thay vì thoát im lặng."""
    ten_thuat_toan = {
        1: "Chuyển ảnh xám (Grayscale)",
        2: "Nhị phân hóa (Thresholding)",
        3: "Biểu đồ histogram (Histogram)",
        4: "Lọc trung bình (Mean Filter)",
        5: "Lọc Gaussian (Gaussian Filter)",
        6: "Phát hiện biên Sobel (Sobel Edge Detection)",
    }
    hanh_xu_ly = {
        1: xu_ly_grayscale,
        2: xu_ly_thresholding,
        3: xu_ly_histogram,
        4: xu_ly_mean_filter,
        5: xu_ly_gaussian_filter,
        6: xu_ly_sobel,
    }

    print(f"\nĐang chạy: {ten_thuat_toan[lua_chon]} ...", flush=True)
    try:
        hanh_xu_ly[lua_chon](anh_bgr)
    except Exception as e:
        print(f"Lỗi khi xử lý: {e}", flush=True)


def vong_lap_thuat_toan(anh_bgr: np.ndarray, ten_anh: str) -> str:
    """
    Vòng lặp menu thuật toán cho một ảnh đã chọn.
    Trả về 'thoat' để kết thúc chương trình, 'doi_anh' để chọn ảnh khác.
    """
    print(f"\nĐang xử lý: {ten_anh}", flush=True)
    in_thong_tin_anh(anh_bgr)

    while True:
        hien_menu_thuat_toan()
        lua_chon = nhap_so_nguyen("Chọn thuật toán: ")
        if lua_chon is None:
            print("Lỗi: vui lòng nhập số nguyên (ví dụ: 1).", flush=True)
            continue

        if lua_chon in (0, 7):
            return "thoat"
        if lua_chon == 8:
            return "doi_anh"

        if 1 <= lua_chon <= 6:
            thuc_thi_thuat_toan(lua_chon, anh_bgr)
        else:
            print("Lỗi: lựa chọn không hợp lệ. Nhập 1-8, 0 hoặc 7 để thoát.", flush=True)


def main() -> None:
    """Diem vao chuong trinh: chon anh -> chon thuat toan -> lap."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("XU LY ANH BANG OpenCV")
    print("=" * 50)

    while True:
        danh_sach = quet_danh_sach_anh()
        if not danh_sach:
            print(
                f"\nLỗi: không có ảnh trong thư mục '{IMAGE_DIR.resolve()}'.\n"
                f"Hãy thêm file .jpg, .jpeg, .png hoặc .bmp vào thư mục image/.",
                flush=True,
            )
            sys.exit(1)

        duong_dan_anh = chon_anh(danh_sach)
        if duong_dan_anh is None:
            print("\nKết thúc chương trình.", flush=True)
            break

        anh = doc_anh(duong_dan_anh)
        if anh is None:
            print(f"Lỗi: không đọc được ảnh '{duong_dan_anh.name}'.", flush=True)
            continue

        ket_qua = vong_lap_thuat_toan(anh, duong_dan_anh.name)
        if ket_qua == "thoat":
            print("\nKết thúc chương trình.", flush=True)
            break


if __name__ == "__main__":
    main()
