# Ứng dụng thư viện OpenCV — Xử lý ảnh cơ bản

Chương trình Python chạy trên **terminal**, triển khai các thuật toán xử lý ảnh cơ bản phục vụ đề tài:

> **Ứng dụng thư viện mã nguồn mở OpenCV xây dựng các thuật toán cơ bản trong xử lý ảnh**

Các thuật toán được **tự cài đặt** (công thức, tích chập, kernel…), hạn chế dùng hàm xử lý sẵn của OpenCV (`cvtColor`, `threshold`, `GaussianBlur`, `Sobel`, `filter2D`…). OpenCV chỉ dùng để **đọc và ghi file ảnh**.

---

## Yêu cầu hệ thống

- **Python 3.12+** (khuyến nghị 3.13)
- Windows / Linux / macOS
- Terminal (không cần GUI, web hay database)

## Cài đặt

**Khuyến nghị:** dùng môi trường ảo `.venv` (Python 3.13) để lệnh `python` chạy đúng thư viện.

```powershell
cd project
py -3.13 -m venv .venv
hoặc py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Hoặc cài trực tiếp (không dùng venv):

```bash
py -3.13 -m pip install -r requirements.txt
```

> **Lưu ý:** Lệnh `python` mặc định trên máy có thể là Python MinGW 3.9 **không cài được** `opencv-python`. Nếu gặp `ModuleNotFoundError: No module named 'cv2'`, hãy dùng `.venv` hoặc `py -3.13` như bên dưới.

## Chạy chương trình

Sau khi kích hoạt venv:

```powershell
.\.venv\Scripts\Activate.ps1
python main.py
```

Hoặc không cần activate:

```powershell
.\.venv\Scripts\python.exe main.py
```

Trên Windows (Python 3.13 toàn hệ thống):

```powershell
py -3.13 main.py
```

Nếu gặp lỗi hiển thị tiếng Việt trên terminal Windows:

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe main.py
```

---

## Cấu trúc thư mục

```
.
├── main.py              # Toàn bộ mã nguồn
├── requirements.txt     # Thư viện phụ thuộc
├── README.md
└── image/               # Ảnh đầu vào + kết quả
    ├── image1.jpg
    ├── image2.png
    ├── ...
    ├── grayscale/
    │   └── result.jpg
    ├── threshold/
    │   └── result.jpg
    ├── histogram/
    │   └── histogram.png
    ├── mean_filter/
    │   └── result.jpg
    ├── gaussian_filter/
    │   └── result.jpg
    └── sobel/
        └── result.jpg
```

**Lưu ý:** Đặt ảnh cần xử lý vào thư mục `image/`. Chương trình **không ghi đè** ảnh gốc; kết quả lưu vào thư mục con tương ứng.

Định dạng ảnh hỗ trợ: `.jpg`, `.jpeg`, `.png`, `.bmp`

---

## Hướng dẫn sử dụng

### 1. Chọn ảnh

Chương trình tự quét thư mục `image/` và hiển thị menu:

```
===== DANH SÁCH ẢNH =====

1. image1.jpg
2. image2.png
3. image3.jpeg

0. Thoát chương trình
Chọn ảnh: 2
```

### 2. Chọn thuật toán

```
===== THUẬT TOÁN =====

1. Chuyển ảnh xám (Grayscale)
2. Nhị phân hóa (Thresholding)
3. Biểu đồ histogram (Histogram)
4. Lọc trung bình (Mean Filter)
5. Lọc Gaussian (Gaussian Filter)
6. Phát hiện biên Sobel (Sobel Edge Detection)
7. Thoát chương trình
8. Chọn ảnh khác
Chọn thuật toán: 1
```

Sau khi xử lý xong, chương trình **quay lại menu thuật toán** để chạy tiếp thuật toán khác trên cùng ảnh.

- **7** hoặc **0** (trong menu thuật toán): thoát chương trình  
- **8**: quay lại chọn ảnh khác  
- **0** (trong menu ảnh): thoát chương trình  

### 3. Kết quả

Mỗi lần xử lý thành công, đường dẫn file kết quả được in ra terminal, ví dụ:

```
Đã lưu: D:\...\project\image\grayscale\result.jpg
```

| Thuật toán | File kết quả |
|------------|--------------|
| Grayscale | `image/grayscale/result.jpg` |
| Thresholding | `image/threshold/result.jpg` |
| Histogram | `image/histogram/histogram.png` |
| Mean Filter | `image/mean_filter/result.jpg` |
| Gaussian Filter | `image/gaussian_filter/result.jpg` |
| Sobel Edge Detection | `image/sobel/result.jpg` |

---

## Các thuật toán

| STT | Tên | Mô tả ngắn |
|-----|-----|------------|
| 1 | **Grayscale** | Chuyển ảnh màu sang xám: `Gray = 0.299×R + 0.587×G + 0.114×B` |
| 2 | **Thresholding** | Nhị phân hóa: pixel > 128 → 255, ngược lại → 0 |
| 3 | **Histogram** | Đếm phân bố mức xám 0–255, vẽ biểu đồ bằng Matplotlib |
| 4 | **Mean Filter** | Làm mịn bằng tích chập kernel trung bình 3×3 |
| 5 | **Gaussian Filter** | Làm mịn bằng kernel Gaussian 5×5 (σ = 1.0) |
| 6 | **Sobel Edge Detection** | Phát hiện biên: độ lớn gradient √(Gx² + Gy²) |

---

## Thư viện sử dụng

| Thư viện | Vai trò |
|----------|---------|
| **OpenCV** | Đọc / ghi file ảnh (`imread`, `imwrite`) |
| **NumPy** | Ma trận ảnh, tính toán thuật toán |
| **Matplotlib** | Vẽ và lưu biểu đồ histogram |

---

## Xử lý lỗi

Chương trình thông báo khi:

- Không có ảnh nào trong thư mục `image/`
- Nhập sai số (không phải số nguyên hoặc ngoài phạm vi menu)
- Không đọc được file ảnh đã chọn
- Không lưu được file kết quả

---

## Máy khác “không chạy được” / treo sau khi chọn thuật toán?

Thường **không phải lỗi**, mà do một trong các nguyên nhân sau:

| Triệu chứng | Nguyên nhân | Cách xử lý |
|-------------|-------------|------------|
| Menu không có dấu tiếng Việt | Terminal Windows chưa UTF-8 | Chạy `chcp 65001` hoặc dùng bản `main.py` mới (tự cấu hình UTF-8) |
| Dừng ở `Chọn thuật toán:` | Chưa nhập số + **Enter** | Gõ `1` rồi Enter (không chỉ gõ rồi đứng im) |
| Không in gì sau khi chọn 4/5/6 | **Ảnh quá lớn** (vd. 4000×2000), xử lý rất lâu | Đợi thêm; bản mới in `Đang chạy: ...` và cảnh báo ảnh lớn. Thử ảnh nhỏ trước, hoặc chọn 1/2/3 |
| Lỗi Histogram | Matplotlib cần GUI | Bản mới dùng backend `Agg` (không cần màn hình) |
| `No module named 'cv2'` | Sai Python / chưa cài thư viện | Dùng `.venv`: `pip install -r requirements.txt` |

**Kiểm tra nhanh trên máy mới:**

```powershell
cd F:\do-an\project\my-project
.\.venv\Scripts\python.exe -c "import cv2, numpy, matplotlib; print('OK')"
.\.venv\Scripts\python.exe main.py
```

Chọn ảnh → gõ `1` → Enter. Nếu thấy `Đã xử lý thành công...` và file `image/grayscale/result.jpg` thì môi trường đúng.

---

## Tùy chỉnh (trong `main.py`)

Có thể chỉnh các hằng số đầu file:

```python
THRESHOLD_VALUE = 128          # Ngưỡng nhị phân hóa
MEAN_KERNEL_SIZE = 3           # Kích thước kernel Mean Filter
GAUSSIAN_KERNEL_SIZE = 5       # Kích thước kernel Gaussian
GAUSSIAN_SIGMA = 1.0           # Độ lệch chuẩn Gaussian
MAX_CANH_XU_LY = 1280          # Tự thu nhỏ ảnh lớn (0 = tắt, xử lý full)
```

Ảnh gốc 4000×2000 trên máy chậm có thể mất vài phút nếu `MAX_CANH_XU_LY = 0`. Mặc định **1280** giúp máy yếu chạy gần như ngay; file kết quả theo kích thước đã thu nhỏ.

---

## Giấy phép & ghi chú

Dự án học tập / đồ án. Mã nguồn tập trung trong một file `main.py` để dễ đọc, chạy thử và báo cáo.
