# MNIST Classification - Dự án Minh chứng Đồ án 1

Dự án này là minh chứng thực hành (demo) cho học phần **Project 1 (Đồ án 1)** dưới sự hướng dẫn của thầy **Nguyễn Nhất Hải**. Dự án tích hợp và liên kết chặt chẽ cả 4 nội dung cốt lõi của môn học bao gồm: **Git**, **Docker**, **Python (AI)** và **Deep Learning**.

---

## Mục lục
1. [Giới thiệu dự án & Dữ liệu](#1-giới-thiệu-dự-án--dữ-liệu)
2. [Kiến trúc mạng Nơ-ron (MLP)](#2-kiến-trúc-mạng-nơ-ron-mlp)
3. [Giải thích chi tiết mã nguồn `train.py`](#3-giải-thích-chi-tiết-mã-nguồn-trainpy)
4. [Đóng gói ứng dụng với Docker](#4-đóng-gói-ứng-dụng-với-docker)
5. [Hướng dẫn chạy chương trình](#5-hướng-dẫn-chạy-chương-trình)
    * [Cách chạy bằng Dòng lệnh (CLI)](#cách-chạy-bằng-dòng-lệnh-cli---khuyên-dùng)
    * [Cách chạy bằng Giao diện Docker Desktop (GUI)](#cách-chạy-bằng-giao-diện-docker-desktop-gui)
6. [Quản lý mã nguồn với Git](#6-quản-lý-mã-nguồn-với-git)
7. [Kết quả và Minh chứng thu được](#7-kết-quả-và-minh-chứng-thu-được)

---

## 1. Giới thiệu dự án & Dữ liệu
Dự án thực hiện bài toán phân loại chữ số viết tay từ bộ dữ liệu chuẩn **MNIST**.
*   **Dữ liệu MNIST:** Là bộ dữ liệu kinh điển gồm các ảnh đen trắng (thang độ xám - grayscale) kích thước $28 \times 28$ pixel. Bộ dữ liệu chứa $60,000$ ảnh huấn luyện (train set) và $10,000$ ảnh kiểm thử (test set) tương ứng với 10 nhãn chữ số từ `0` đến `9`.
*   **Mục tiêu:** Huấn luyện một mô hình học sâu nhận dạng chính xác chữ số tương ứng từ ảnh đầu vào.

---

## 2. Kiến trúc mạng Nơ-ron (MLP)
Mô hình được xây dựng dưới dạng một mạng nơ-ron đa tầng (**Multi-Layer Perceptron - MLP**) với cấu trúc cụ thể:
*   **Lớp đầu vào (Input Layer):** Nhận ảnh $28 \times 28$ pixel, được là phẳng (flatten) thành một vector phẳng gồm $784$ phần tử đầu vào ($x \in \mathbb{R}^{784}$).
*   **Lớp ẩn (Hidden Layer):** Gồm $128$ nơ-ron liên kết đầy đủ (Fully Connected / Dense). Lớp này áp dụng hàm kích hoạt phi tuyến **ReLU** (Rectified Linear Unit): $f(z) = \max(0, z)$ giúp mô hình học được các đặc trưng phi tuyến và tăng tốc độ hội tụ đạo hàm.
*   **Lớp đầu ra (Output Layer):** Gồm $10$ nơ-ron tương ứng với xác suất dự đoán của 10 lớp chữ số từ `0` đến `9`.
*   **Hàm chi phí (Loss Function):** Sử dụng hàm **Entropy chéo (Cross-Entropy Loss)**, kết hợp sẵn phép tính Softmax để tối ưu hóa việc phân loại đa lớp.
*   **Thuật toán tối ưu (Optimizer):** Stochastic Gradient Descent (SGD) với xung lượng (momentum) là $0.9$, tốc độ học (learning rate) $\alpha = 0.01$.

---

## 3. Giải thích chi tiết mã nguồn `train.py`
Tệp [train.py](train.py) chịu trách nhiệm thực thi toàn bộ quy trình:

1.  **Chuẩn hóa dữ liệu (Preprocessing):** 
    Ảnh được chuyển đổi thành Tensor và chuẩn hóa thang màu theo trung bình $0.1307$ và độ lệch chuẩn $0.3081$ để tăng hiệu quả huấn luyện:
    ```python
    transforms.Normalize((0.1307,), (0.3081,))
    ```
2.  **Bộ nạp dữ liệu (DataLoader):**
    Sử dụng PyTorch DataLoader để chia dữ liệu thành từng lô nhỏ (batch size = 64) và xáo trộn (shuffle) ngẫu nhiên dữ liệu train để mô hình học khách quan.
3.  **Vòng lặp huấn luyện (Training Loop):**
    Chạy qua 5 epochs. Trong mỗi lô dữ liệu:
    *   **Forward pass:** Dự đoán đầu ra và tính sai số (Loss).
    *   **Zero gradients:** Xóa sạch các đạo hàm tích lũy cũ bằng `optimizer.zero_grad()`.
    *   **Backward pass:** Lan truyền ngược tính đạo hàm bằng `loss.backward()`.
    *   **Update weights:** Cập nhật tham số ($W, b$) bằng `optimizer.step()`.
4.  **Vẽ đồ thị (Visualization):**
    Tự động lưu lịch sử Loss và Accuracy của từng epoch. Vẽ 2 đồ thị biểu diễn sự giảm dần của Loss và tăng dần của Accuracy, lưu thành file ảnh `training_result.png`.

---

## 4. Đóng gói ứng dụng với Docker
Toàn bộ môi trường phát triển được đóng gói trong [Dockerfile](Dockerfile):
*   Sử dụng base image siêu nhẹ `python:3.9-slim`.
*   Cài đặt phiên bản CPU của PyTorch (`--index-url https://download.pytorch.org/whl/cpu`) giúp giảm dung lượng ảnh Docker từ hơn 2GB xuống còn 1.4GB, tiết kiệm băng thông và tài nguyên CPU chạy thử nghiệm.
*   Nhúng sẵn mã nguồn và thiết lập chạy mặc định tệp tin `train.py`.

---

## 5. Hướng dẫn chạy chương trình

### Cách chạy bằng Dòng lệnh (CLI) - KHUYÊN DÙNG
Cách này giúp bạn tự động cấu hình Volume để xuất tệp biểu đồ ra máy thật nhanh nhất.

1.  Mở PowerShell hoặc Command Prompt tại thư mục dự án `mnist_demo`.
2.  Xóa container cũ để tránh xung đột tên (nếu có):
    ```bash
    docker rm run-trainer
    ```
3.  Khởi chạy container và liên kết thư mục `output` ra ngoài máy thật:
    *   **Nếu dùng PowerShell (Windows):**
        ```powershell
        docker run --name run-trainer -v ${pwd}/output:/app/output mnist-trainer:1.0
        ```
    *   **Nếu dùng Command Prompt (CMD - Windows):**
        ```cmd
        docker run --name run-trainer -v %cd%/output:/app/output mnist-trainer:1.0
        ```
    *   **Nếu dùng Bash Shell (Linux/macOS):**
        ```bash
        docker run --name run-trainer -v $(pwd)/output:/app/output mnist-trainer:1.0
        ```

### Cách chạy bằng Giao diện Docker Desktop (GUI)
Nếu bạn muốn sử dụng giao diện trực quan của Docker Desktop:
1.  Bật Docker Desktop và tìm đến mục **Images** bên menu trái.
2.  Tìm ảnh có tên **`mnist-trainer`** (tag `1.0`), di chuột vào cột **Actions** bên phải và nhấp nút **Play** (Run).
3.  Trong cửa sổ hiển thị, nhấp mở rộng mục **Optional settings**:
    *   **Container name:** Điền `run-trainer`.
    *   **Volumes:**
        *   *Host path:* Chọn đường dẫn thư mục `mnist_demo/output` trên máy tính vật lý của bạn.
        *   *Container path:* Gõ chính xác `/app/output`.
4.  Nhấp nút **Run**.
5.  Bạn có thể nhấp vào tab **Containers** để xem trực quan dòng Log huấn luyện chạy thời gian thực.

---

## 6. Quản lý mã nguồn với Git
Mã nguồn được quản lý bằng Git với lịch sử commit được gộp sạch sẽ làm minh chứng báo cáo:
```bash
# Xem lich su commit rut gon
git log --oneline
```
Output thực tế hiển thị commit chính thức duy nhất:
```verbatim
* 11d5ac5 Final
```

---

## 7. Kết quả và Minh chứng thu được
Sau khi container kết thúc quá trình chạy:
*   Mô hình đạt độ chính xác trên tập kiểm tra độc lập (**Test Accuracy**) khoảng **97.5%**.
*   Tệp đồ thị biểu diễn kết quả **`training_result.png`** sẽ tự động xuất hiện tại thư mục `mnist_demo/output` trên máy tính của bạn như hình dưới đây:

*(Lưu ý: Biểu đồ thể hiện trực quan quá trình tối ưu hóa Loss giảm dần tiệm cận 0 và Accuracy tăng dần đạt trên 97% qua các epochs).*
