# QUY TẮC SỐNG CÒN CỦA DỰ ÁN (PROJECT GOLDEN RULES)

## 1. Tuyệt đối không để xảy ra Look-ahead Bias (Rò rỉ dữ liệu tương lai)
> **Cảnh báo từ đề bài (Leakage Gate):** *"Any use of future information caps Parts 4 and 5 together at 30% of their points."*

- **Nguyên tắc Point-in-Time (Tính đến thời điểm $q$):**
  - Tại bất kỳ quý $q$ nào, mọi đặc trưng (features), tỷ số tài chính, biến trễ (lags), tốc độ tăng trưởng (growth rates), hay thống kê trượt (rolling statistics) của ngân hàng $i$ **chỉ được phép sử dụng thông tin từ quý $q$ trở về trước** ($\le q$).
  - Khi xử lý ngân hàng ngừng hoạt động (exited banks: phá sản hoặc sáp nhập), trạng thái và nhãn $y_{i,q} = \mathbb{1}\{\text{bank } i \text{ fails in } (q, q+4\text{ quarters}]\}$ chỉ dùng để giám sát có điều kiện đúng theo cửa sổ tương lai 4 quý, tuyệt đối không đưa thông tin từ bảng `failures` hay `institutions` phát sinh sau quý $q$ vào tập biến đầu vào $X_{i,q}$.

- **Nguyên tắc Fit trên tập Training / Development:**
  - Mọi bước tính toán thống kê tổng hợp (mean, median, quantiles, winsorization/clipping thresholds), xử lý khuyết thiếu (imputation), chuẩn hóa (scaling), mã hóa biến phân loại (encoding), chọn lọc đặc trưng (feature selection), tinh chỉnh siêu tham số (hyperparameter tuning) và xác định ngưỡng cảnh báo (alarm threshold) **chỉ được phép `fit` trên tập Training (hoặc bên trong từng fold quá khứ của Time-Series Split trên tập Development)**.
  - Tập Validation và tập Test chỉ được áp dụng phép biến đổi (`transform`) từ các tham số đã học trên tập Training trước đó.

## 2. Phân chia dữ liệu theo thời gian (Temporal Split & Purging Gap)
- Chỉ sử dụng các quý tính đến **2025Q2** cho việc huấn luyện và đánh giá mô hình (vì các quý sau đó chưa đủ cửa sổ 4 quý để xác định nhãn hoàn chỉnh).
- Tập **Test** phải nằm hoàn toàn **sau** tập **Development**, cách nhau một khoảng đệm (**gap**) tối thiểu **4 quý** (purging/embargo gap) để ngăn chặn rò rỉ nhãn (do nhãn tại quý $q$ nhìn trước 4 quý: $(q, q+4]$).
- Tập **Test** phải chứa ít nhất **50 vụ phá sản** và **chỉ được đánh giá đúng 1 lần duy nhất (evaluate once)** ở bước cuối cùng.

## 3. Quản lý dữ liệu & Tái lập (Reproducibility)
- **Không commit dữ liệu lên GitHub:** Tuyệt đối không đẩy `data_raw/`, `data_processed/`, `*.parquet`, `*.csv` lên kho chứa Git.
- Cố định `random_state` / `seed` ở mọi bước lấy mẫu và huấn luyện mô hình để đảm bảo kết quả chạy lại khớp 100% từ `README.md`.
