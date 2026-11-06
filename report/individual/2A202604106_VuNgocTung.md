# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** [Điền MSSV]
- **Họ và tên:** [Điền họ tên]
- **SEED:** [Điền SEED trong notebook]

## 1. Bằng chứng

| Tệp | Có? |
|---|---|
| `day01_lab.ipynb` — notebook đã chạy, còn output | ✅ |
| `lab1_report.json` — do ô "Báo cáo cuối" sinh ra | ✅ |
| `hand_calc.jpg` — ảnh tính tay có các bước | ✅ |

## 2. Kết quả chính

SEED=4051 · thử thách #0
Bộ kiểm tra: 10/10 đạt
  ✅ 1.1 vectorize
  ✅ 1.2 sigmoid / predict_proba
  ✅ 1.3 train_logreg
  ✅ 1.4 metrics
  ✅ 2.1 build_mlp
  ✅ 2.2 train_mlp
  ✅ 3.1–3.2 cosine / top_k_similar
  ✅ 3.3 knn_predict
  ✅ 4.2–4.3 prompt / parse_label
  ✅ 4.1 + 4.4 call_openai / llm_classify_batch

Dự đoán đã ghi:
  [16:36:07] ml_f1: F1 = 0.85, precision cao hơn recall vì mô hình dự đoán dương tính khá chính xác, hạn chế dự đoán nhầm mẫu âm tính thành dương tính. Tuy nhiên, mô hình vẫn có thể bỏ sót một số mẫu dương tính.
  [16:36:30] ml_lr: lr=0.001 thì loss giảm chậm, lr=50 thì loss có thể giảm rất nhanh nhưng có nguy cơ dao động hoặc không hội tụ.
  [16:38:09] dl_curve: 1. Val loss không giảm mãi, có thể chững lại hoặc tăng do overfitting. 2. MLP có thể bằng hoặc kém Logistic Regression về F1 do dễ overfitting và cần nhiều dữ liệu hơn.
  [16:38:27] dl_relu: Nếu bỏ `nn.ReLU()`, hai lớp Linear liên tiếp tương đương với một lớp Linear (Logistic Regression); F1 dự kiến tương đương hoặc gần Logistic Regression vì mạng mất khả năng học quan hệ phi tuyến.
  [16:41:21] emb_search: spam – vì nội dung có dấu hiệu lừa đảo như “selected to receive”, “complimentary holiday”, “reply now to claim”, dù không chứa từ khóa spam quen thuộc.
  [16:42:58] emb_tricky: Encoder + LogReg (Phần 3b) đoán đúng nhiều hơn vì Embedding nắm bắt ngữ nghĩa và ngữ cảnh tốt hơn BoW, có thể nhận diện spam dù né từ khóa hoặc phân biệt từ khóa spam trong ngữ cảnh vô hại.
  [16:43:13] llm_models: (1) Gemini 3.8 Flash dự kiến F1 cao hơn Gemini 3.5 Flash Lite khoảng 1–3%; (2) Model lớn đắt hơn khoảng 2–3 lần; (3) LLM chưa chắc thắng Embedding + LogReg (3b) vì embedding đã biểu diễn ngữ nghĩa tốt.
Thiếu dự đoán: ['llm_fewshot', 'tradeoff', 'hard_eval']

                                     f1  f1_SUB
1. BoW + LogReg (tự code)        0.888   0.857
2. BoW + MLP (deep learning)     0.949   0.947
3a. Encoder (MiniLM) + kNN       0.912   0.974
3b. Encoder (MiniLM) + LogReg    0.976   0.974
3c. Encoder (MiniLM) + MLP head  0.976   0.974

💾 Đã lưu lab1_report.json (chỉ chứa SEED, không chứa MSSV)

## 3. Dự đoán và thực tế

| Dự đoán | Kết quả thực tế | Vì sao lệch |
|---|---|---|
| Encoder + LogReg (Phần 3b) đoán đúng nhiều hơn vì Embedding nắm bắt ngữ nghĩa và ngữ cảnh tốt hơn BoW, có thể nhận diện spam dù né từ khóa hoặc phân biệt từ khóa spam trong ngữ cảnh vô hại. | BoW đúng 5/6 · Encoder+LogReg đúng 6/6 |

## 4. Phân tích 50 case khó
Nhận xét: Tập dữ liệu khó giúp kiểm tra khả năng nhận diện spam trong những tình huống phức tạp mà tập test thông thường có thể chưa thể hiện đầy đủ.

## 5. Thử thách cá nhân
- Không thực hiện
## 6. Phương án cho sản phẩm thật

Với yêu cầu 10 triệu SMS/ngày và p95 dưới 50 ms/tin, nên ưu tiên mô hình có độ trễ dưới 50 ms và throughput lớn hơn 116 tin/giây. Có thể kết hợp Logistic Regression xử lý tin chắc chắn và LLM xử lý tin lưỡng lự có xác suất từ 0.3 đến 0.7.

Cách này giúp giảm chi phí và độ trễ, đồng thời tận dụng khả năng hiểu ngữ cảnh của LLM. Đánh đổi là hệ thống phức tạp hơn và cần kiểm soát thời gian xử lý của LLM.

Chi phí LLM mỗi tháng = USD/1k tin × 300.000 (với 300 triệu SMS/tháng).

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| Dự đoán model | So sánh khả năng của BoW, MLP, Embedding và LLM trong bài toán lọc spam. | Phân tích ưu, nhược điểm và dự đoán chất lượng từng mô hình. | Đối chiếu với kết quả chạy notebook. |
| Case khó | Tạo ít nhất 5 case mới để kiểm tra khả năng nhận diện spam. | Đề xuất 6 case thuộc các nhóm natural_spam, manipulation, prompt_injection và transactional_ham. | Chạy lại các model và kiểm tra dự đoán sai. |
| Chọn mô hình | Chọn phương án lọc 10 triệu SMS/ngày với p95 dưới 50 ms. | Đề xuất chọn mô hình nhanh hoặc kết hợp Router với LLM. | Đối chiếu F1, throughput, p95 và chi phí. |
| Router | Thiết kế hệ thống kết hợp Logistic Regression và LLM. | Logistic Regression xử lý tin chắc chắn, LLM xử lý tin lưỡng lự. | Xem xét đánh đổi giữa tốc độ, chất lượng và chi phí. |

**Điều đã học:**

Mô hình phức tạp hơn không phải lúc nào cũng cho kết quả tốt hơn. Cần đánh giá mô hình bằng dữ liệu thực tế và nhiều chỉ số như F1, độ trễ, throughput và chi phí. Ngoài ra, có thể kết hợp nhiều mô hình bằng Router để cân bằng tốc độ và chất lượng.
