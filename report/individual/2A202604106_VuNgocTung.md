# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202604106
- **Họ và tên:** Vũ Ngọc Tùng
- **SEED:** 4051

## 1. Bằng chứng

| Tệp | Có? |
|---|---|
| `day01_lab.ipynb` — notebook đã chạy, còn output | ✅ |
| `lab1_report.json` — do ô "Báo cáo cuối" sinh ra | ✅ |
| `hand_calc.jpg` — ảnh tính tay có các bước | X |

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

| Dự đoán | Tôi đoán | Thực tế | Vì sao lệch |
|---|---|---|---|
| `dl_curve` (MLP so với LogReg) | Val loss chững hoặc tăng do overfitting; MLP bằng hoặc kém LogReg | Val loss thấp nhất ở epoch 4/30 (vế overfitting đúng). MLP F1 **0.949** (recall 0.904), cao hơn LogReg **0.888** (recall 0.798) | Tôi gộp "overfit" với "F1 kém". Val loss tăng chỉ cho thấy mô hình quá tự tin ở các tin sai, còn F1 dùng ngưỡng 0.5 nên vẫn cao. LogReg tự code mới là bên underfit: recall trên chính tập train chỉ 0.81 |
| `dl_relu` (bỏ `nn.ReLU()`) | Hai lớp Linear thu về một lớp Linear, nên F1 gần LogReg (~0.888) | F1 không ReLU = **0.949**, giống hệt bản có ReLU | Lý thuyết đúng (hàm vẫn tuyến tính), nhưng cùng lớp hàm không có nghĩa cùng kết quả huấn luyện. Hai bên khác cách tối ưu: Adam + mini-batch 64 (30 epoch) so với gradient descent thuần lr=0.5 (300 epoch). Khoảng cách đến từ tối ưu hoá, không phải tính phi tuyến |
| `ml_lr` (lr = 50) | Loss giảm rất nhanh nhưng có thể dao động hoặc không hội tụ | Loss bật lên ~2.5 ở bước đầu, rơi về ~0.1 sau vài epoch, kết thúc **0.010**, không dao động (lr=0.001: 0.630; lr=0.5: 0.098) | Loss logistic là hàm lồi và mỗi từ xuất hiện trong rất ít tin nên gradient từng trọng số nhỏ, lr=50 chỉ như một bước vừa phải. Cú vọt 2.5 do bias: bước đầu `mean(error) ≈ 0.37`, nhân 50 làm b nhảy xuống ~−18, mô hình đoán toàn ham nên loss tạm tăng rồi tự điều chỉnh. Đây là loss train, chưa chứng minh tổng quát hoá tốt hơn |

## 4. Phân tích 50 case khó

Model tụt nhiều nhất: BoW + LogReg. F1 giảm từ 0.888 (test thường) xuống 0.378 trên 50 case khó. Accuracy còn 54%, precision 0.875 nhưng recall chỉ 0.241 (bắt được 7/29 spam).

Nhóm tụt nhiều nhất:

manipulation: đúng 1/8.
natural_spam: đúng 2/10.
obfuscation: đúng 2/6.
prompt_injection: đúng 3/6.
ham_looks_spam (10/10) và talks_about_spam (4/4) đúng hết, nhưng không phải vì mô hình hiểu ngữ cảnh. Bias học được là −3.01 nên mô hình nghiêng hẳn về ham, gặp gì lạ cũng đoán ham.

Case #21 (manipulation, thật = spam, đoán = ham): “Hi Mum, I dropped my phone in the toilet, this is my new number. Can you send £200 to my friend’s account? I’ll explain later x”

Không có từ khoá spam nào đủ mạnh, £200 có trọng số ≈ 0. Các từ trọng số lớn lại kéo về ham: i (−1.30), my (−0.70). Tổng điểm z = −3.51, p(spam) ≈ 0.03.
Cơ chế: BoW chỉ cộng trọng số từng từ nên không nhận ra ý đồ giả danh người thân để xin tiền. Spam lừa đảo viết bằng giọng đời thường, trùng từ vựng với ham.

Case #29 (obfuscation, thật = spam, đoán = ham): “C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd. Cl4im n0w at w1n-gift.c0m”

Hầu hết token bị thay số nên không có trong vocab. Chỉ còn 3 token nhận ra được: a (+0.51), £1000 (≈ 0), at (−0.35). Mô hình gần như không có feature spam nào, z = −2.85 chủ yếu đến từ bias, p(spam) ≈ 0.055.
Cơ chế: vocab đóng cứng từ tập train. Chữ bị biến dạng thành token lạ (OOV) và bị bỏ qua, nên chỉ cần đổi vài ký tự là qua mặt được.

Ghi chú về mức độ kiểm chứng: API key Gemini liên tục hết quota (lỗi 429, free tier 15 request/phút) nên phần LLM chưa chạy được. Kết quả 50 case khó của MLP, Embedding và LLM cũng chưa được chạy trong notebook, nên kết luận “BoW + LogReg tụt nhiều nhất” mới được kiểm chứng trên BoW + LogReg, chưa so sánh được với các model còn lại. Dự kiến (chưa đo): các model dùng embedding sẽ tụt ít hơn nhưng vẫn yếu ở manipulation và natural_spam, còn LLM có thể yếu ở prompt_injection.

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
