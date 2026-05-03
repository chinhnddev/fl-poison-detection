# Verification Report: SPCHM-Trust Implementation

## 1. Pipeline thực tế

Triển khai hiện tại bám sát quy trình SPCHM-Trust theo hướng federated object detection. Trên server, `aggregate_fit` thu các delta từ client và chuyển sang `run_spchm_trust_round` khi chế độ phòng thủ SPCHM-Trust được bật ([federated/server_app.py](federated/server_app.py#L208), [federated/server_app.py](federated/server_app.py#L231)). Trong vòng xử lý này, server nạp proxy set, sinh reference predictions từ global model, dựng client model bằng `global + delta`, chạy suy luận trên cùng proxy set, sau đó tổng hợp độ lệch theo ảnh và theo client để tạo trust score và trust weight ([defense/spchm_trust.py](defense/spchm_trust.py#L309), [defense/spchm_trust.py](defense/spchm_trust.py#L407), [defense/spchm_trust.py](defense/spchm_trust.py#L445)).

Sau bước đánh giá độ nhất quán, code thực hiện MAD normalization, tính trọng số tin cậy, tổng hợp delta có trọng số và cập nhật mô hình toàn cục bằng phép cộng `global + aggregated_delta` ([defense/spchm_trust.py](defense/spchm_trust.py#L236), [defense/spchm_trust.py](defense/spchm_trust.py#L248), [defense/spchm_trust.py](defense/spchm_trust.py#L514), [federated/server_app.py](federated/server_app.py#L273)). Diagnostics per-client được lưu vào JSONL khi `round_stats_out` được bật, do đó pipeline không chỉ có hành vi thực thi mà còn có vết kiểm chứng cho từng round ([federated/server_app.py](federated/server_app.py#L286)).

## 2. Checklist đối chiếu requirement và code

| Mục | Requirement | Code hiện tại làm gì | Đánh giá | Gợi ý sửa |
|---|---|---|---|---|
| a | Client gửi delta; server tái tạo client model bằng `global + delta` | Client tính `delta = local - global` và return delta; server reconstructs client params bằng `global_params + delta` | Khớp requirement | Không cần |
| b | Load proxy set sạch, sinh reference predictions cố định trong round | Server gọi `load_dataset_images(...)`, rồi `predictor.load_parameters(global_params)` và `predictor.predict(...)` để tạo `reference_predictions` | Khớp requirement | Xác minh thêm resolution đường dẫn proxy/YAML giữa môi trường chạy |
| c | Hungarian matching, IoU gating, cost = `(1-IoU)+λ_cls*1[class_mismatch]` | `score_prediction_consistency` tạo cost matrix, gọi `linear_sum_assignment`, rồi loại match nếu IoU dưới ngưỡng; tính `d_box`, `d_cls`, `r_miss`, `r_ghost` | Khớp requirement | Không cần |
| d | Aggregate anomaly score từ 4 metric với hệ số λ | `aggregate_client_consistency` trung bình per-image metrics và `compute_composite_score` cộng tuyến tính theo `lambda_box`, `lambda_cls`, `lambda_miss`, `lambda_ghost` | Khớp requirement | Không cần |
| e | MAD normalization: `z_i = max(0,(s_i-median)/(1.4826*MAD+eps))` | `mad_normalize_scores` tính median, MAD và áp dụng công thức chuẩn hóa với `eps` | Khớp requirement | Không cần |
| f | Trust-only/proxy-only weighting; fallback FedAvg khi tổng trọng số gần 0 | `compute_trust_weights` dùng `exp(-tau*z_i) * cosine_root`, nhân `num_examples`, normalize và fallback theo số mẫu nếu weight sum gần 0 | Khớp requirement | Có thể làm rõ nhánh proxy-only nếu cần văn bản mô tả tường minh hơn |
| g | `Δagg = Σ w_i Δ_i`, rồi `θ_global ← θ_global + Δagg` | `aggregate_delta_with_weights` tạo weighted sum; server cộng kết quả vào global weights và lưu checkpoint | Khớp requirement | Không cần |
| h | Ghi diagnostics cho từng client: metrics trung gian, z_i, trust_i, w_i | `run_spchm_trust_round` tạo `client_diagnostics`; server ghi JSONL gồm `d_box`, `d_cls`, `r_miss`, `r_ghost`, `s_i`, `z_i`, `trust_raw`, `trust_weight`, `fallback_used`, ... | Khớp requirement | Nếu cần quan sát tức thời, bổ sung `logger.info` per-client là tùy chọn |

## 3. Nhận xét theo hướng học thuật

1. Mức độ tương thích giữa triển khai và requirement là cao: các bước cốt lõi của SPCHM-Trust đã hiện diện trong code và có thể truy vết bằng tên hàm, tham số, và luồng dữ liệu.
2. Không thấy dấu hiệu thay đổi bản chất thuật toán ở lớp server-side aggregation; phần còn lại chủ yếu là vấn đề cấu hình, logging, và khả năng tái lập thực nghiệm.
3. Điểm cần xác minh thêm là cơ chế phân giải đường dẫn proxy/YAML trong môi trường chạy thực tế, vì đây là yếu tố hạ tầng có thể ảnh hưởng đến việc nạp proxy set chứ không phải logic phòng thủ SPCHM-Trust.
4. Về tổng thể, implementation có thể được xem là đáp ứng đúng khung phương pháp của luận văn, với các sai khác tiềm năng nằm ở mức triển khai kỹ thuật thay vì sai lệch thuật toán cốt lõi.

## 4. Các file tham chiếu chính

- [federated/server_app.py](federated/server_app.py)
- [federated/client_app.py](federated/client_app.py)
- [defense/spchm_trust.py](defense/spchm_trust.py)
- [config.spchm_trust.yaml](config.spchm_trust.yaml)
