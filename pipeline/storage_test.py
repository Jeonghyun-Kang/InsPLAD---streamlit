import uuid
from PIL import Image
import storage

def run_test():
    print("AWS Storage 연결 테스트 시작...")

    try:
        dummy_orig = Image.new('RGB', (100, 100), color='red')
        dummy_heat = Image.new('RGB', (100, 100), color='green')
        dummy_over = Image.new('RGB', (100, 100), color='blue')

        class_name = "test-glass-insulator"
        anomaly_score = 0.9876
        is_anomaly = True
        report = "이것은 테스트 리포트입니다. AWS 업로드 확인용입니다."
        
        ragas_scores = {
            "faithfulness": 0.9,
            "answer_relevancy": 0.85
        }

        print("S3 및 DynamoDB 업로드 중...")
        
        record_id = storage.save_result(
            class_name=class_name,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            report=report,
            original_img=dummy_orig,
            heatmap_img=dummy_heat,
            overlay_img=dummy_over,
            ragas_scores=ragas_scores
        )

        print("-" * 30)
        print(f"테스트 성공!")
        print(f"생성된 Record ID: {record_id}")
        print(f"1. S3 확인: 'images/{record_id}/' 폴더와 'reports/{record_id}.txt' 확인")
        print(f"2. DynamoDB 확인: '{storage.table.table_name}' 테이블에 데이터 추가 확인")
        print("-" * 30)

    except Exception as e:
        print(f"테스트 실패: {e}")

if __name__ == "__main__":
    run_test()