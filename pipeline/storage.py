import boto3, io, uuid, os, time
from datetime import datetime
from PIL import Image
from dotenv import load_dotenv

load_dotenv()


s3 = boto3.client("s3", region_name="ap-northeast-2")
dynamodb = boto3.resource("dynamodb", region_name="ap-northeast-2")
table = dynamodb.Table("insplad_log")
BUCKET = "insplad-anomaly-detection-747456039784-ap-northeast-2-an"

def upload_image(img: Image.Image, key: str):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    s3.put_object(Bucket=BUCKET, Key=key, Body=buf, ContentType="image/png")
    return f"s3://{BUCKET}/{key}"

def save_result(class_name, anomaly_score, is_anomaly, report,
                original_img, heatmap_img, overlay_img, ragas_scores):
    record_id = str(uuid.uuid4())
    ts = int(time.time())

    orig_url   = upload_image(original_img,  f"images/{record_id}/original.png")
    heat_url   = upload_image(heatmap_img,   f"images/{record_id}/heatmap.png")
    overlay_url= upload_image(overlay_img,   f"images/{record_id}/overlay.png")

    # 리포트 텍스트 저장
    s3.put_object(Bucket=BUCKET, Key=f"reports/{record_id}.txt",
                  Body=report.encode(), ContentType="text/plain")

    # 메타데이터 → DynamoDB
    table.put_item(Item={
        "inspection_id": record_id,
        "timestamp": ts,
        "class_name": class_name,
        "anomaly_score": str(round(anomaly_score, 4)),
        "is_anomaly": is_anomaly,
        "s3_original": orig_url,
        "s3_heatmap": heat_url,
        "s3_overlay": overlay_url,
        "ragas_faithfulness": str(ragas_scores["faithfulness"]),
        "ragas_relevancy": str(ragas_scores["answer_relevancy"]),
    })
    return record_id

