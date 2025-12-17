import base64
import os
import time
import threading
import numpy as np
import cv2
import torch
import asyncio
from datetime import datetime
from collections import deque
from contextlib import asynccontextmanager

# FastAPI 관련 임포트
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from ultralytics import YOLO
import google.generativeai as genai
from google.generativeai import GenerativeModel
from components.OmniSR import OmniSR

# =====================
# 설정 & 경로
# =====================
CROP_SAVE_DIR = "./input"
SR_SAVE_DIR = "./output"
os.makedirs(CROP_SAVE_DIR, exist_ok=True)
os.makedirs(SR_SAVE_DIR, exist_ok=True)

# =====================
# 전역 변수
# =====================
model = None      # YOLO
sr_model = None   # SR Model
llm = None        # Gemini
sr_queue = deque()
sr_done = set()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# 모델 로드 및 SR 워커 함수
# =====================
def load_models():
    global model, sr_model, llm
    model = YOLO("5cls_v7.pt")
    genai.configure(api_key="AIzaSyC7qizhUh1MTAo1sCCtteWv0_SKHgcAlt4")
    llm = GenerativeModel("gemini-2.0-flash")

    weights_path = "./weights/epoch994_OmniSR.pth"
    model_params = {
        "upsampling": 4, "res_num": 5, "block_num": 1, "bias": True,
        "block_script_name": "OSA", "block_class_name": "OSA_Block",
        "window_size": 8, "pe": True, "ffn_bias": True
    }
    sr_model = OmniSR(num_in_ch=3, num_out_ch=3, num_feat=64, **model_params)

    # weights_only=False 경고 회피를 위해 safe globals 설정이 복잡하므로
    # 경고는 뜨지만 기능상 문제는 없음
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint["params"] if "params" in checkpoint else checkpoint

    clean_state = {k.replace("module.", ""): v for k, v in state_dict.items()
                   if "total_ops" not in k and "total_params" not in k}

    sr_model.load_state_dict(clean_state, strict=True)
    sr_model.to(device)
    sr_model.eval()

    # GPU Warmup
    dummy = torch.zeros(1, 3, 64, 64).to(device)
    sr_model(dummy)
    print("[System] All Models Loaded & GPU Warmed Up!")

def process_sr_image(img_path, save_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None: return False
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = sr_model(img_tensor)

    output = output.squeeze().clamp_(0, 1).cpu().numpy()
    output = np.transpose(output, (1, 2, 0))
    output = output[:, :, [2, 1, 0]]
    output = (output * 255.0).round().astype(np.uint8)
    cv2.imwrite(save_path, output)
    return True

def sr_worker():
    # print("[System] SR Worker Thread Started...")
    while True:
        if not sr_queue:
            time.sleep(0.1)
            continue

        crop_name = sr_queue.popleft()
        if crop_name in sr_done: continue

        input_path = os.path.join(CROP_SAVE_DIR, crop_name)
        output_name = f"SR_x2_{crop_name}"
        output_path = os.path.join(SR_SAVE_DIR, output_name)

        try:
            if process_sr_image(input_path, output_path):
                sr_done.add(crop_name)
                print(f"[SR DONE] {output_name}")
            else:
                print(f"[SR FAIL] Read Error {crop_name}")
        except Exception as e:
            print(f"[SR ERROR] {e}")

# =====================
# FastAPI Lifespan (시작/종료 이벤트)
# =====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시 실행
    load_models()
    # SR 워커 스레드 시작
    threading.Thread(target=sr_worker, daemon=True).start()
    yield
    # 종료 시 실행 (필요하면 리소스 해제)
    print("[System] Shutting down...")

# FastAPI 앱 생성
app = FastAPI(lifespan=lifespan)

# 정적 파일 경로 마운트 (이미지 다운로드용)
app.mount("/sr", StaticFiles(directory=SR_SAVE_DIR), name="sr")

# =====================
# 데이터 모델 (Pydantic)
# =====================
class ExplainRequest(BaseModel):
    filename: str

# =====================
# API Endpoints
# =====================
def encode_image(img):
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")

# YOLO
@app.post("/modal")
async def modal(image: UploadFile = File(...)):
    # 1. 파일 읽기
    file_bytes = await image.read()
    npimg = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse(content={"error": "Invalid image"}, status_code=400)

    # 2. YOLO 및 저장 (CPU 바운드 작업이지만 빠르므로 그냥 진행)
    frame_encoded = encode_image(frame)
# 원본 프레임 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_filename = f"original_{timestamp}.jpg"
    original_path = os.path.join(CROP_SAVE_DIR, original_filename)
    cv2.imwrite(original_path, frame)
    print(f"[System] Original Frame Saved: {original_path}")
# [추가 1] 원본 프레임도 SR 작업 큐에 등록 (이제 원본도 화질 개선됨)
    sr_queue.append(original_filename)
    results = model(frame)[0]
    descriptions = []

    for idx, box in enumerate(results.boxes):
        # if int(box.cls[0]) != 4: continue # tank only
# [수정 2] 클래스 이름 가져오기
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]  # 예: "person", "car", "tank"
        x1, y1, x2, y2 = map(int, box.xyxy[0]) # 크롭하는 위치
        crop = frame[y1:y2, x1:x2]
        crop_encoded = encode_image(crop)

        crop_filename = f"crop_{timestamp}_{idx+1}.jpg"
        cv2.imwrite(os.path.join(CROP_SAVE_DIR, crop_filename), crop)

        # 큐 등록
        sr_queue.append(crop_filename)

        descriptions.append({
            "bbox": [x1, y1, x2, y2],
            "crop_image": crop_encoded,
            "crop_filename": crop_filename,
            "sr_filename": f"SR_x2_{crop_filename}",
            "description": "",
            # [수정 3] JSON에 클래스 이름 추가
            "class_name": class_name
        })

    return {
        "frame_image": frame_encoded,
        "original_filename": original_filename,       # <--- 클라이언트가 LLM 요청할 때 필요
        "original_sr_filename": f"SR_x2_{original_filename}", # <--- 클라이언트가 SR 요청할 때 필요
        "descriptions": descriptions
    }

# LLM
@app.post("/explain")
async def explain(req: ExplainRequest):
    # Pydantic이 자동으로 JSON body 파싱해줌
    file_path = os.path.join(CROP_SAVE_DIR, req.filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    img = cv2.imread(file_path)
    encoded = encode_image(img)

    try:
        prompt = " 너는 탐지된 객체를 통해 전장상황을 인식하여 판별하는 AI야. 탐지된 객체를 보고 현재 상황을 분석해서 문장형식으로 군대식으로 보고해. 마크다운형식쓰지마. 200자 이내."
# 이 이미지에 있는 주요 대상을 상세히 묘사해. 상황, 특징, 상태 등을 포함해서 설명해. 너는 현장에 있는 군인이야.

        # [핵심] Native Async Await (FastAPI의 장점)
        # 이제 Flask처럼 복잡한 설정 없이 그냥 쓰면 됨
        resp = await llm.generate_content_async([
            {"mime_type": "image/jpeg", "data": encoded},
            prompt
        ])
        text = getattr(resp, "text", "설명 생성 실패")
    except Exception as e:
        text = f"에러 발생: {str(e)}"

    return {"description": text}

# =====================
# 실행
# =====================
if __name__ == "__main__":
    # FastAPI는 uvicorn으로 실행합니다.
    # host 0.0.0.0으로 외부 접속 허용
    uvicorn.run(app, host="192.168.0.39", port=5001)
    # uvicorn.run(app, host="0.0.0.0", port=5001)
