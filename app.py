from ultralytics import YOLO
import cv2
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Request

model = YOLO('best.pt')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_POS_MSEC, 100)

# FastAPI 앱 초기화
app = FastAPI()

# 템플릿 디렉토리 설정
templates = Jinja2Templates(directory="templates")

# 정적 파일 제공 설정 (CSS 파일)
app.mount("/static", StaticFiles(directory="static"), name="static")


def generate_frames():
    while True:
        # 웹캠에서 프레임 읽기
        success, frame = cap.read()
        if not success:
            break

        # YOLOv8로 객체 탐지
        results = model(frame)
        
        # 결과를 시각화
        annotated_frame = results[0].plot()

        # 프레임을 JPEG로 인코딩
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        # 프레임을 클라이언트에 전달
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video_feed/")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)