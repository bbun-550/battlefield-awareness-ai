import sys
import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO
from PyQt5.QtWidgets import QApplication  # 화면 해상도 감지를 위해 import
import random

class ScreenDetector:
    """
    전체 화면을 실시간으로 캡처하고 YOLO 객체 탐지를 수행하는 클래스
    """
    def __init__(self, model_path=r'{수정 : MODEL_PATH'):
        """
        초기화 함수: 화면 해상도를 설정하고 YOLO 모델을 로드합니다.
        """
        # 1. 화면 해상도 감지 (PyQt5 사용)
        # QApplication 인스턴스가 있어야 화면 정보를 가져올 수 있습니다.
        # 이미 QApplication이 실행 중이면 기존 것을 사용하고, 아니면 새로 생성합니다.
        self.app = QApplication.instance() or QApplication(sys.argv)

        # 주 모니터(primaryScreen)의 geometry(정보) 가져오기
        geometry = self.app.primaryScreen().geometry()

        self.screen_width = geometry.width()
        self.screen_height = geometry.height()

        print(f"감지된 화면 해상도: {self.screen_width}x{self.screen_height}")

        # 2. mss (화면 캡처) 설정
        # 캡처할 영역을 전체 화면 크기로 지정
        self.monitor = {"top": 0, "left": 0, "width": self.screen_width, "height": self.screen_height}
        self.sct = mss()

        # 3. YOLO 모델 로드
        # model_path: 사용할 YOLO 모델 파일 (.pt)
        # 'yolov8n.pt'는 가장 작고 빠른 모델입니다.
        try:
            self.model = YOLO(model_path)
            # 모델의 클래스 이름들 (예: '{0: 'blue', 1: 'car', 2: 'red', 3: 'rock', 4: 'tank'}' 등)
            self.class_names = self.model.names
            print(self.class_names)
        except Exception as e:
            print(f"오류: YOLO 모델 로드 실패 ({model_path})")
            print(e)
            sys.exit(1)

        self.line_width = 3
        self.font_size = 1
        self.show_conf = False # (plot이 아니므로 conf=False가 아님)
        self.color_map = {

            0: (255, 0, 0),    # 파란색
            1: (168, 168, 168), # 회색
            2: (0, 0, 255),    # 빨간색
            3: (168, 168, 168), # 회색
            4: (0, 0, 255)     # 빨간색
        }

    def run_detection(self):
        """
        실시간 객체 탐지를 시작합니다. (OpenCV로 직접 그리기)
        """
        print("객체 탐지를 시작합니다. 종료하려면 'q' 키를 누르세요.")

        # ▼▼▼ 왼쪽 상단 텍스트 표시 위치 변수 ▼▼▼
        summary_start_x = 20
        summary_start_y = 40
        summary_line_height = 35 # 줄 간격
        summary_font_size = 1.0  # 요약 텍스트 크기
        summary_thickness = 2
        summary_header_text = "-Summary-"
        header_color = (255, 255, 0) # 헤더 색상 (예: 노란색)
        # ▲▲▲ 텍스트 위치 변수 끝 ▲▲▲

        while True:
            # 매 프레임마다 클래스별 카운터 초기화
            class_counter = {}

            try:
                # 1. 화면 캡처
                img_mss = self.sct.grab(self.monitor)
                img = np.array(img_mss)
                annotated_frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR) # 원본 이미지

                # 2. YOLO 객체 탐지 수행
                results = self.model(annotated_frame, verbose=False, conf=0.5, iou=0.38)

                # 3. 탐지 결과 수동 시각화 (plot() 대신)
                for box in results[0].boxes:
                    # 3-1. 정보 추출
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    # 3-2. 색상 가져오기
                    color = self.color_map.get(cls_id, (192, 192, 192))

                    # 3-3a. 카운터 업데이트
                    current_count = class_counter.get(cls_id, 0) + 1
                    class_counter[cls_id] = current_count

                    # 3-3b. 라벨 텍스트 생성 (예: "car 1")
                    label = f"{self.class_names[cls_id]} {current_count}"

                    if self.show_conf:
                        label += f" ({confidence:.2f})"

                    # 3-4. 바운딩 박스 그리기
                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        self.line_width
                    )

                    # 3-5. 라벨 텍스트 그리기
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, self.font_size, 2)
                    cv2.rectangle(annotated_frame, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_DUPLEX,
                        self.font_size,
                        (255, 255, 255),
                        thickness=2
                    )

                # ▼▼▼ 4. 화면 왼쪽 위에 클래스별 총 감지 수 표시 (새로 추가된 부분) ▼▼▼

                # 텍스트가 표시될 y 좌표
                summary_item_count = len(class_counter) + 1
                if summary_item_count > 1: # 헤더 외에 표시할 내용이 있을 때
                    overlay = annotated_frame.copy()
                    cv2.rectangle(overlay,
                                  (summary_start_x - 10, summary_start_y - summary_line_height + 10),
                                  (summary_start_x + 200, summary_start_y + (summary_item_count * summary_line_height) - 15),
                                  (0, 0, 0), -1)
                    alpha = 0.6
                    annotated_frame = cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0)


                # 4-1. 고정된 헤더("Summary ----") 표시
                current_y = summary_start_y
                cv2.putText(
                    annotated_frame,
                    summary_header_text,
                    (summary_start_x, current_y),
                    cv2.FONT_HERSHEY_DUPLEX,
                    summary_font_size,
                    header_color, # 헤더 색상
                    thickness=summary_thickness
                )

                # 4-2. 헤더 아래에 클래스별 총 감지 수 표시
                current_y += summary_line_height # 다음 줄로 y좌표 이동

                for cls_id in sorted(class_counter.keys()):
                    count = class_counter[cls_id]
                    class_name = self.class_names[cls_id]
                    text = f"{class_name}: {count}" # 예: "car: 3"

                    # 해당 클래스의 색상 가져오기
                    color = self.color_map.get(cls_id, (255, 255, 255))

                    cv2.putText(
                        annotated_frame,
                        text,
                        (summary_start_x, current_y),
                        cv2.FONT_HERSHEY_DUPLEX,
                        summary_font_size,
                        color, # 클래스별 색상으로 표시
                        thickness=summary_thickness
                    )
                    current_y += summary_line_height # 다음 줄로 y좌표 이동

                # ▲▲▲ 감지 수 표시 완료 ▲▲▲


                # 5. 결과 창 표시
                cv2.imshow("YOLO Screen Detection", annotated_frame)

                # 6. 종료 조건
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except Exception as e:
                print(f"탐지 중 오류 발생: {e}")
                break
            except KeyboardInterrupt:
                print("\n사용자에 의해 탐지 중지됨.")
                break

    def close(self):
        """
        사용한 리소스를 정리합니다. (OpenCV 창 닫기)
        """
        print("프로그램을 종료합니다.")
        cv2.destroyAllWindows()


# 이 스크립트가 메인으로 실행될 때만 아래 코드를 실행
if __name__ == "__main__":

    # 사용할 YOLO 모델 파일 경로
    # 'yolov8n.pt' (Nano), 'yolov8s.pt' (Small), 'yolov8m.pt' (Medium) 등
    # 파일이 없으면 ultralytics가 자동으로 다운로드합니다.
    MODEL_FILE_PATH = 'weights/5cls_v3_case1_best.pt'

    detector = None
    try:
        # 1. 탐지기 객체 생성
        detector = ScreenDetector(model_path=MODEL_FILE_PATH)

        # 2. 탐지 실행
        detector.run_detection()

    except Exception as e:
        print(f"프로그램 실행 중 오류: {e}")
    finally:
        # 3. 프로그램 종료 시 항상 close()를 호출하여 리소스 정리
        if detector:
            detector.close()