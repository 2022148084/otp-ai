import modal

# [Helper] 빌드 타임 다운로드 (CPU 모드)
def download_models():
    import os
    os.system("mkdir -p /root/weights")
    os.system("wget -O /root/weights/network-sintel-final.pytorch http://content.sniklaus.com/github/pytorch-spynet/network-sintel-final.pytorch")
    os.system("wget -O /root/weights/insta_kakao_final_agent_model_pytorch.pth https://github.com/YeeDEA/kakao_insta_detector/raw/refs/heads/main/final_agent_model_pytorch.pth")
    
    from paddleocr import PaddleOCR
    print("⬇️ Downloading PaddleOCR models (CPU build)...")
    # 빌드 시점엔 CPU로 다운로드만 수행
    PaddleOCR(lang="korean", use_angle_cls=False, show_log=False, use_gpu=False)

# 1. 환경 설정
image = (
    modal.Image.from_registry("nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04", add_python="3.10")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "wget")
    .pip_install(
        "torch", 
        "torchvision",
        extra_options="--index-url https://download.pytorch.org/whl/cu118"
    )
    .pip_install(
        "paddlepaddle-gpu==2.6.1",
        "paddleocr==2.7.3",
        "opencv-python-headless",
        "numpy<2.0.0",
        "tqdm"
    )
    .run_function(download_models)
)

app = modal.App("kakao-ocr-unified")

@app.cls(image=image, gpu="T4", scaledown_window=100, min_containers=0)
class OCRService:
    
    @modal.enter()
    def initialize(self):
        import torch
        import torch.nn as nn
        from paddleocr import PaddleOCR
        import os
        from torchvision import models
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n👉 Current Computing Device: {self.device}\n")

        # --- 학습 코드와 동일한 구조로 모델 정의 ---
        model_path = "/root/weights/insta_kakao_final_agent_model_pytorch.pth"
        
        try:
            if os.path.exists(model_path):
                # 1. 뼈대 로드 (MobileNetV2)
                self.classifier = models.mobilenet_v2(weights=None)
                
                # 2. Classifier 구조 교체 (학습 코드와 100% 일치시킴)
                # 구조: Dropout(0.3) -> Linear(1280, 64) -> ReLU -> Linear(64, 1)
                self.classifier.classifier = nn.Sequential(
                    nn.Dropout(p=0.3),
                    nn.Linear(1280, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                )
                
                # 3. 가중치 로드
                state_dict = torch.load(model_path, map_location=self.device)
                self.classifier.load_state_dict(state_dict)
                self.classifier.to(self.device).eval()
                print("✅ Custom MobileNetV2 Loaded Successfully!")
            else:
                print("⚠️ 모델 파일이 없습니다.")
                self.classifier = None
                
        except Exception as e:
            print(f"⚠️ Failed to load classifier: {e}")
            self.classifier = None
        
        # --- SPyNet 정의 ---
        class SPyNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                class Preprocess(torch.nn.Module):
                    def forward(self, tenInput):
                        tenInput = tenInput.flip([1])
                        tenInput = tenInput - torch.tensor([0.485,0.456,0.406], dtype=tenInput.dtype, device=tenInput.device).view(1,3,1,1)
                        tenInput = tenInput * torch.tensor([1/0.229,1/0.224,1/0.225], dtype=tenInput.dtype, device=tenInput.device).view(1,3,1,1)
                        return tenInput
                class Basic(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.netBasic = torch.nn.Sequential(
                            torch.nn.Conv2d(8,32,7,1,3), torch.nn.ReLU(False),
                            torch.nn.Conv2d(32,64,7,1,3), torch.nn.ReLU(False),
                            torch.nn.Conv2d(64,32,7,1,3), torch.nn.ReLU(False),
                            torch.nn.Conv2d(32,16,7,1,3), torch.nn.ReLU(False),
                            torch.nn.Conv2d(16,2,7,1,3)
                        )
                    def forward(self, x): return self.netBasic(x)
                self.netPreprocess = Preprocess()
                self.netBasic = torch.nn.ModuleList([Basic() for _ in range(6)])
            def forward(self, tenOne, tenTwo):
                tenOne_list = [self.netPreprocess(tenOne)]
                tenTwo_list = [self.netPreprocess(tenTwo)]
                for _ in range(5):
                    if tenOne_list[0].shape[2] > 32:
                        tenOne_list.insert(0, torch.nn.functional.avg_pool2d(tenOne_list[0],2))
                        tenTwo_list.insert(0, torch.nn.functional.avg_pool2d(tenTwo_list[0],2))
                tenFlow = tenOne_list[0].new_zeros([1,2,tenOne_list[0].shape[2]//2, tenOne_list[0].shape[3]//2])
                for i in range(len(tenOne_list)):
                    tenUpsampled = torch.nn.functional.interpolate(tenFlow, scale_factor=2, mode="bilinear", align_corners=True) * 2.0
                    if tenUpsampled.shape[2] != tenOne_list[i].shape[2]: tenUpsampled = torch.nn.functional.pad(tenUpsampled,[0,0,0,1])
                    if tenUpsampled.shape[3] != tenOne_list[i].shape[3]: tenUpsampled = torch.nn.functional.pad(tenUpsampled,[0,1,0,0])
                    tenInput = tenTwo_list[i]
                    tenFlow_warp = tenUpsampled
                    H, W = tenFlow_warp.shape[2], tenFlow_warp.shape[3]
                    tenHor = torch.linspace(-1.0, 1.0, W, device=tenFlow_warp.device).view(1,1,1,W).repeat(1,1,H,1)
                    tenVer = torch.linspace(-1.0, 1.0, H, device=tenFlow_warp.device).view(1,1,H,1).repeat(1,1,1,W)
                    tenGrid = torch.cat([tenHor, tenVer], 1)
                    tenFlowNorm = torch.cat([tenFlow_warp[:,0:1]*(2.0/(tenInput.shape[3]-1.0)), tenFlow_warp[:,1:2]*(2.0/(tenInput.shape[2]-1.0))], 1)
                    warped = torch.nn.functional.grid_sample(tenInput, (tenGrid + tenFlowNorm).permute(0,2,3,1), mode="bilinear", padding_mode="reflection", align_corners=True)
                    tenFlow = self.netBasic[i](torch.cat([tenOne_list[i], warped, tenUpsampled], 1)) + tenUpsampled
                return tenFlow
        
        self.spynet = SPyNet().to(self.device).eval()
        
        weight_path = "/root/weights/network-sintel-final.pytorch"
        if os.path.exists(weight_path):
            state_dict = torch.load(weight_path, map_location="cpu")
            self.spynet.load_state_dict({k.replace("module","net"):v for k,v in state_dict.items()})
        
        self.ocr = PaddleOCR(lang="korean", use_angle_cls=False, show_log=False, use_gpu=True)
        print("✅ Service Ready")

    # ----------------------------------------------------------------
    # 분류 함수 (출력 형식 수정됨: 'kakao' or 'insta')
    # ----------------------------------------------------------------
    def classify_image(self, img_bgr):
        import cv2
        import numpy as np
        import torch

        if self.classifier is None:
            return "kakao" # 모델 없을 경우 기본값 설정 (혹은 에러 처리)

        # [학습 코드와 동일한 전처리]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        
        # Normalize
        img_float = img_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_norm = (img_float - mean) / std
        
        img_t = img_norm.transpose(2, 0, 1)
        
        # .float()로 강제 형변환
        tensor = torch.from_numpy(img_t).unsqueeze(0).to(self.device).float()

        with torch.no_grad():
            output = self.classifier(tensor)
            prob = torch.sigmoid(output).item()

        # [수정됨] 단순 문자열 반환
        if prob > 0.5:
            return "kakao"
        else:
            return "insta"

    def estimate_flow(self, tenOne, tenTwo):
        import torch
        H, W = tenOne.shape[1:]
        Hp, Wp = (H + 31) // 32 * 32, (W + 31) // 32 * 32
        tenOne = torch.nn.functional.interpolate(tenOne.unsqueeze(0), (Hp, Wp))
        tenTwo = torch.nn.functional.interpolate(tenTwo.unsqueeze(0), (Hp, Wp))
        with torch.no_grad():
            flow = torch.nn.functional.interpolate(self.spynet(tenOne, tenTwo), size=(H, W))
        flow[:,0] *= float(W)/float(Wp)
        flow[:,1] *= float(H)/float(Hp)
        return flow[0].cpu().numpy()
    
    # ----------------------------------------------------------------
    # 인스타용 파싱 함수
    # ----------------------------------------------------------------
    def parse_ocr_result_insta(self, ocr_result, image_width):
        import re
        if not ocr_result: return ""
        
        # 1. 원본 데이터 정렬 (Y축 기준)
        example_polys = [line[0] for line in ocr_result]
        example_texts = [line[1][0] for line in ocr_result]
        
        max_y = 0
        if example_polys:
            max_y = max([max([p[1] for p in poly]) for poly in example_polys])
            
        all_items = []
        for i in range(len(example_texts)):
            text, box = example_texts[i], example_polys[i]
            y_coords, x_coords = [p[1] for p in box], [p[0] for p in box]
            y_center = (min(y_coords)+max(y_coords))/2
            
            all_items.append({
                'text': text, 
                'y_center': y_center,
                'x_left': min(x_coords), 
                'x_right': max(x_coords),
            })
        
        all_items.sort(key=lambda x: x['y_center'])

        all_items.sort(key=lambda x: x['y_center'])

        # ---------------------------------------------------------
        # [Step 1] 헤더 분석: 이름(Y) 기준으로 가까운 하단 텍스트(ID/상태) 모두 무시
        # ---------------------------------------------------------
        opponent_name = "상대방"
        start_idx = 0            
        noise_pattern = re.compile(r"^(\d{1,2}:\d{2}|\d+%|\d+)$")
        
        found_name = False
        name_y_center = 0
        
        # 헤더로 간주할 Y축 거리 임계값 (이름 바로 밑에 붙어있는 것들은 무시)
        # 보통 ID나 상태메시지는 이름과 100px 이내에 붙어있습니다.
        HEADER_MARGIN = 120 
        
        for i, item in enumerate(all_items):
            text = item['text']
            
            # 1. 상단 노이즈 패스
            if noise_pattern.match(text): continue
            
            if not found_name:
                # 이름을 찾음
                opponent_name = text
                found_name = True
                name_y_center = item['y_center']
                continue 
            
            # 2. 이름을 찾은 후:
            # 현재 아이템의 Y좌표가 이름과 너무 가깝다면(헤더 영역) 무시하고 넘어감
            if found_name and (item['y_center'] - name_y_center < HEADER_MARGIN):
                continue

            # 3. Y 거리 차이가 충분히 벌어졌다면 여기서부터가 실제 채팅
            start_idx = i
            break

        chat_items = all_items[start_idx:]
        
        # ---------------------------------------------------------
        # [Step 2] 푸터 분석: 하단 입력창 제거
        # ---------------------------------------------------------
        input_area_threshold = max_y * 0.90 
        chat_items = [item for item in chat_items if item['y_center'] < input_area_threshold]

        # ---------------------------------------------------------
        # [Step 3] 채팅 라인 그룹화
        # ---------------------------------------------------------
        time_regex = re.compile(r"(오전|오후)?\s*\d{1,2}[:시]\s?\d{2}|^\d{4}년|\d{1,2}월\s?\d{1,2}일")

        for item in chat_items:
            item['is_timestamp'] = bool(time_regex.search(item['text']))

        final_lines = []
        current_line = []
        if chat_items:
            base_y = chat_items[0]['y_center']
            for item in chat_items:
                if abs(item['y_center'] - base_y) < 20:
                    current_line.append(item)
                else:
                    current_line.sort(key=lambda x: x['x_left'])
                    final_lines.append(current_line)
                    current_line = [item]
                    base_y = item['y_center']
            if current_line:
                current_line.sort(key=lambda x: x['x_left'])
                final_lines.append(current_line)

        # ---------------------------------------------------------
        # [Step 4] 파싱 및 임시 저장 (후처리를 위해 구조체로 저장)
        # ---------------------------------------------------------
        temp_logs = []  # 문자열 대신 딕셔너리 리스트 사용
        center_x = image_width / 2
        
        DEFAULT_TIME = "2025. 12. 3. 04:56"
        last_timestamp = DEFAULT_TIME

        for line_items in final_lines:
            # 1) 타임스탬프 갱신
            timestamps = [x['text'] for x in line_items if x['is_timestamp']]
            if timestamps:
                last_timestamp = timestamps[-1]

            # 2) 메시지 내용 추출
            msg_texts = [x['text'] for x in line_items if not x['is_timestamp']]
            if not msg_texts: continue
            
            full_msg = " ".join(msg_texts)

            # 3) 화자 구분
            first_item = next(x for x in line_items if not x['is_timestamp'])
            if first_item['x_left'] > center_x - (image_width * 0.1): 
                speaker = "나"
            else:
                speaker = opponent_name

            # **중요: 바로 문자열로 만들지 않고 데이터로 저장**
            temp_logs.append({
                'time': last_timestamp,
                'speaker': speaker,
                'msg': full_msg
            })

        # ---------------------------------------------------------
        # [Step 5] 후처리: 타임스탬프 역방향 채우기 (Backfill)
        # ---------------------------------------------------------
        # 1. 전체 로그 중 '2000년...'이 아닌 첫 번째 유효 시간을 찾음
        first_valid_time = DEFAULT_TIME
        for log in temp_logs:
            if log['time'] != DEFAULT_TIME:
                first_valid_time = log['time']
                break
        
        # 2. 유효 시간이 발견되었다면, 앞부분의 미상 시간들을 모두 이 시간으로 덮어씀
        if first_valid_time != DEFAULT_TIME:
            for log in temp_logs:
                if log['time'] == DEFAULT_TIME:
                    log['time'] = first_valid_time
                else:
                    # 유효한 시간을 만나면(이미 정상이므로) 루프 중단
                    break

        # ---------------------------------------------------------
        # [Step 6] 최종 문자열 변환
        # ---------------------------------------------------------
        return "\n".join([f"{log['time']}, {log['speaker']} : {log['msg']}" for log in temp_logs])

    def parse_ocr_result_kakao(self, ocr_result, image_width, image_height):
        if not ocr_result: return ""
        import re

        # ==============================================================================
        # 데이터 정제 및 타임스탬프/노이즈 분류
        # ==============================================================================
        
        def get_timestamp_token(text):
            # 1. 구분자 정규화 (다양한 노이즈 패턴 대응)
            normalized = re.sub(r'[.,-]', ':', text)
            
            # 2. 숫자와 콜론만 남기고 추출
            clean_nums = re.sub(r"[^\d:]", "", normalized)

            # 3. [전략 A] 완벽한 포맷 (Strict Match)
            if re.match(r"^(\d{1,2}):(\d{2})$", clean_nums):
                return clean_nums

            # 4. [전략 B] 깨진 타임스탬프 후보군 (Loose Match) -> "xx:xx" 변환
            if ':' in normalized and re.search(r'\d', normalized):
                if len(normalized) <= 8:
                    return "xx:xx"
            
            return None

        items = []
        for line in ocr_result:
            box = line[0]
            text = line[1][0]
            
            # 필터링 리스트
            if text in ['<', '>', '-', '=', '파싱용', '메시지입력', '전송', '카톡', '대화', '|', 'emoticon']: continue
            if re.match(r"^\d{1,3}%?$", text): continue 

            # (기존 좌표 필터링 로직 제거 - Step 2 이후로 이동)

            ts_token = get_timestamp_token(text)
            
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            
            items.append({
                'text': text, 
                'clean_text': text.replace(" ", ""),
                'ts_token': ts_token,
                'y_center': sum(ys)/len(ys),
                'x_left': min(xs),
                'height': max(ys) - min(ys)
            })

        items.sort(key=lambda x: x['y_center'])

        # ==============================================================================
        # 라인 그룹화 (Tough Rule)
        # ==============================================================================
        
        lines = []
        if items:
            current_line = [items[0]]
            base_y = items[0]['y_center']
            
            for item in items[1:]:
                if abs(item['y_center'] - base_y) < (item['height'] * 0.7 + 5):
                    current_line.append(item)
                else:
                    current_line.sort(key=lambda x: x['x_left'])
                    lines.append(current_line)
                    current_line = [item]
                    base_y = item['y_center']
            
            if current_line:
                current_line.sort(key=lambda x: x['x_left'])
                lines.append(current_line)

        # ==============================================================================
        # 영역 기반 필터링 (라인 병합 후 처리)
        # ==============================================================================
        # 개별 글자가 아닌 '완성된 줄' 단위로 위치를 판단하여 제거합니다.
        
        if image_height > 0:
            filtered_lines = []
            for line in lines:
                # 해당 라인의 대표값 계산 (평균 y값, 가장 왼쪽 x값)
                line_y_center = sum([item['y_center'] for item in line]) / len(line)
                line_x_left = min([item['x_left'] for item in line])
                
                # 1) 상단 1/8 제거 (시스템 영역)
                if line_y_center < image_height / 15: continue
                
                # 2) 하단 1/12 제거 (메시지 입력창)
                if line_y_center > image_height * (14/15): continue
                
                # 3) 상단 1/4 이면서 좌측 1/6 영역 (공지사항 아이콘 등)
                if line_y_center < image_height / 4 and line_x_left <= image_width / 9 : continue
                
                filtered_lines.append(line)
            lines = filtered_lines

        # ==============================================================================
        # 문맥 파싱 (동적 화자 로직 & 필터링 추가)
        # ==============================================================================
        
        parsed_logs = []
        known_speakers = set(["나"]) 
        last_left_speaker = "알수없음"
        current_date = "2025. 12. 5." 
        center_x = image_width / 2 if image_width else 200

        i = 0
        while i < len(lines):
            line = lines[i]
            full_line_str = " ".join([x['text'] for x in line])
            
            # 답장 로직 제거 ("에게 답장"이 포함된 줄과 그 다음 줄 스킵)
            if "에게 답장" in full_line_str:
                i += 2
                continue

            # 날짜 헤더 처리
            # 날짜를 추출하여 current_date를 갱신하되, parsed_logs에는 추가하지 않음(출력 제외)
            date_match = re.search(r"20\d{2}[^0-9]+\d{1,2}[^0-9]+\d{1,2}", full_line_str)
            if date_match:
                nums = re.findall(r"\d+", full_line_str)
                if len(nums) >= 3:
                    current_date = f"{nums[0]}. {nums[1]}. {nums[2]}."
                i += 1
                continue

            # [라인 요소 분해]
            time_obj = None
            content_texts = []
            
            avg_x = sum([x['x_left'] for x in line]) / len(line)
            is_me = avg_x > center_x

            for item in line:
                if item['ts_token']:
                    time_obj = item['ts_token']
                    continue
                content_texts.append(item['text'])

            clean_content = " ".join(content_texts)
            
            # 빈 내용이면 스킵
            if not clean_content: 
                i += 1
                continue

            # 시스템 메시지 제거 로직
            # 제거 대상 키워드 정의
            sys_remove_keywords = ["초대했습니다", "들어왔습니다", "나갔습니다", "원을 보냈어요", "행운의 주인공"]
            
            # 화자 파악을 위해 초대/나감 메시지는 분석이 필요함 (DB갱신용)
            if "초대했습니다" in clean_content or "나갔습니다" in clean_content or "들어왔습니다" in clean_content:
                names = re.findall(r"([가-힣a-zA-Z0-9]+)님", clean_content)
                for n in names: known_speakers.add(n)
            
            # 키워드가 포함되어 있으면 로그에 추가하지 않고 스킵 (출력 제외)
            if any(keyword in clean_content for keyword in sys_remove_keywords):
                i += 1
                continue

            # [화자 및 메시지 처리]
            final_speaker = None
            is_name_tag_line = False

            if is_me:
                final_speaker = "나"
            else:
                txt_nospace = clean_content.replace(" ", "")
                
                if txt_nospace in known_speakers:
                    is_name_tag_line = True
                elif 2 <= len(txt_nospace) <= 6 and re.match(r"^[가-힣a-zA-Z0-9]+$", txt_nospace):
                    ending_checker = txt_nospace[-1]
                    msg_indicators = ['다', '요', '음', '는', '게', '지', '네', '가', '나', '어', 'ㅋ', 'ㅎ', '?', '!']
                    
                    if ending_checker in msg_indicators:
                        is_name_tag_line = False
                    else:
                        is_name_tag_line = True
                        known_speakers.add(txt_nospace)

                if is_name_tag_line:
                    last_left_speaker = clean_content 
                    i += 1
                    continue 
                else:
                    final_speaker = last_left_speaker

            # 메시지에 현재 날짜(current_date)를 함께 저장하여 정확한 날짜 표기 보장
            parsed_logs.append({
                'type': 'msg',
                'speaker': final_speaker,
                'text': clean_content,
                'time': time_obj,
                'date': current_date 
            })
            
            i += 1

        # ==============================================================================
        # 시간 역전파 (Back-fill)
        # ==============================================================================
        
        final_lines = []
        future_time = "10:22"
        
        # 마지막으로 발견된 "유효한" 시간을 기본값으로 설정
        for log in reversed(parsed_logs):
            t = log.get('time')
            if t and t != "xx:xx" and ":" in t:
                future_time = t
                break
        
        for log in reversed(parsed_logs):
            curr_t = log.get('time')
            # 현재 시간이 유효하면 future_time을 갱신
            if curr_t and curr_t != "xx:xx" and ":" in curr_t:
                future_time = curr_t
            
            display_time = future_time
            msg_date = log.get('date', "2025. 12. 5.") # 저장된 날짜 사용
            
            final_lines.append(f"{msg_date} {display_time}, {log['speaker']} : {log['text']}")

        final_lines.reverse()
        return "\n".join(final_lines)

    def _flush_turn(self, current_turn, chat_logs, ts_str, center_x):
        first_line = current_turn[0]
        l_center = (min(x['x_left'] for x in first_line) + max(x['x_right'] for x in first_line)) / 2
        speaker = "나" if l_center > center_x else " ".join([x['text'] for x in first_line])
        start_idx = 0 if speaker == "나" or len(current_turn) == 1 else 1
        msgs = [" ".join([x['text'] for x in row]) for row in current_turn[start_idx:]]
        full_msg = " ".join(msgs)
        if full_msg: chat_logs.append(f"{ts_str}, {speaker} : {full_msg}")

    # =================================================================
    # API Methods
    # =================================================================
    @modal.method()
    def process_image(self, image_bytes: bytes):
        import cv2
        import numpy as np
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: return "Error: Invalid image."
        
        cls_result = self.classify_image(img)
        
        result = self.ocr.ocr(img, cls=False)
        if not result or not result[0]: return f"[{cls_result}] OCR 결과 없음"
        
        # [수정됨] 분류 결과에 따라 다른 파싱 함수 호출
        if cls_result == "kakao":
            ocr_text = self.parse_ocr_result_kakao(result[0], img.shape[1], img.shape[0])
        else:
            # 'insta'일 경우
            ocr_text = self.parse_ocr_result_insta(result[0], img.shape[1])
            
        return f"--- 분류 결과: {cls_result} ---\n\n{ocr_text}"
    
    @modal.method()
    def process_video(self, video_bytes: bytes):
        import cv2
        import numpy as np
        import torch
        import os
        
        temp_path = "/tmp/input_video.mp4"
        with open(temp_path, "wb") as f:
            f.write(video_bytes)
            
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened(): return {"text": "Error: Cannot open video."}

        print("🎥 Extracting keyframes (Original Logic - No Skip)...")
        ret, prev = cap.read()
        if not ret: return {"text": "Error: Empty video."}
        
        prev_rgb = cv2.cvtColor(prev, cv2.COLOR_BGR2RGB)
        h, w = prev_rgb.shape[:2]
        
        scroll_acc = 0
        extracted_frames_rgb = [prev_rgb]
        
        # [복구됨] 프레임 스킵 없이 모든 프레임 검사
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            prev_crop = prev_rgb[h//4 : h*3//4, w//4 : w*3//4]
            curr_crop = frame_rgb[h//4 : h*3//4, w//4 : w*3//4]
            
            tenOne = torch.from_numpy(prev_crop.transpose(2,0,1).copy()).float().to(self.device) / 255.0
            tenTwo = torch.from_numpy(curr_crop.transpose(2,0,1).copy()).float().to(self.device) / 255.0
            
            flow = self.estimate_flow(tenOne, tenTwo)
            dy = np.median(flow[:,:,1])
            
            if abs(dy) < 0.3: dy = 0
            scroll_acc += dy
            
            # 임계값은 팀원 원래 코드(0.75) 유지
            if abs(scroll_acc) > h * 0.75:
                extracted_frames_rgb.append(frame_rgb)
                scroll_acc = 0
                print(f"📸 Captured frame {len(extracted_frames_rgb)}")
            
            prev_rgb = frame_rgb
            
        cap.release()
        
        print(f"📝 Running OCR on {len(extracted_frames_rgb)} frames...")
        all_logs = []

        # [추가됨] 영상 분류 로직 (기본값 kakao)
        cls_result = ""
        if extracted_frames_rgb:
            first_frame_bgr = cv2.cvtColor(extracted_frames_rgb[0], cv2.COLOR_RGB2BGR)
            cls_result = self.classify_image(first_frame_bgr)
            all_logs.append(f"--- [영상 분류 결과: {cls_result}] ---\n")

        for frame in extracted_frames_rgb:
            result = self.ocr.ocr(frame, cls=False)
            if not result or not result[0]: continue
            
            # [수정됨] 분류 결과에 따라 다른 파싱 함수 호출
            if cls_result == "kakao":
                parsed_text = self.parse_ocr_result_kakao(result[0], w, h)
            else:
                # 'insta'일 경우
                parsed_text = self.parse_ocr_result_insta(result[0], w)
                
            if parsed_text:
                all_logs.append(parsed_text)
                
        return {"text": "\n".join(all_logs)}

# --- 로컬 테스트용 ---
@app.local_entrypoint()
def main(file_path: str = "test_video.mp4"):
    service = OCRService()
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        
        print(f"🚀 Sending {file_path} to Modal...")
        if file_path.endswith(('.mp4', '.mov')):
            result = service.process_video.remote(data)
            print("\n--- [Video Result] ---")
            print(result.get("text"))
        else:
            result = service.process_image.remote(data)
            print("\n--- [Image Result] ---")
            print(result)
            
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")