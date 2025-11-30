import modal

# [Helper] 빌드 타임 다운로드 (CPU 모드)
def download_models():
    import os
    os.system("mkdir -p /root/weights")
    os.system("wget -O /root/weights/network-sintel-final.pytorch http://content.sniklaus.com/github/pytorch-spynet/network-sintel-final.pytorch")
    
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
        from paddleocr import PaddleOCR
        import os
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n👉 Current Computing Device: {self.device}\n")
        
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

    def parse_ocr_result(self, ocr_result, image_width):
        import re
        if not ocr_result: return ""
        
        example_polys = [line[0] for line in ocr_result]
        example_texts = [line[1][0] for line in ocr_result]
        time_regex = re.compile(r"^(오전|오후)\s*\d{1,2}[:\s]*\d{2}")
        
        processed_items = []
        for i in range(len(example_texts)):
            text, box = example_texts[i], example_polys[i]
            y_coords, x_coords = [p[1] for p in box], [p[0] for p in box]
            processed_items.append({
                'text': text, 'y_center': (min(y_coords)+max(y_coords))/2,
                'x_left': min(x_coords), 'x_right': max(x_coords),
                'is_timestamp': bool(time_regex.match(text))
            })
        processed_items.sort(key=lambda x: x['y_center'])
        
        final_lines = []
        current_items = []
        if processed_items:
            base_y = processed_items[0]['y_center']
            for item in processed_items:
                if abs(item['y_center'] - base_y) < 30:
                    current_items.append(item)
                else:
                    current_items.sort(key=lambda x: x['x_left'])
                    final_lines.append(current_items)
                    current_items = [item]
                    base_y = item['y_center']
            if current_items:
                current_items.sort(key=lambda x: x['x_left'])
                final_lines.append(current_items)

        chat_logs = []
        current_turn = []
        center_x = image_width / 2
        last_timestamp = "시간미상"

        def format_time(s):
            s = s.replace(" ", "")
            try:
                clean_s = s.replace("오전","").replace("오후","").strip()
                if ":" not in clean_s: 
                    parts = [clean_s[:-2], clean_s[-2:]] if len(clean_s) >=3 else [clean_s, "00"]
                else: parts = clean_s.split(":")
                h, m = int(parts[0]), parts[1]
                if "오후" in s and h != 12: h += 12
                if "오전" in s and h == 12: h = 0
                return f"{h:02}:{m}"
            except: return s

        for line_items in final_lines:
            texts = [x for x in line_items if not x['is_timestamp']]
            tss = [x for x in line_items if x['is_timestamp']]
            if texts: current_turn.append(texts)
            if tss: 
                ts_str = format_time(tss[0]['text'])
                last_timestamp = ts_str
                if not current_turn: continue
                self._flush_turn(current_turn, chat_logs, ts_str, center_x)
                current_turn = []
        
        if current_turn:
            self._flush_turn(current_turn, chat_logs, last_timestamp, center_x)

        return "\n".join(chat_logs)

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
        result = self.ocr.ocr(img, cls=False)
        if not result or not result[0]: return ""
        return self.parse_ocr_result(result[0], img.shape[1])

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
        for frame in extracted_frames_rgb:
            result = self.ocr.ocr(frame, cls=False)
            if not result or not result[0]: continue
            parsed_text = self.parse_ocr_result(result[0], w)
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