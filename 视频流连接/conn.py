###嵌入摄像头视频位置###
@app.route('/video_feed')
def video_feed():
    def generate():
        global latest_frame
        try:
            with requests.get(ESP32_STREAM_URL, stream=True, timeout=5) as r:
                buffer = b''  # 用于累积数据
                boundary = b'--frame'  # MJPEG流分隔符
                for chunk in r.iter_content(1024):
                    buffer += chunk
                    while True:
                        # 查找帧起始位置
                        start_idx = buffer.find(boundary)
                        if start_idx == -1:
                            break
                        # 查找帧结束位置
                        end_idx = buffer.find(boundary, start_idx + len(boundary))
                        if end_idx == -1:
                            break
                        # 提取完整帧数据
                        frame_data = buffer[start_idx:end_idx]
                        buffer = buffer[end_idx:]  # 保留剩余数据

                        # 提取JPEG数据（假设头信息以\r\n\r\n结尾）
                        header_end = frame_data.find(b'\r\n\r\n')
                        if header_end != -1:
                            jpeg_data = frame_data[header_end + 4:]
                            # 更新最新帧
                            with frame_lock:
                                latest_frame = jpeg_data

                        yield frame_data  # 转发帧到前端
        except Exception as e:
            print("Stream Error:", e)
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/capture')
def capture():
    with frame_lock:
        if latest_frame is None:
            return Response(status=404)

        # 使用PIL处理图片
        img = Image.open(io.BytesIO(latest_frame))

        # 计算缩放比例（示例设为宽度600px）
        base_width = 600
        w_percent = (base_width / float(img.size[0]))
        h_size = int((float(img.size[1]) * float(w_percent)))

        # 高质量缩放
        img = img.resize((base_width, h_size), Image.Resampling.LANCZOS)

        # 转换为JPEG数据
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=85)

        return Response(img_byte_arr.getvalue(), mimetype='image/jpeg')


@app.route('/identify', methods=['POST'])
def identify_species():
    with frame_lock:
        if latest_frame is None:
            return jsonify({"error": "No frame available"}), 400

        # 处理图片
        img = Image.open(io.BytesIO(latest_frame))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=85)
        base64_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

        # 调用AI模型
        client = OpenAI(
            api_key='sk-dd08879040994cb196446a5707b1b1cd',
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        # 第一轮识别
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "你是一个生物学家助手"}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    },
                    {"type": "text", "text": "识别图中的动植物，直接给出学名以及一到两个别名"}
                ]
            }
        ]

        completion = client.chat.completions.create(
            model="qwen-vl-max-latest",
            messages=messages,
        )
        species_info = completion.choices[0].message.content

        # 第二轮获取图片
        species_name = re.sub(r"[^a-zA-Z\u4e00-\u9fa5]", "", species_info)[:10]
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": f"对这个动物或者植物做一个30字的简介"}
            ]
        })

        completion = client.chat.completions.create(
            model="qwen-vl-max-latest",
            messages=messages,
        )
        image_url = completion.choices[0].message.content

        return jsonify({
            "species": species_info,
            "image_url": image_url
        })