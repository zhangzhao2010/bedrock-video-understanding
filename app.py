import cv2
import io
from pathlib import Path
import time
from PIL import Image
import streamlit as st
import boto3
import base64
import os
import magic

bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-west-2")


# 模型选项
IMAGE_MODELS = (
    "us.amazon.nova-pro-v1:0",
    "us.amazon.nova-lite-v1:0",
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
)


def extract_frames(video_path, output_dir, fps=1):
    """
    从视频中每秒提取一帧并保存到指定目录

    参数:
        video_path: 视频文件路径
        output_dir: 输出图片保存目录
        fps: 每秒提取的帧数，默认为1
    """
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {video_path}")
        return

    # 获取视频信息
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps

    print(f"视频信息:")
    print(f"- 路径: {video_path}")
    print(f"- FPS: {video_fps}")
    print(f"- 总帧数: {total_frames}")
    print(f"- 时长: {duration:.2f} 秒")

    # 计算帧间隔
    frame_interval = int(video_fps / fps)
    if frame_interval < 1:
        frame_interval = 1

    # 提取帧
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 每隔frame_interval帧保存一次
        if frame_count % frame_interval == 0:
            # 计算当前时间点（秒）
            timestamp = frame_count / video_fps
            # 保存图片
            output_path = os.path.join(
                output_dir, f"frame_{saved_count:04d}_{timestamp:.2f}s.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1

            # 显示进度
            if saved_count % 10 == 0:
                print(f"已保存 {saved_count} 帧图片，当前视频时间点: {timestamp:.2f}s")

        frame_count += 1

    # 释放资源
    cap.release()
    print(f"完成! 共提取了 {saved_count} 帧图片，保存在 {output_dir}")


def get_mime_type(file_path):
    mime = magic.Magic(mime=True)
    mime_type = mime.from_file(file_path)
    return mime_type


def resize_image(images):
    # 处理所有下载的图片
    image_bytes_list = []
    for file_path in images:
        # 获取文件格式
        image_format = get_mime_type(file_path)
        image_format = image_format.split('/')[1]
        # 读取图片文件
        file_size = os.path.getsize(file_path)

        with Image.open(file_path) as img:
            # 检查是否需要调整大小
            max_size = 720
            if max(img.width, img.height) > max_size:
                # 计算调整后的尺寸，保持宽高比
                ratio = max_size / max(img.width, img.height)
                new_size = (int(img.width * ratio),
                            int(img.height * ratio))
                # 调整图像大小
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            # 如果文件大于3MB，需要进一步压缩
            if file_size > 3 * 1024 * 1024:  # 3MB in bytes
                # 计算新的尺寸（50%）
                new_size = (img.width // 2, img.height // 2)
                # 调整图像大小
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            # 将图像保存到内存
            img_byte_arr = io.BytesIO()
            format_name = img.format if img.format else image_format
            img.save(img_byte_arr, format=format_name)
            image_bytes = img_byte_arr.getvalue()
            image_bytes_list.append((image_format, image_bytes))

    return image_bytes_list


def call_claude(model_id, system_prompt, temperature, top_p, length, video_local_path, prompt):
    frames_dir = f'{video_local_path}_frames'
    extract_frames(video_local_path, frames_dir, 1)

    image_paths = [os.path.join(frames_dir, f)
                   for f in os.listdir(frames_dir) if f.endswith('.jpg')]
    images = resize_image(image_paths)

    content = []
    for format, img in images:
        content.append({
            "image": {
                "format": format,
                "source": {
                    "bytes": img
                }
            }
        })
        if len(content) >= 20:
            break

    content.append({
        "text": prompt
    })

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    system = [{
        "text": system_prompt
    }]

    inferenceConfig = {
        "maxTokens": int(length),
        'temperature': temperature,
        'topP': top_p
    }

    start_time = time.time()

    response = bedrock_runtime.converse(
        modelId=model_id,
        messages=messages,
        system=system,
        inferenceConfig=inferenceConfig
    )

    # 计算耗时
    elapsed_time = time.time() - start_time
    st.info(f"API调用耗时: {elapsed_time:.2f}秒")
    return response


def call_nova(model_id, system_prompt, temperature, top_p, length, video_local_path, prompt):
    with open(video_local_path, "rb") as file:
        media_bytes = file.read()
        media_base64 = base64.b64encode(media_bytes)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "video": {
                        "format": "mp4",
                        "source": {
                            "bytes": media_bytes
                        }
                    }
                },
                {"text": prompt},
            ],
        }
    ]

    system = [{
        "text": system_prompt
    }]

    inferenceConfig = {
        "maxTokens": int(length),
        'temperature': temperature,
        'topP': top_p
    }

    start_time = time.time()

    response = bedrock_runtime.converse(
        modelId=model_id,
        messages=messages,
        system=system,
        inferenceConfig=inferenceConfig
    )

    # 计算耗时
    elapsed_time = time.time() - start_time
    st.info(f"API调用耗时: {elapsed_time:.2f}秒")
    return response


# Initialize session state
if "previous_model" not in st.session_state:
    st.session_state.previous_model = None
if "s3_url" not in st.session_state:
    st.session_state.s3_url = ""

with st.sidebar:
    st.title("参数设置")
    model = st.selectbox(
        "模型选择", IMAGE_MODELS)

    # Check if model changed
    if st.session_state.previous_model is not None and st.session_state.previous_model != model:
        st.session_state.previous_model = model
        st.rerun()

    # Update previous model
    st.session_state.previous_model = model

    system_prompt = st.text_area(
        "系统提示词", value="""You are an expert in video review and tagging. Please analyze the input video in detail based on the following criteria and return the results in a structured format:
Face Detection: Determine whether the video contains individuals showing a complete, frontal face (the camera angle must be a direct frontal shot). Provide detailed explanations and note any anomalies.
Video Quality Assessment: Evaluate if the video is shaky and whether the lighting is sufficient. Provide ratings for stability and lighting along with detailed explanations.
Content Detection: Check whether the video involves inappropriate content such as pornography, violence, or suggestive implications. List the specific risk types detected and explain your reasoning.
Return your analysis in a JSON format as shown in the example below:
{
  "face_detection": {
    "status": "Yes/No",
    "details": "Detailed explanation and criteria used"
  },
  "video_quality": {
    "stability": "Good/Bad",
    "lighting": "Sufficient/Insufficient",
    "details": "Detailed explanation and criteria used"
  },
  "content_detection": {
    "inappropriate_content": "None/Detected",
    "content_types": ["Pornography", "Violence", "Suggestive"],
    "details": "Detailed explanation and criteria used"
  }
}
Ensure that the output strictly follows the JSON structure provided above.
Do not include any other content or instructions in the output content except for the JSON structure""", height=200)

    temperature = st.select_slider(
        "温度", value=0.7, options=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    top_p = st.select_slider(
        "Top P", value=0.9, options=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    length = st.text_input("生成长度", value="1024")

    # s3_bucket = st.text_input("S3 Bucket", value="")

st.header('AWS Bedrock 视频理解样例')


# 创建tmp目录（如果不存在）
tmp_dir = "./tmp"
Path(tmp_dir).mkdir(parents=True, exist_ok=True)

# 创建文件上传组件，指定接受视频文件
uploaded_video = st.file_uploader(
    "请选择要上传的视频文件", type=["mp4", "mov", "avi", "mkv"])

if uploaded_video is not None:
    if uploaded_video.size / (1024 * 1024) > 25:
        st.error("视频大小不能超过25MB")
        st.stop()

    # 显示上传的视频信息
    file_details = {
        "文件名": uploaded_video.name,
        "文件类型": uploaded_video.type,
        "文件大小": f"{uploaded_video.size / (1024 * 1024):.2f} MB"
    }

    col1, col2 = st.columns([1, 2])
    with col1:
        # 显示上传的视频
        st.video(uploaded_video)

    # 保存视频到本地tmp目录
    save_path = os.path.join(tmp_dir, uploaded_video.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_video.getbuffer())

    with col2:
        st.write("文件详情:")
        for key, value in file_details.items():
            st.write(f"{key}: {value}")
        video_local_path = st.text_input(
            'video_local_path', value=save_path, disabled=True)


prompt = st.text_area("提示词", value="", height=200, key="prompt")

if st.button("提交"):
    if not prompt:
        st.error("请输入prompt")
        st.stop()
    if not video_local_path:
        st.error("请上传视频")
        st.stop()
    if model.startswith("us.amazon.nova"):
        my_bar = st.progress(0, text="Processing...")
        with st.spinner('Processing...'):
            try:
                response = call_nova(model, system_prompt, temperature,
                                     top_p, length, video_local_path, prompt)
                st.json(response.get("output"))
            except Exception as e:
                st.error(f"call nova failed: {str(e)}")
            finally:
                my_bar.progress(100, text="Done")
    elif model.startswith("us.anthropic.claude"):
        my_bar = st.progress(0, text="Processing...")
        with st.spinner('Processing...'):
            try:
                response = call_claude(model, system_prompt, temperature,
                                       top_p, length, video_local_path, prompt)
                st.json(response.get("output"))
                st.json(response.get("usage"))
            except Exception as e:
                st.error(f"call nova failed: {str(e)}")
            finally:
                my_bar.progress(100, text="Done")
    else:
        st.error("暂不支持此模型")
