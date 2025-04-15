import base64
import shutil
import json
import time
import os
import math
from urllib.parse import urlparse
import cv2
import boto3
import uuid
from PIL import Image
import requests
from video_quality_checker import check_video_quality

TMP_DIR = './tmp'
NOVA_PROMPT = """
You are a professional video review and tagging model expert, responsible for reviewing and tagging individual videos according to the given review rules.
Please carefully read the review rules in <rules> and strictly follow these rules to review and classify videos.
When outputting results, please follow the requirements in <output_format> to provide readable and easily parsable fields in JSON format.

<rules>
# Tag Categories and Definitions

## I. Sensitive Content Classification

### A. Inappropriate Content
1. **Nudity Content** [Tag: `NUDITY`]
   - Complete or partial nudity
   - Exposure of breasts, buttocks, or genitals
   - Clearly visible underwear
   - Excessive highlighting of body contours, women wearing tight pants clearly outlining private parts (camel toe or bulges)

2. **Sexually Suggestive Behavior** [Tag: `SEXUAL_SUGGESTION`]
   - Self-stimulating behaviors, such as masturbation (hands touching breasts, buttocks, triangular area, not including scratching)
   - Seductive movements and poses
   - Sexual interaction behaviors
   - Use of adult props (vibrators, dildos, vibrating eggs, etc.), seductive playing with rod-shaped objects (cucumbers, bananas, etc.)

3. **Framing Issues** [Tag: `INAPPROPRIATE_FRAMING`]
   - Excessive focus on sensitive areas, sensitive body parts (chest, triangular area, thighs, buttocks) occupying more than 1/2 of the screen area

4. **Seductive Oral Behaviors** [Tag: `MOUTH_ORAL`]
   - Protruding tongue, licking lips, biting lips
   - Seductive licking behaviors, such as licking objects or inserting fingers into mouth and sucking (excluding normal eating, nail biting)

## II. Other Sensitive Content

1. **Restricted Information** [Tag: `RESTRICTED_CONTENT`]
   - Personal privacy information, displaying phone numbers, addresses, bank card numbers, emails, etc.
   - Prohibited items (cigarettes, alcohol, weapons, etc.)
   - Politically and religiously sensitive content, such as religious icons, national flags, politically sensitive figures, etc.
   - Dangerous or self-harming behaviors

## III. Technical and Subject Issues

1. **Technical Issues** [Tag: `TECHNICAL_ISSUE`]
   - The subject's face is not fully visible; the person must show a complete face(both two eyes, nose, and mouth must appear simultaneously)
   - The subject's face angle is not a front-facing shot, poor camera angle, such as side shots and overhead shots (lying on bed or sofa), resulting in poor video quality
   - Poor lighting, such as excessive brightness or darkness, resulting in poor video quality
   - The subject's posture is lying down
   - Inappropriate distance, such as too far to clearly see the face, or too close making the face occupy the entire video
   - Black screen, such as the screen being black

2. **Subject Issues** [Tag: `SUBJECT_ISSUE`]
   - Minors appearing
   - Only males appearing
   - Unclear face
   - Bed or bedroom scenes appearing
   - No person appearing in the frame

3. **Video Quality Issues** [Tag: `VIDEO_QUALITY_ISSUE`]
   - Shaky footage or screen shaking, such as a blurry frame of the image
   - Evaluate if the video is shaky and whether the lighting is sufficient. Provide ratings for stability and lighting along with detailed explanations.

## No Issue Content
[Tag: `NO_ISSUE`]

# Review Result Output Rules Description
The review results must be strictly detected according to each tag and provide a final judgment, along with detailed explanations and confidence scores. Confidence scores range from 100 (highest) to 1 (lowest). Please refer to the example in <output_format> for the output format.
</rules>

<output_format>
Please return the analysis results in JSON format, example format as follows:
{
    "NUDITY": {
        "explanation": "Female underwear appears in the video",
        "is_exist": 1,
        "confidence": 80
    },
    "SEXUAL_SUGGESTION": {
        "explanation": "The video shows a female with seductive movements and poses.",
        "is_exist": 1,
        "confidence": 80
    },
    "INAPPROPRIATE_FRAMING": {
        "explanation": "The video shows female chest exposure exceeding 1/2 of the area, violating platform content standards.",
        "is_exist": 1,
        "confidence": 80
    },
    "RESTRICTED_CONTENT": {
        "explanation": "The video involves personal privacy, sensitive information, prohibited items, or political/religious sensitive information.",
        "is_exist": 1,
        "confidence": 99
    },
    "TECHNICAL_ISSUE": {
        "explanation": "The video has poor lighting, and no frontal face of the person appears in the video",
        "is_exist": 1,
        "confidence": 99
    },
    "SUBJECT_ISSUE": {
        "explanation": "Only males appear in the video.",
        "is_exist": 1,
        "confidence": 80
    },
    "NO_ISSUE": {
        "explanation": "Only when all other items are determined to be 'non-existent', mark as 'no issue', indicating that the video meets platform requirements for compliance, frontal face, good video quality, etc.",
        "is_exist": 0,
        "confidence": 99
    }
}
If the is_exist of a tag is 0 and confidence score is greater than 80, do not include the tag in the return result.
Please ensure the output format strictly adheres to the above JSON structure.
Do not include any other content or instructions in the output apart from the JSON structure.
</output_format>

The above is the video content that needs to be reviewed. Please review and tag the video according to the requirements above.
Analyze the video according to each tag's description in sequence. You can first think about and describe the video content, then analyze and tag it.
Before outputting the final result, please self-check and reflect on whether the output meets the requirements.
"""

SYSTEM_PROMPT = "You are an expert in video moderation. You are responsible for reviewing the video content and providing a detailed analysis of the video content. You will be given a video and a prompt. You will analyze the video according to the prompt and provide a detailed analysis of the video content. The analysis should be in JSON format."


def extract_and_merge_all_frames(local_video_path: str):
    local_dir = os.path.dirname(local_video_path)

    frame_dir = f'{local_dir}/frames'
    os.makedirs(frame_dir, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(local_video_path)
    if not cap.isOpened():
        raise RuntimeError("无法打开视频文件")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    frame_paths = []
    frame_count = 0
    current_time = 0

    while current_time < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)  # 设置时间位置
        success, frame = cap.read()
        if not success:
            break

        frame_filename = f'{frame_dir}/frame_{frame_count:03d}.jpg'
        cv2.imwrite(frame_filename, frame)
        frame_paths.append(frame_filename)

        frame_count += 1
        current_time += 1  # 每秒抽一帧

    cap.release()

    # 拼接图片，每行最多3列
    images = [Image.open(fp) for fp in frame_paths]
    if not images:
        raise RuntimeError("没有成功抽帧")

    if len(images) > 20:
        raise RuntimeError("More than 20 images")

    frame_width, frame_height = images[0].size
    cols = 3
    rows = math.ceil(len(images) / cols)

    merged_width = cols * frame_width
    merged_height = rows * frame_height

    merged_image = Image.new("RGB", (merged_width, merged_height))

    for idx, img in enumerate(images):
        x = (idx % cols) * frame_width
        y = (idx // cols) * frame_height
        merged_image.paste(img, (x, y))

    merged_path = f'{local_dir}/merged_image.jpg'
    merged_image.save(merged_path)
    print(f"拼接图保存到: {merged_path}")

    return merged_path, len(images)


def imageModeration(image_path):
    client = boto3.client('rekognition')

    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()

    response = client.detect_moderation_labels(
        Image={
            'Bytes': image_data
        },
        MinConfidence=50
    )
    return response


def faceDetection(image_path):
    client = boto3.client('rekognition')

    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()

    response = client.detect_faces(
        Image={
            'Bytes': image_data
        },
        Attributes=['AGE_RANGE', 'GENDER', 'FACE_OCCLUDED']
    )
    return response


def analysis_merged_images(image: str, sub_image_count):
    min_age = 18
    gender = "Female"
    face_count = 0
    face_not_occluded_count = 0

    response = faceDetection(image)
    for face_detail in response.get('FaceDetails', []):
        if face_detail.get('Confidence') > 80:
            face_count += 1
            if face_detail.get('AgeRange').get('Low') < min_age:
                min_age = face_detail.get('AgeRange').get('Low')
            if face_detail.get('Gender').get('Value') == "Male":
                gender = "Male"
            if face_detail.get('FaceOccluded').get('Value') == False:
                face_not_occluded_count += 1

    result = {}
    # 出现了几张人脸，数量必须大于 sub_image_count*2/3
    if face_count < sub_image_count * 2 / 3:
        result['FACE_ISSUE'] = {
            'explanation': 'There is no face continuously appearing in the video frame',
            'is_exist': 1,
            'confidence': 99
        }
    # 未成年判断
    if min_age < 18:
        result['MINORS_ISSUE'] = {
            'explanation': 'Minors may appear in the video',
            'is_exist': 1,
            'confidence': 99
        }
    # 判断是否出现过男人
    if gender == "Male":
        result['MALE_ISSUE'] = {
            'explanation': 'Male may appear in the video',
            'is_exist': 1,
            'confidence': 99
        }
    # 判断未遮挡人脸的数量 必须大于 sub_image_count*2/3
    if face_not_occluded_count < sub_image_count * 2 / 3:
        result['FACE_OCCLUDED_ISSUE'] = {
            'explanation': 'The face is occluded in the video frame',
            'is_exist': 1,
            'confidence': 99
        }

    return result


bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")


def call_nova_use_s3_file(s3_uri, model_id, prompt, system_prompt, temperature, top_p, max_token):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "video": {
                        "format": "mp4",
                        "source": {
                            "s3Location": {
                                "uri": s3_uri
                            }
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
        "maxTokens": int(max_token),
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
    print(f"API调用耗时: {elapsed_time:.2f}秒")
    return response


def download_video_from_s3(s3_uri):
    # 解析 S3 URI
    assert s3_uri.startswith("s3://")
    _, bucket_key = s3_uri.split("s3://", 1)
    bucket, key = bucket_key.split("/", 1)

    # 创建存储目录
    tmp_dir = TMP_DIR
    os.makedirs(tmp_dir, exist_ok=True)

    tmp_uuid = uuid.uuid4()
    local_dir = f'{tmp_dir}/{tmp_uuid}'
    os.makedirs(local_dir, exist_ok=True)

    # 下载视频到本地
    s3 = boto3.client('s3')
    local_video_path = f'{local_dir}/{tmp_uuid}.mp4'
    s3.download_file(bucket, key, local_video_path)

    return local_video_path


def download_video_from_url(video_url):
    # 创建存储目录
    tmp_dir = TMP_DIR
    os.makedirs(tmp_dir, exist_ok=True)

    tmp_uuid = uuid.uuid4()
    local_dir = f'{tmp_dir}/{tmp_uuid}'
    os.makedirs(local_dir, exist_ok=True)

    # 从URL中提取文件名
    filename = extract_filename_from_url(video_url)

    # 构建完整的输出路径
    output_path = os.path.join(local_dir, filename)

    # 确保输出目录存在
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    with requests.get(video_url, stream=True) as response:
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print(f"视频已成功下载到: {output_path}")
    return output_path


def extract_filename_from_url(url):
    parsed_url = urlparse(url)
    path = parsed_url.path
    filename = path.split('/')[-1]
    return filename


def handler(event, context):
    try:
        video_s3_uri = event.get('video_s3_uri', '')
        video_url = event.get('video_url', '')

        if video_s3_uri:
            local_video_path = download_video_from_s3(video_s3_uri)
        elif video_url:
            local_video_path = download_video_from_url(video_url)
        else:
            raise RuntimeError("Invalid param")

        merged_imaged, sub_image_count = extract_and_merge_all_frames(
            local_video_path)

        # ffmpeg check video quality
        video_quality_check_result = check_video_quality(local_video_path)
        video_quality_result = {}
        for r in video_quality_check_result['issues']:
            video_quality_result[str(r['type']).upper()] = {
                'explanation': r['description'],
                'is_exist': 1,
                'confidence': 99
            }

        if len(video_quality_result.keys()) > 0:
            return {
                'err_no': 0,
                'err_msg': '',
                'data': video_quality_result
            }

        # rekognition check face
        rek_moderation_result = analysis_merged_images(
            merged_imaged, sub_image_count)
        for k, rek_r in rek_moderation_result.items():
            if rek_r['is_exist'] == 0:
                del rek_moderation_result[k]

        if len(rek_moderation_result.keys()) > 0:
            return {
                'err_no': 0,
                'err_msg': '',
                'data': rek_moderation_result
            }

        # nova check
        model_id = event.get('model_id', 'us.amazon.nova-pro-v1:0')
        prompt = event.get('prompt', NOVA_PROMPT)
        system_prompt = event.get('system_prompt', SYSTEM_PROMPT)
        temperature = event.get('temperature', 0.5)
        top_p = event.get('top_p', 0.9)
        max_token = event.get('max_token', 2048)

        nova_response = call_nova_use_s3_file(
            video_s3_uri, model_id, prompt, system_prompt, temperature, top_p, max_token)

        nova_result = json.loads(nova_response.get("output").get(
            "message").get("content")[0].get("text"))

        return {
            'err_no': 0,
            'err_msg': '',
            'data': nova_result
        }
    except Exception as e:
        # raise e
        return {
            'err_no': 1,
            'err_msg': str(e),
            'data': {}
        }
    finally:
        local_dir = os.path.dirname(local_video_path)
        shutil.rmtree(local_dir)


if __name__ == "__main__":
    event = {
        # 'video_s3_uri': 's3://mybucket/test.mp4',
        'video_url': 'https://example.com/test.mp4',
    }
    context = {}
    r = handler(event, context)
    print(r)
