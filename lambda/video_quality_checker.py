import subprocess
import json
import re
import os
import sys
from datetime import timedelta


class VideoQualityChecker:
    def __init__(self, video_path):
        """初始化视频质量检查器"""
        self.video_path = video_path
        self.video_info = None
        self.frame_rate = None
        self.duration = None
        self.format_valid = False
        self.format_errors = []
        self.black_frames = []
        self.freezes = []
        self.has_audio = False
        self.audio_issues = []

    def check_all(self):
        """执行所有检查"""
        self.check_format()
        # 即使有格式问题，也尝试检测黑屏和卡顿
        self.check_black_frames()
        self.check_freezes()
        self.check_audio()
        return self.get_report()

    def check_format(self):
        """检查视频格式是否有效"""
        # 首先检查文件是否存在
        if not os.path.exists(self.video_path):
            self.format_valid = False
            self.format_errors.append(f"文件不存在: {self.video_path}")
            return

        # 使用FFmpeg检测格式错误
        cmd = [
            'ffmpeg',
            '-v', 'error',  # 只输出错误信息
            '-i', self.video_path,
            '-f', 'null',
            '-'
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        stdout, stderr = process.communicate()

        # 检查常见错误模式
        errors = []

        # 检查时间戳错误
        timestamp_errors = re.findall(
            r'(Non-monotonous DTS|DTS.*out of order|Invalid DTS|Invalid PTS|PTS.*DTS.*invalid)', stderr)
        if timestamp_errors:
            errors.append(f"时间戳错误: {', '.join(timestamp_errors)}")

        # 检查截断文件错误
        if re.search(r'(Truncating|truncated|unexpected end of file|End of file|incomplete frame)', stderr, re.IGNORECASE):
            errors.append("文件可能被截断或不完整")

        # 检查解码错误
        decode_errors = re.findall(
            r'(Error while decoding|decode|Invalid data|corrupt|corruption)', stderr, re.IGNORECASE)
        if decode_errors:
            errors.append(f"解码错误: {', '.join(decode_errors)}")

        # 检查无流错误
        if "no streams" in stderr.lower() or "could not find" in stderr.lower():
            errors.append("未找到有效的媒体流")

        # 检查音视频同步问题
        if "Audio and video streams" in stderr and "not correctly synchronized" in stderr:
            errors.append("音视频同步问题")

        # 收集所有错误
        if errors:
            self.format_valid = False
            self.format_errors.extend(errors)
            if stderr and not any(err in stderr for err in errors):
                self.format_errors.append(f"其他错误: {stderr}")
        elif stderr:
            self.format_valid = False
            self.format_errors.append(f"未分类错误: {stderr}")

        # 即使有错误，也尝试获取媒体信息
        try:
            cmd = [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format", "-show_streams",
                self.video_path
            ]
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            if result.returncode == 0:
                self.video_info = json.loads(result.stdout)

                # 获取视频流信息
                video_stream = None
                for stream in self.video_info.get("streams", []):
                    if stream.get("codec_type") == "video":
                        video_stream = stream
                        break

                if video_stream is not None:
                    # 获取帧率
                    frame_rate = video_stream.get(
                        "r_frame_rate", "").split("/")
                    if len(frame_rate) == 2 and float(frame_rate[1]) > 0:
                        self.frame_rate = float(
                            frame_rate[0]) / float(frame_rate[1])

                    # 获取视频时长
                    self.duration = float(self.video_info.get(
                        "format", {}).get("duration", 0))

                    # 如果没有发现错误，则认为格式有效
                    if not self.format_errors:
                        self.format_valid = True
                else:
                    if not self.format_errors:
                        self.format_valid = False
                        self.format_errors.append("未找到有效的视频流")
            else:
                if not self.format_errors:
                    self.format_valid = False
                    self.format_errors.append("FFprobe无法解析视频文件")

        except Exception as e:
            if not self.format_errors:
                self.format_valid = False
                self.format_errors.append(f"检测视频格式时出错: {str(e)}")

    def check_black_frames(self, threshold=0.98, min_duration=2.0):
        """检测视频中的黑屏部分（连续超过min_duration秒）"""
        try:
            # 使用FFmpeg的blackdetect过滤器检测黑屏
            cmd = [
                "ffmpeg", "-i", self.video_path,
                "-vf", f"blackdetect=d={min_duration}:pic_th={threshold}",
                "-f", "null", "-"
            ]

            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # 解析blackdetect输出
            pattern = r"blackdetect.*black_start:(\d+\.?\d*).*black_end:(\d+\.?\d*).*black_duration:(\d+\.?\d*)"
            matches = re.findall(pattern, result.stderr)

            self.black_frames = []
            for match in matches:
                start_time = float(match[0])
                end_time = float(match[1])
                duration = float(match[2])

                if duration >= min_duration:
                    self.black_frames.append({
                        "start": start_time,
                        "end": end_time,
                        "duration": duration
                    })

        except Exception as e:
            print(f"检测黑屏时出错: {e}")

    def check_freezes(self, noise=0.001, min_duration=0.1):
        """检测视频中的卡顿部分（丢帧/卡帧）

        参数:
        noise -- 噪声容忍度，值越小检测越敏感，默认0.001
        min_duration -- 最小卡顿持续时间（秒），默认0.5秒
        """
        try:
            # 使用FFmpeg的freezedetect过滤器检测卡顿
            # 正确的参数是n(noise)和d(duration)
            cmd = [
                "ffmpeg", "-i", self.video_path,
                "-vf", f"freezedetect=n={noise}:d={min_duration}",
                "-f", "null", "-"
            ]

            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # 解析freezedetect输出
            pattern = r"freeze_start:(\d+\.?\d*).*freeze_duration:(\d+\.?\d*).*freeze_end:(\d+\.?\d*)"
            matches = re.findall(pattern, result.stderr)

            self.freezes = []
            for match in matches:
                start_time = float(match[0])
                duration = float(match[1])
                end_time = float(match[2])

                self.freezes.append({
                    "start": start_time,
                    "end": end_time,
                    "duration": duration
                })

        except Exception as e:
            print(f"检测卡顿时出错: {e}")

    def check_audio(self):
        """检查视频是否有音频流以及音频质量"""
        try:
            # 首先检查是否有音频流
            if self.video_info:
                audio_streams = [stream for stream in self.video_info.get("streams", [])
                                 if stream.get("codec_type") == "audio"]

                if not audio_streams:
                    self.has_audio = False
                    self.audio_issues.append("视频没有音频流")
                    return

                self.has_audio = True

                # 检查音频是否全静音或音量过低
                cmd = [
                    "ffmpeg", "-i", self.video_path,
                    "-af", "volumedetect",
                    "-f", "null", "-"
                ]

                result = subprocess.run(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                # # 解析音量检测结果
                # mean_volume_match = re.search(r"mean_volume: ([-\d.]+) dB", result.stderr)
                # max_volume_match = re.search(r"max_volume: ([-\d.]+) dB", result.stderr)

                # if mean_volume_match and max_volume_match:
                #     mean_volume = float(mean_volume_match.group(1))
                #     max_volume = float(max_volume_match.group(1))

                #     # 如果最大音量非常低，可能是静音视频
                #     if max_volume <= -50:
                #         self.audio_issues.append(f"视频可能是静音的 (最大音量: {max_volume} dB)")
                #     # 如果平均音量非常低，可能有音频问题
                #     elif mean_volume <= -35:
                #         self.audio_issues.append(f"视频音量可能过低 (平均音量: {mean_volume} dB)")

                # 检查音频是否有噪音或失真
                # 这需要更复杂的音频分析，这里只是一个简单的示例
                # 可以根据需要扩展这部分
            else:
                self.has_audio = False
                self.audio_issues.append("无法检测音频流 (视频信息不可用)")

        except Exception as e:
            print(f"检测音频时出错: {e}")
            self.audio_issues.append(f"音频检测错误: {str(e)}")

    def format_time(self, seconds):
        """格式化时间为 HH:MM:SS.mmm"""
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{int(td.microseconds / 1000):03d}"

    def get_report(self):
        """生成视频质量检测报告"""
        report = {
            "video_path": self.video_path,
            "format_valid": self.format_valid,
            "has_audio": self.has_audio,
            "issues": []
        }

        if self.frame_rate is not None:
            report["frame_rate"] = self.frame_rate

        if self.duration is not None:
            report["duration"] = self.duration

        # 添加格式错误
        if self.format_errors:
            for error in self.format_errors:
                report["issues"].append({
                    "type": "format",
                    "description": error
                })

        # 添加黑屏问题
        for black in self.black_frames:
            report["issues"].append({
                "type": "black_frame",
                "start_time": self.format_time(black["start"]),
                "end_time": self.format_time(black["end"]),
                "duration": round(black["duration"], 2),
                "description": f"检测到黑屏: {self.format_time(black['start'])} - {self.format_time(black['end'])} (持续 {round(black['duration'], 2)} 秒)"
            })

        # 添加卡顿问题
        for freeze in self.freezes:
            report["issues"].append({
                "type": "freeze",
                "start_time": self.format_time(freeze["start"]),
                "end_time": self.format_time(freeze["end"]),
                "duration": round(freeze["duration"], 2),
                "description": f"检测到卡顿: {self.format_time(freeze['start'])} - {self.format_time(freeze['end'])} (持续 {round(freeze['duration'], 2)} 秒)"
            })

        # 添加音频问题
        if not self.has_audio:
            report["issues"].append({
                "type": "audio",
                "description": "视频没有音频轨道"
            })

        for issue in self.audio_issues:
            report["issues"].append({
                "type": "audio",
                "description": issue
            })

        return report


def check_video_quality(video_path):
    """检查视频质量并输出报告"""
    checker = VideoQualityChecker(video_path)
    report = checker.check_all()

    print(f"视频质量初步检测报告 - {os.path.basename(video_path)}")
    print("-" * 60)

    print(f"格式检查: {'✅ 正常' if report['format_valid'] else '❌ 异常'}")
    print(f"音频检查: {'✅ 有音频' if report['has_audio'] else '❌ 无音频'}")

    if "duration" in report:
        print(f"时长: {timedelta(seconds=report['duration'])}")

    if "frame_rate" in report:
        print(f"帧率: {report['frame_rate']:.2f} fps")

    print("-" * 60)

    if not report["issues"]:
        print("✅ 未检测到质量问题")
    else:
        # 按类型分组显示问题
        format_issues = [issue for issue in report["issues"]
                         if issue["type"] == "format"]
        black_issues = [issue for issue in report["issues"]
                        if issue["type"] == "black_frame"]
        freeze_issues = [issue for issue in report["issues"]
                         if issue["type"] == "freeze"]
        audio_issues = [issue for issue in report["issues"]
                        if issue["type"] == "audio"]

        print(f"⚠️ 检测到 {len(report['issues'])} 个质量问题:")

        if format_issues:
            print("\n【格式问题】")
            for i, issue in enumerate(format_issues, 1):
                print(f"{i}. {issue['description']}")

        if black_issues:
            print("\n【黑屏问题】")
            for i, issue in enumerate(black_issues, 1):
                print(f"{i}. {issue['description']}")

        if freeze_issues:
            print("\n【卡顿问题】")
            for i, issue in enumerate(freeze_issues, 1):
                print(f"{i}. {issue['description']}")

        if audio_issues:
            print("\n【音频问题】")
            for i, issue in enumerate(audio_issues, 1):
                print(f"{i}. {issue['description']}")

    return report


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python VQC.py <视频文件路径>")
        sys.exit(1)

    video_path = sys.argv[1]
    check_video_quality(video_path)
