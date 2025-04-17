import subprocess
import json
import re
import os
import sys
from datetime import timedelta


class VideoQualityChecker:
    def __init__(self, video_path):
        """Initialize video quality checker"""
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
        """Execute all checks"""
        # self.check_format()
        # Even if there are format issues, still try to detect black screens and freezes
        self.check_black_frames()
        self.check_freezes()
        # self.check_audio()
        return self.get_report()

    def check_format(self):
        """Check if video format is valid"""
        # First check if file exists
        if not os.path.exists(self.video_path):
            self.format_valid = False
            self.format_errors.append(
                f"File does not exist: {self.video_path}")
            return

        # Use FFmpeg to detect format errors
        cmd = [
            'ffmpeg',
            '-v', 'error',  # Only output error messages
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

        # Check common error patterns
        errors = []

        # Check timestamp errors
        timestamp_errors = re.findall(
            r'(Non-monotonous DTS|DTS.*out of order|Invalid DTS|Invalid PTS|PTS.*DTS.*invalid)', stderr)
        if timestamp_errors:
            errors.append(f"Timestamp errors: {', '.join(timestamp_errors)}")

        # Check truncated file errors
        if re.search(r'(Truncating|truncated|unexpected end of file|End of file|incomplete frame)', stderr, re.IGNORECASE):
            errors.append("File may be truncated or incomplete")

        # Check decoding errors
        decode_errors = re.findall(
            r'(Error while decoding|decode|Invalid data|corrupt|corruption)', stderr, re.IGNORECASE)
        if decode_errors:
            errors.append(f"Decoding errors: {', '.join(decode_errors)}")

        # Check no stream errors
        if "no streams" in stderr.lower() or "could not find" in stderr.lower():
            errors.append("No valid media streams found")

        # Check audio/video sync issues
        if "Audio and video streams" in stderr and "not correctly synchronized" in stderr:
            errors.append("Audio and video synchronization issues")

        # Collect all errors
        if errors:
            self.format_valid = False
            self.format_errors.extend(errors)
            if stderr and not any(err in stderr for err in errors):
                self.format_errors.append(f"Other errors: {stderr}")
        elif stderr:
            self.format_valid = False
            self.format_errors.append(f"Unclassified errors: {stderr}")

        # Even if there are errors, try to get media info
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

                # Get video stream info
                video_stream = None
                for stream in self.video_info.get("streams", []):
                    if stream.get("codec_type") == "video":
                        video_stream = stream
                        break

                if video_stream is not None:
                    # Get frame rate
                    frame_rate = video_stream.get(
                        "r_frame_rate", "").split("/")
                    if len(frame_rate) == 2 and float(frame_rate[1]) > 0:
                        self.frame_rate = float(
                            frame_rate[0]) / float(frame_rate[1])

                    # Get video duration
                    self.duration = float(self.video_info.get(
                        "format", {}).get("duration", 0))

                    # If no errors found, consider format valid
                    if not self.format_errors:
                        self.format_valid = True
                else:
                    if not self.format_errors:
                        self.format_valid = False
                        self.format_errors.append(
                            "No valid video stream found")
            else:
                if not self.format_errors:
                    self.format_valid = False
                    self.format_errors.append(
                        "FFprobe failed to parse video file")

        except Exception as e:
            if not self.format_errors:
                self.format_valid = False
                self.format_errors.append(
                    f"Error detecting video format: {str(e)}")

    def check_black_frames(self, threshold=0.98, min_duration=2.0):
        """Detect black screen sections in video (continuous for more than min_duration seconds)"""
        try:
            # Use FFmpeg's blackdetect filter to detect black screens
            cmd = [
                "ffmpeg", "-i", self.video_path,
                "-vf", f"blackdetect=d={min_duration}:pic_th={threshold}",
                "-f", "null", "-"
            ]

            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # Parse blackdetect output
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
            print(f"Error detecting black frames: {e}")

    def check_freezes(self, noise=0.001, min_duration=0.1):
        """Detect video freezes (dropped/stuck frames)

        Parameters:
        noise -- noise tolerance, lower value means more sensitive detection, default 0.001
        min_duration -- minimum freeze duration (seconds), default 0.1 seconds
        """
        try:
            # Use FFmpeg's freezedetect filter to detect freezes
            # Correct parameters are n(noise) and d(duration)
            cmd = [
                "ffmpeg", "-i", self.video_path,
                "-vf", f"freezedetect=n={noise}:d={min_duration}",
                "-f", "null", "-"
            ]

            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # Parse freezedetect output
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
            print(f"Error detecting freezes: {e}")

    def check_audio(self):
        """Check if video has audio stream and its quality"""
        try:
            # First check if there is an audio stream
            if self.video_info:
                audio_streams = [stream for stream in self.video_info.get("streams", [])
                                 if stream.get("codec_type") == "audio"]

                if not audio_streams:
                    self.has_audio = False
                    self.audio_issues.append("Video has no audio stream")
                    return

                self.has_audio = True

                # Check if audio is completely muted or volume is too low
                cmd = [
                    "ffmpeg", "-i", self.video_path,
                    "-af", "volumedetect",
                    "-f", "null", "-"
                ]

                result = subprocess.run(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                # # Parse volume detection results
                # mean_volume_match = re.search(r"mean_volume: ([-\d.]+) dB", result.stderr)
                # max_volume_match = re.search(r"max_volume: ([-\d.]+) dB", result.stderr)

                # if mean_volume_match and max_volume_match:
                #     mean_volume = float(mean_volume_match.group(1))
                #     max_volume = float(max_volume_match.group(1))

                #     # If max volume is very low, video might be muted
                #     if max_volume <= -50:
                #         self.audio_issues.append(f"Video may be muted (max volume: {max_volume} dB)")
                #     # If average volume is very low, there might be audio issues
                #     elif mean_volume <= -35:
                #         self.audio_issues.append(f"Video volume may be too low (average volume: {mean_volume} dB)")

                # Check if audio has noise or distortion
                # This requires more complex audio analysis, this is just a simple example
                # Can be expanded as needed
            else:
                self.has_audio = False
                self.audio_issues.append(
                    "Cannot detect audio stream (video info unavailable)")

        except Exception as e:
            print(f"Error detecting audio: {e}")
            self.audio_issues.append(f"Audio detection error: {str(e)}")

    def format_time(self, seconds):
        """Format time as HH:MM:SS.mmm"""
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{int(td.microseconds / 1000):03d}"

    def get_report(self):
        """Generate video quality detection report"""
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

        # Add format errors
        # if self.format_errors:
        #     for error in self.format_errors:
        #         report["issues"].append({
        #             "type": "format",
        #             "description": error
        #         })

        # Add black screen issues
        for black in self.black_frames:
            report["issues"].append({
                "type": "black_frame",
                "start_time": self.format_time(black["start"]),
                "end_time": self.format_time(black["end"]),
                "duration": round(black["duration"], 2),
                "description": f"Black screen detected: {self.format_time(black['start'])} - {self.format_time(black['end'])} (duration: {round(black['duration'], 2)} seconds)"
            })

        # Add freeze issues
        for freeze in self.freezes:
            report["issues"].append({
                "type": "freeze",
                "start_time": self.format_time(freeze["start"]),
                "end_time": self.format_time(freeze["end"]),
                "duration": round(freeze["duration"], 2),
                "description": f"Freeze detected: {self.format_time(freeze['start'])} - {self.format_time(freeze['end'])} (duration: {round(freeze['duration'], 2)} seconds)"
            })

        # Add audio issues
        # if not self.has_audio:
        #     report["issues"].append({
        #         "type": "audio",
        #         "description": "Video has no audio track"
        #     })

        for issue in self.audio_issues:
            report["issues"].append({
                "type": "audio",
                "description": issue
            })

        return report


def check_video_quality(video_path):
    """Check video quality and output report"""
    checker = VideoQualityChecker(video_path)
    report = checker.check_all()

    # print(f"Video Quality Initial Report - {os.path.basename(video_path)}")
    # print("-" * 60)

    # print(
    #     f"Format check: {'✅ Valid' if report['format_valid'] else '❌ Invalid'}")
    # print(
    #     f"Audio check: {'✅ Has audio' if report['has_audio'] else '❌ No audio'}")

    # if "duration" in report:
    #     print(f"Duration: {timedelta(seconds=report['duration'])}")

    # if "frame_rate" in report:
    #     print(f"Frame rate: {report['frame_rate']:.2f} fps")

    # print("-" * 60)

    if not report["issues"]:
        # print("✅ No quality issues detected")
        pass
    else:
        # Group issues by type
        format_issues = [issue for issue in report["issues"]
                         if issue["type"] == "format"]
        black_issues = [issue for issue in report["issues"]
                        if issue["type"] == "black_frame"]
        freeze_issues = [issue for issue in report["issues"]
                         if issue["type"] == "freeze"]
        audio_issues = [issue for issue in report["issues"]
                        if issue["type"] == "audio"]

        print(f"⚠️ Detected {len(report['issues'])} quality issues:")

        if format_issues:
            print("\n[Format Issues]")
            for i, issue in enumerate(format_issues, 1):
                print(f"{i}. {issue['description']}")

        if black_issues:
            print("\n[Black Screen Issues]")
            for i, issue in enumerate(black_issues, 1):
                print(f"{i}. {issue['description']}")

        if freeze_issues:
            print("\n[Freeze Issues]")
            for i, issue in enumerate(freeze_issues, 1):
                print(f"{i}. {issue['description']}")

        if audio_issues:
            print("\n[Audio Issues]")
            for i, issue in enumerate(audio_issues, 1):
                print(f"{i}. {issue['description']}")

    return report


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python VQC.py <video file path>")
        sys.exit(1)

    video_path = sys.argv[1]
    check_video_quality(video_path)
