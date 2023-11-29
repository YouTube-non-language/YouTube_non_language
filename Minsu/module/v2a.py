from moviepy.editor import VideoFileClip
import os

def extract_audio(video_path):
    # 입력 받은 동영상의 경로를 입력 받고 오디오가 저장될 폴더를 만들어서 추출된 동영상의 이름과 동일한 '.mp3'파일을 만듭니다.
    audio_path = '/'.join(video_path.split('/')[:-2]) + "/audios"
    video_file = video_path.split('/')[-1]
    audio_file = video_file.split('.')[0] + ".mp3"
    audio_ = audio_path + '/' + audio_file

    try:
        # ./audios 디렉토리를 만듭니다.
        os.makedirs(audio_path)
    except FileExistsError:
        pass
    
    # 동영상 파일을 VideoFileClip 객체로 불러옵니다.
    video = VideoFileClip(video_path)
    
    # 동영상 파일로부터 오디오를 추출합니다.
    audio = video.audio
    
    # 오디오를 파일로 출력합니다.
    audio.write_audiofile(audio_)
    
    # VideoFileClip 객체와 연결된 자원을 해제합니다.
    video.close()

video_path = "../videos/오리엔테이션_(고3-기본).mp4"

if __name__ == "__main__" :
    # 동영상 파일에서 오디오를 추출합니다.
    extract_audio(video_path)