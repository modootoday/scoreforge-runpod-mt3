# scoreforge-runpod-mt3

Google Magenta의 MT3 모델을 사용한 **다중 악기 음악 전사** RunPod Serverless 워커

## 특징

- **오케스트라급 전사**: 피아노, 현악기, 관악기, 금관악기, 타악기 등 128개 General MIDI 악기 지원
- **폴리포닉 처리**: 동시에 연주되는 여러 악기/음표 인식
- **악기별 트랙 분리**: 각 악기별로 노트를 분리하여 출력
- **MIDI 출력**: 전사 결과를 MIDI 파일로 생성 가능

## 지원 악기 (일부)

| 카테고리 | 악기 |
|----------|------|
| 피아노 | Acoustic Grand Piano, Electric Piano, Harpsichord |
| 현악기 | Violin, Viola, Cello, Contrabass, String Ensemble |
| 목관 | Flute, Oboe, Clarinet, Bassoon, Piccolo |
| 금관 | Trumpet, Trombone, French Horn, Tuba |
| 타악기 | Timpani, Xylophone, Vibraphone, Marimba |
| 기타 | Acoustic Guitar, Electric Guitar, Bass |

## 모델

| 모델 | 설명 | 용도 |
|------|------|------|
| `mt3` | Multi-instrument | 오케스트라, 밴드, 앙상블 |
| `ismir2021` | Piano-only (127 velocity bins) | 피아노 솔로 (고정밀) |

## 배포 방법

### 1. Docker 이미지 빌드

```bash
cd packages/scoreforge-runpod-mt3
docker build -t scoreforge-mt3:latest .
```

### 2. Docker Hub에 푸시

```bash
docker tag scoreforge-mt3:latest <your-dockerhub>/scoreforge-mt3:latest
docker push <your-dockerhub>/scoreforge-mt3:latest
```

### 3. RunPod 엔드포인트 생성

1. [RunPod Console](https://www.runpod.io/console/serverless) 접속
2. "New Endpoint" 클릭
3. Docker 이미지: `<your-dockerhub>/scoreforge-mt3:latest`
4. GPU 타입: **RTX 3090** 이상 (24GB VRAM 권장)
5. 환경 변수 설정:
   - `SUPABASE_URL`: Supabase 프로젝트 URL
   - `SUPABASE_SERVICE_ROLE_KEY`: Supabase 서비스 키

## API

### 요청

```json
{
  "input": {
    "audio_url": "https://example.com/orchestra.mp3",
    "model_type": "mt3",
    "output_midi": true,
    "storage_bucket": "transcriptions",
    "storage_prefix": "mt3"
  }
}
```

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `audio_url` | string | (필수) | 오디오 파일 URL |
| `model_type` | string | `"mt3"` | 모델 타입 (`mt3` 또는 `ismir2021`) |
| `output_midi` | boolean | `false` | MIDI 파일 생성 여부 |
| `storage_bucket` | string | `"transcriptions"` | Supabase 버킷 |
| `storage_prefix` | string | `"mt3"` | 저장 경로 prefix |

### 응답

```json
{
  "notes": [
    {
      "pitch": 60,
      "startTime": 0.5,
      "duration": 0.25,
      "velocity": 80,
      "instrument": 40,
      "instrumentName": "Violin"
    },
    {
      "pitch": 48,
      "startTime": 0.5,
      "duration": 0.5,
      "velocity": 70,
      "instrument": 42,
      "instrumentName": "Cello"
    }
  ],
  "note_count": 1250,
  "instruments": {
    "40": "Violin",
    "41": "Viola",
    "42": "Cello",
    "0": "Acoustic Grand Piano"
  },
  "tracks": [
    {
      "program": 40,
      "name": "Violin",
      "notes": [...]
    },
    {
      "program": 42,
      "name": "Cello",
      "notes": [...]
    }
  ],
  "midi_url": "https://xxx.supabase.co/storage/v1/object/public/transcriptions/mt3/abc123/transcription.mid"
}
```

## 기술 스택

- **JAX + CUDA**: GPU 가속 추론
- **T5X**: Google의 Transformer 프레임워크
- **MT3**: Multi-Task Multitrack Music Transcription 모델

## 참고 자료

- [MT3 GitHub](https://github.com/magenta/mt3)
- [MT3 논문 (arXiv)](https://arxiv.org/abs/2111.03017)
- [Google Magenta](https://magenta.tensorflow.org/)

## 라이선스

MIT License
