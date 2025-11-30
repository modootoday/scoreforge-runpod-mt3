# scoreforge-runpod-mt3

Google Magenta의 MT3 모델을 사용한 다중 악기 음악 전사 RunPod Serverless 워커

> ⚠️ **주의:** MT3는 복잡한 JAX/T5X 설정이 필요합니다. 현재 플레이스홀더 구현입니다.

## 상태

이 패키지는 현재 개발 중입니다. MT3 모델은 다음이 필요합니다:
- JAX with GPU support
- T5X framework
- Pre-trained MT3 checkpoints

## 배포 방법 (준비 중)

### RunPod GitHub 연동

1. [RunPod Console](https://www.runpod.io/console/serverless) 접속
2. "New Endpoint" → "GitHub Repo" 선택
3. `modootoday/scoreforge-runpod-mt3` 레포 연결
4. GPU 타입: **RTX 3090** 이상 선택
5. 배포 완료 후 Endpoint URL 복사

## API (예정)

### 요청

```json
{
  "input": {
    "audio_url": "https://example.com/audio.mp3",
    "model_type": "mt3"
  }
}
```

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `audio_url` | string | (필수) | 오디오 파일 URL |
| `model_type` | string | "mt3" | 모델 타입 ("mt3" 또는 "ismir2021") |

### 응답

```json
{
  "notes": [
    {
      "pitch": 60,
      "startTime": 0.5,
      "duration": 0.25,
      "velocity": 80,
      "instrument": 0
    }
  ],
  "note_count": 150,
  "instruments": [0, 25, 40]
}
```

## 대안

MT3 설정이 완료되기 전까지는 [Basic-Pitch](../scoreforge-runpod-basic-pitch)를 사용하세요.

## 라이선스

MIT License
