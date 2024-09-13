# speech emotion recognition from log-Mel spectrogram using vertically long patch
This repo is the official implementation of ["Accuracy Enhancement Method for Speech Emotion Recognition from Spectrogram using Temporal Frequency Correlation and Positional Information Learning through Knowledge Transfer"](https://doi.org/10.1109/ACCESS.2024.3447770).

<img src="https://github.com/kjy7567/speech_emotion_recognition_from_log_Mel_spectrogram_using_vertically_long_patch/blob/main/fig/overall_process.png"/>

```python
from model import Teacher, Student

teacher = Teacher(
    image_size = img_size,
    patch_size = patch_size,
    num_classes = num_classes,
    dim = 256,
    depth = 6,
    heads = 5,
    mlp_dim = 256,
    dropout = 0.,
    emb_dropout = 0.,
    channels = 1,
    max_bs = batch_size
)

student = Student(
    image_size = img_size,
    patch_size = patch_size,
    num_classes = num_classes,
    dim = 256,
    depth = 3,
    heads = 5,
    mlp_dim = 256,
    dropout = 0.,
    emb_dropout = 0.,
    channels = 1,
    max_bs = batch_size
)
```

<img src="https://github.com/kjy7567/speech_emotion_recognition_from_log_Mel_spectrogram_using_vertically_long_patch/blob/main/fig/attention_mask.png" width="50%" height="50%" />
