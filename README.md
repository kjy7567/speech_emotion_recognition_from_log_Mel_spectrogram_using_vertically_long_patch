# speech emotion recognition from log-Mel spectrogram using vertically long patch
This repo is the official implementation of ["Accuracy Enhancement Method for Speech Emotion Recognition from Spectrogram using Temporal Frequency Correlation and Positional Information Learning through Knowledge Transfer"](https://doi.org/10.1109/ACCESS.2024.3447770).
<img src="https://github.com/kjy7567/speech_emotion_recognition_from_log_Mel_spectrogram_using_vertically_long_patch/blob/main/fig/overall_process.png"/>
# How to load model
I saved the model as python dict() format like below:
```python
torch.save({
            'model_state_dict': model.state_dict(),
            'CE': cross_entropy_loss
            'L1': L1_loss
            ...
            }, PATH)
```
So, you can load the pretrained weight like below:
```python
# model.load_state_dict(torch.load(PATH_WEIGHT_FILE)['model_state_dict'])
model.load_state_dict(torch.load('./weight/teacher_92.64_CREMA_D.ckpt')['model_state_dict'])
```
<img src="https://github.com/kjy7567/speech_emotion_recognition_from_log_Mel_spectrogram_using_vertically_long_patch/blob/main/fig/attention_mask.png" width="50%" height="50%" />
