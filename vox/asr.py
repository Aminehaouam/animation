import json, numpy as np, soundfile as sf, sherpa_onnx
d='/tmp/asr/'
rec=sherpa_onnx.OfflineRecognizer.from_whisper(encoder=d+'sherpa-onnx-whisper-base.en/base.en-encoder.int8.onnx',decoder=d+'sherpa-onnx-whisper-base.en/base.en-decoder.int8.onnx',tokens=d+'sherpa-onnx-whisper-base.en/base.en-tokens.txt',language='en',task='transcribe',num_threads=4)
vc=sherpa_onnx.VadModelConfig(); vc.silero_vad.model=d+'silero_vad.onnx'; vc.silero_vad.min_silence_duration=0.12; vc.silero_vad.min_speech_duration=0.2; vc.sample_rate=16000
vad=sherpa_onnx.VoiceActivityDetector(vc,buffer_size_in_seconds=60)
a,sr=sf.read(d+'trailer16k.wav',dtype='float32'); out=[]
w=512
def flush():
    while not vad.empty():
        seg=vad.front; st=seg.start/16000; s=np.array(seg.samples)
        stream=rec.create_stream(); stream.accept_waveform(16000,s); rec.decode_stream(stream)
        out.append({'start':round(st,2),'end':round(st+len(s)/16000,2),'text':stream.result.text.strip()}); vad.pop()
for i in range(0,len(a),w):
    vad.accept_waveform(a[i:i+w]); flush()
vad.flush(); flush()
json.dump(out,open('/home/user/animation/vox/build/transcript_fine.json','w'),indent=1)
for o in out: print(o['start'],o['end'],o['text'])
