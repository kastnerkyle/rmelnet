# rmelnet
Experimental dump of R-MelNet related code and demo files

See samples/melnet\_trunc\_pt33 for samples from the R-MelNet pipeline. tts\*.wav files represent the initial tts (generated via hts) that were used to extract the initial pronunciation / phonemization of the text. raw\*.wav files are the output from the model with priming trimmed, and cut off based on the attention termination.

concat.wav contains the combination of all the raw files, using the command. It is useful for hearing variability across samples.
```
ffmpeg -f concat -safe 0 -i <( for f in $(ls */raw*.wav | sort -n -t "_" -k2); do echo "file '$(pwd)/$f'"; done ) output.wav
```

Baseline comparisons for fastspeech2 and portaspeech were generated from their huggingface spaces, at https://huggingface.co/facebook/fastspeech2-en-ljspeech and https://huggingface.co/spaces/NATSpeech/PortaSpeech respectively. 

An experimental code dump will be released soon, followed by a pretrained model and notebook demo for sampling.
