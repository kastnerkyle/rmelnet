# rmelnet
Experimental dump of R-MelNet related code and demo files

See the site to listen to samples https://kastnerkyle.github.io/rmelnet/ , alternatively see the instructions below.


Experimental code and model are released, but a directly runnable inference pipeline is still TODO - see raw\_code/ for details


See samples/melnet\_trunc\_pt33 for samples from the R-MelNet pipeline. tts\*.wav files represent the initial tts (generated via hts) that were used to extract the initial pronunciation / phonemization of the text. raw\*.wav files are the output from the model with priming trimmed, and cut off based on the attention termination.

concat.wav contains the combination of all the raw files, using the command. It is useful for hearing variability across samples.
```
ffmpeg -f concat -safe 0 -i <( for f in $(ls */raw*.wav | sort -n -t "_" -k2); do echo "file '$(pwd)/$f'"; done ) output.wav
```

Baseline comparisons for fastspeech2 and portaspeech were generated from their huggingface spaces, at https://huggingface.co/facebook/fastspeech2-en-ljspeech and https://huggingface.co/spaces/NATSpeech/PortaSpeech respectively. 

