### Video understanding
Use OpenAI GPT-5 models to interpret scenes from a movie and perform summarization
## Implementation
Image frames are split from videos and sent to the GPT-5 model to be interpreted when a scene change is detected.

A scene change is detected when the histogram between frames has a difference beyond a threshold. The key is the cv2.HISTCMP_BHATTACHARYYA metric used when doing compareHist()

The individual frames descriptions are concatenated and sent to the GPT-5 model where it performs a summarization over all the frame descriptions

## Future work

Experiment with different thresholds

1. Try batching frames together with one call
2. Try performing intermediate summarizations on frame descriptions, especially if you have a lot of frames
3. Try different models for frame descriptions and summarization
4. Try different platforms ( Groq, Lambda, etc... ) for speed and throughput
