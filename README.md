# Backend for Detection-of-fake-face 

## Run the server on your machine

```
python app.py
```

## Predicting image with a REST API via POST method

```
curl -F "file=@path/to/local/file/example.png" http://127.0.0.1:5000/testImage
```

Gives : 

```
{
    "label": "Fake",
    "score": 99.99645948410034
}
```


## Predicting video with a REST API

```
curl -F "file=@path/to/local/file/exampleVideo.mp4" http://127.0.0.1:5000/testVideo
```

Gives the id of the video which can be found under 'test_video_results' folder: 

```
{
    "videoId": "tmpp8072483"
}
```