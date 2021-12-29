# Backend for Detection-of-fake-face 

## Run the server on your machine

```
python app.py
```

## Predicting image with a REST API via POST method

The input image should be in base64 format: 

```
curl -F "file=@path/to/local/file/example.png" http://127.0.0.1:5000/queryImage
```
### Response data to Front-end

```
{
    "label": "Fake",
    "score": 99.99645948410034
}
```

### Server output
```
This image could be  Real  and the possibility is 99.99010562896729
127.0.0.1 - - [29/Dec/2021 15:18:19] "POST /queryImage HTTP/1.1" 200 -
```

## Predicting video with a REST API
Gives the id of the video which can be found under 'test_video_results' folder: 
```
curl -F "file=@path/to/local/file/exampleVideo.mp4" http://127.0.0.1:5000/queryVideo
```

### Response data to Front-end

Return the id of the tested video.
```
{
    "videoId": "tmpp8072483"
}
```


### Server output:
```
fps = 30.0
number of frames = 414
duration (S) = 13.8
duration (S) = 138.0
Number of frames complted:10
Number of frames complted:20
Number of frames complted:30
Number of frames complted:40
Number of frames complted:50
Number of frames complted:60
Number of frames complted:70
Number of frames complted:80
Number of frames complted:90
Number of frames complted:100
Number of frames complted:110
Number of frames complted:120
Number of frames complted:130
videoId:20211229-152806.mp4
127.0.0.1 - - [29/Dec/2021 15:28:49] "POST /queryVideo HTTP/1.1" 200 -
```

### Tested video example
![demo](./assets/demo/example.mp4)
