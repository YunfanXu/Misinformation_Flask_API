# Deploying a Detection-of-fake-face Model with Flask

## Predicting image categories with a REST API

```
curl localhost:5000/image -F file=@capuchon.jpg
```

Gives : 

```
{
    "label": "Fake",
    "score": 99.99415874481201
}
```